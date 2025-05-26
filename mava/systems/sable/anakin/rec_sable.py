# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import time
from functools import partial
from typing import Any, Callable, Dict, Tuple, List

import chex
import flax
import flax.jax_utils
import hydra
import jax
import jax.numpy as jnp
import optax
from optax import MultiSteps

from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict as Params
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import ActorState, EvalActFn, get_eval_fn, get_num_eval_envs
from mava.networks import SableNetwork
from mava.networks.utils.sable import get_init_hidden_state
from mava.systems.ppo.types import PPOTransition as Transition
from mava.systems.sable.types import (
    ActorApply,
    HiddenStates,
    LearnerApply,
)
from mava.systems.sable.types import RecLearnerState as LearnerState
from mava.types import Action, ExperimentOutput, LearnerFn, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import concat_time_and_agents, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
import os
from flax.core import freeze, unfreeze


def get_learner_fn(
    envs: List[MarlEnv],
    apply_fns: Tuple[ActorApply, LearnerApply],
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply functions for executing and training the network.
    sable_action_select_fn, sable_apply_fn = apply_fns
    num_envs = config.arch.num_envs


    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The current model parameters.
                - opt_states (OptState): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - hstates (HiddenStates): The hidden state of the network.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: LearnerState,task_id: int, env: MarlEnv, _: Any
        ) -> Tuple[LearnerState, Tuple[Transition, Metrics]]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep, hstates = learner_state

            # Select action
            key, policy_key = jax.random.split(key)

            # Apply the actor network to get the action, log_prob, value and updated hstates.
            # for all envs -> 64
            last_obs = last_timestep.observation
            # use sable get action method 
            # now i have the action i should take, the value head, the new h_states and the log prob of that action
            action, log_prob, value, hstates = sable_action_select_fn(
                params,
                last_obs,
                hstates,
                policy_key,
                task_id = task_id
            )

            # Step environment -> go to the next step using the action given from the network
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
            # env_state, timestep = env.step(env_state, action)

            # Reset hidden state if done. -> shape bool of num_envs
            done = timestep.last()
            # expand it to 5 dims -> num_envs,1,1,1,1
            done = jnp.expand_dims(done, (1, 2, 3, 4))
            # whereever done is true, we zero out its corresponding hs
            hstates = tree.map(lambda hs: jnp.where(done, jnp.zeros_like(hs), hs), hstates)
            # make the done at the agent level, the shape of prev_done -> [num_envs,num_agents]
            prev_done = last_timestep.last().repeat(env.num_agents).reshape(num_envs, -1)

            # pack the new transition again and return it 
            transition = Transition(
                prev_done, action, value, timestep.reward, log_prob, last_timestep.observation
            )

            # pack the learner state from the new timesteo data to be passed again as a carry
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, hstates)
            return learner_state, (transition, timestep.extras["episode_metrics"])

        #get the info from the learner state created from the learner setup function
        # env_state, last_timestep_old -> created using env.reset()
        # also 5 hstates
        # we have 5 env_states , timesteps, hstates -> one for each task
        # each one of them have [num_envs, ... rest of shapes]
        params, opt_states, key, env_state_old, last_timestep_old, hstates_old = learner_state
        
        # create lists to save the traj_batches, advantages, targets, and the new hstates, env_states and timesteps

        # each traj_batch contains
        # Transition(
        # done    = timestep.last(), 
        # action  = action, -> coming from sable_get_action
        # value   = value, -> coming from sable forward pass in the encoder -> value head
        # reward  = timestep.reward, -> coming from the env after i take the action
        # log_prob= log_prob,-> coming from sable decoder -> needed for the advantages estimation
        # obs     = obs     -> the next obs after taking the action
        # )
        traj_batches_list= []
        advantages_list = []
        targets_list = []
        updated_hstates_list = []
        env_states_list = []
        timesteps_list = []
        episode_metric_list = []

        for i in range(len(envs)):
            # create a new learner state so we step in the env through it
            env_state_i = env_state_old[i]
            last_timestep_i = last_timestep_old[i]
            hstates_i = hstates_old[i]
            new_learner_state = LearnerState(params, opt_states, key, env_state_i, last_timestep_i, hstates_i)
            # env is now global -> leads to jit silent error in the future when the env is different -> need to be baked inside
            env = envs[i]
            # baking env inside the function so it can be jitted
            def _env_step_for_task_i(carry, dummy):
                return _env_step(carry, i, env ,dummy)

            new_learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
                f=_env_step_for_task_i,
                init=new_learner_state,
                xs=None,
                length=config.system.rollout_length,
            )

            # now we have new_learner_state -> the last timestep in the episode
            # traj_batch -> all the trajectory of this batch (parallel envs)
            episode_metric_list.append(episode_metrics)

            # Calculate advantage
            params_new, opt_states_new, key, env_state_new, last_timestep_new, updated_hstates_new = new_learner_state
            env_states_list.append(env_state_new)
            timesteps_list.append(last_timestep_new)
            
            key, last_val_key = jax.random.split(key)
            
            # get the last value using the get action and ignoring all other fields
            _, _, last_val, _ = sable_action_select_fn(  # type: ignore
                params_new, last_timestep_new.observation, updated_hstates_new, last_val_key,task_id=i
            )
            
            # repeat the done to be on the agent level -> new shape -> num_envs, num_agents in that env
            last_done = jax.vmap(lambda x: x.repeat(config.system.num_agents[i], axis=-1))(last_timestep_new.last())

            def _calculate_gae(
                traj_batch: Transition,
                current_val: chex.Array,
                current_done: chex.Array,
            ) -> Tuple[chex.Array, chex.Array]:
                """Calculate the GAE."""

                def _get_advantages(
                    carry: Tuple[chex.Array, chex.Array, chex.Array], transition: Transition
                ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
                    """Calculate the GAE for a single transition."""
                    # each one of these has a shape of num_envs , num_agents -> scalar for each env and each agent
                    gae, next_value, next_done = carry
                    # get these from the traj_batch -> same shape as above
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # GAE calculation
                    gamma = config.system.gamma
                    delta = reward + gamma * next_value * (1 - next_done) - value
                    gae = delta + gamma * config.system.gae_lambda * (1 - next_done) * gae

                    # return the curray and the stack of all gaes
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    # shape of current_val -> last_val -> num_envs,num_agents
                    (jnp.zeros_like(current_val), current_val, current_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            
            # calculate the advantages -> given all the traj_batches for this task and also the last state value and done.
            # shape of advantages -> [rollout_length , num_envs, num_agents] -> for each rollout, each env, each agent will have one adv
            # adv = target - values -> then target = adv + values -> should have the same shape as the envs
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)
            advantages_list.append(advantages)
            targets_list.append(targets)
            traj_batches_list.append(traj_batch)
            updated_hstates_list.append(updated_hstates_new)
            # num_tasks = learner_state.env_state.shape[0]

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple,task_id:int) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_state, key = train_state
                traj_batch, advantages, targets, prev_hstates = batch_info

                def _loss_fn(
                    params: Params,
                    traj_batch: Transition,
                    gae: chex.Array,
                    value_targets: chex.Array,
                    prev_hstates: HiddenStates,
                    rng_key: chex.PRNGKey,
                    task_id: int,
                ) -> Tuple:
                    """Calculate Sable loss."""
                    # Rerun network
                    value, log_prob, entropy = sable_apply_fn(  # type: ignore
                        params,
                        traj_batch.obs,
                        traj_batch.action,
                        prev_hstates,
                        traj_batch.done,
                        task_id,
                        rng_key,
                        
                    )

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    entropy = entropy.mean()

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - value_targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = (
                        actor_loss
                        - config.system.ent_coef * entropy
                        + config.system.vf_coef * value_loss
                    )
                    return total_loss, (actor_loss, entropy, value_loss)
                
                # Calculate loss
                key, entropy_key = jax.random.split(key)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(
                    params,
                    traj_batch,
                    advantages,
                    targets,
                    prev_hstates,
                    entropy_key,
                    task_id
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="batch")
                # pmean over devices.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="device")
                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)
                total_loss, (actor_loss, entropy, value_loss) = loss_info
                loss_info = {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": actor_loss,
                            "entropy": entropy,
                        }


                return (new_params, new_opt_state, key), loss_info



            (params, opt_states, traj_batches_list, advantages_list, targets_list, key, prev_hstates) = update_state

            # Shuffle minibatches
            key, batch_shuffle_key, agent_shuffle_key, entropy_key = jax.random.split(key, 4)

            # Shuffle batch
            # the batch is the num_envs
            batch_size = config.arch.num_envs
            # if batch size was 3 -> random.permutation -> ( 2, 1, 3) for example so we can shuffle the batch using this permutation
            batch_perm = jax.random.permutation(batch_shuffle_key, batch_size)
            prev_hs_minibatch_list = []
            minibatches_list = []
            for i in range(len(traj_batches_list)):
                # collect the batch of the first task
                batch = (traj_batches_list[i], advantages_list[i], targets_list[i])
                # shuffle tha batch along the env dim
                batch = tree.map(lambda x: jnp.take(x, batch_perm, axis=1), batch)

                # Shuffle hidden states along the env dim
                prev_hstates_new = tree.map(lambda x: jnp.take(x, batch_perm, axis=0), prev_hstates[i])

                # Shuffle agents -> create the key to shuffle along the agent dim
                agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents[i])
                # shuffle the batch along the agent dim
                batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=2), batch)

                # Concatenate time and agents
                # after concatinating the rollout and the agent -> shape will be -> [num_env , rollout*num_agents]
                batch = tree.map(concat_time_and_agents, batch)

                # Split into minibatches
                # spilt the batch -> shape (num_envs, rollout*num_agents) to (num_minibatches , num_envs/num_minibatches, rollout*num_agents)
                minibatches = tree.map(
                    lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                    batch,
                )
                # same here , prev_hstates_new -> (num_envs, .. rest of the dims of the hs)
                # not it is reshaped to (num_minibatches, num_envs/num_minibatches, .... rest of the dims of the hs)
                prev_hs_minibatch = tree.map(
                    lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                    prev_hstates_new,
                )
                prev_hs_minibatch_list.append(prev_hs_minibatch)
                minibatches_list.append(minibatches)


            N_minibatches = config.system.num_minibatches
            N_tasks = len(minibatches_list)

            epoch_total_loss_sum = { "total_loss": 0.0, "value_loss": 0.0, "actor_loss": 0.0, "entropy": 0.0 }
            epoch_loss_count = 0

            for i in range(N_minibatches):
                for j in range(N_tasks):
                        # extract the data for the specific task
                        traj_data_for_task, adv_data_for_task, targets_data_for_task = minibatches_list[j]
       
                        # extract the data for the specific minibatch from the data of the specific task
                        
                        # now here we have the current traj data for the task j and mini batch i
                        current_traj_data_mb = tree.map(
                            lambda x: x[i],
                            traj_data_for_task
                        )

                        # same for the advantages and targets
                        current_adv_data_mb = adv_data_for_task[i]
                        current_targets_data_mb = targets_data_for_task[i]
                        
                        # get the hidden state of the task
                        hs_data_for_task = prev_hs_minibatch_list[j]

                        # get the hs of the specific minibatch -> now we have the hs of the task and minibatch
                        current_hs_data_mb = tree.map(
                            lambda x: x[i],
                            hs_data_for_task
                        )
                        # make the data 
                        batch_info_single_mb_task = (current_traj_data_mb, current_adv_data_mb, current_targets_data_mb, current_hs_data_mb)

                        # run the update minibatch function for the single task mini batch data and get the loss and the updated params and opt_state
                        (params, opt_states, key), loss_info_one_task_one_mb = _update_minibatch(
                            (params, opt_states, key), 
                            batch_info_single_mb_task,
                            task_id=j 
                        )             

                        # loop through the keys and the values of the loss -> accumilate all the losses for each minibatch and task
                        for k_loss, v_loss in loss_info_one_task_one_mb.items():
                            epoch_total_loss_sum[k_loss] += v_loss
                        epoch_loss_count += 1 
            # get the final loss value / avg over all the losses
            final_epoch_avg_loss = {k: v / epoch_loss_count for k, v in epoch_total_loss_sum.items() if epoch_loss_count > 0}
            # i am returning update_hstated_list here because of the mismatch of the scan operation  TODO ask ruan about this
            update_state = (params, opt_states, traj_batches_list, advantages_list, targets_list, key, updated_hstates_list)
            return update_state, final_epoch_avg_loss
        
        # until here, i have all the info i need for the update, i have the adv, the targets, the traj_batches, .. everything
        # i need to update the params and opt_state now
        update_state = (params, opt_states, traj_batches_list, advantages_list, targets_list, key, updated_hstates_list)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs # ppo_epochs now = 2 -> expecting 2 at the start of the dims
        )

        params, opt_states, traj_batches_list, advantages_list, targets_list, key, updated_hstates_list = update_state
        learner_state = LearnerState(
            params,
            opt_states,
            key,
            env_states_list,
            timesteps_list,
            updated_hstates_list,
        )
        # TODO trace the episode metrics in the evaluation part
        return learner_state, (episode_metrics, loss_info)


    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The initial model parameters.
                - opt_state (OptState): The initial optimizer state.learner_state
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - hstates (HiddenStates): The initial hidden states of the network.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    envs: List[MarlEnv], keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Callable, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    num_tasks = len(envs)
    # this will be used in the sable network for example -> retention to create a sperate decay matricies 
    all_num_agents = [env.num_agents for env in envs]
    
    # save it in the config (the big one not the task specific one)
    config.system.num_agents = all_num_agents

    # PRNG keys.
    key, net_key = keys
    
    # get the action dims and action space type of each task -> used in sable network 
    task_action_dims = []
    task_action_space_types = []
    for i, env_task in enumerate(envs):

        task_act_dim = env_task.action_dim
        _, task_act_type = get_action_head(env_task.action_spec)

        task_action_dims.append(task_act_dim)
        task_action_space_types.append(task_act_type)

    # i created a list of the chunk sizes in the root config
    # this list will be populated with the num_agents for each task multiplied by the rollout length 
    # this list will be used to determine the chunk size for each task which will vary depending on the number of agents in the task
    for i in range(num_tasks):
        config.network.memory_config.chunk_size.append(config.system.rollout_length * all_num_agents[i])


    # Define network.
    # sable network now will have
    # all_num_agents -> list containing num of agents for each task
    # task_action_dims -> list containing the action dim for each task
    # config.network.net_config -> global network config -> num_blocks, embed_dim, n_heads
    # config.network.memory_config -> global memory cfg -> decay_scaling_factor..etc -> this cfgs also contains the chunck sizes list
    # task_action_space_types -> list containing the action spaces types -> all dicrete for now 
    # num_tasks -> int -> how many tasks we are dealing with right now
    sable_network = SableNetwork(
        all_n_agents=all_num_agents,
        n_agents_per_chunk=all_num_agents,
        task_action_dims=task_action_dims,
        net_config=config.network.net_config,
        memory_config=config.network.memory_config,
        task_action_space_types=task_action_space_types,   
        num_tasks=num_tasks

    )

    # Define optimiser.
    # the op
    lr = make_learning_rate(config.system.actor_lr, config)

    # MultiStep -> wrapper -> taked the inner optimizer and change its behaviour 
    optim = MultiSteps(
        # optax.chain -> it is like have a chain of optimizers -> sequentially executing the first one, then move to the next
        optax.chain(
            # first transformation in the chain -> if the L2 norm of the entire gradient vector exceeds
            # max_grad_norm -> clipping -> prevent exploding gradient 
            optax.clip_by_global_norm(config.system.max_grad_norm),
            # second transformation -> Adam optimizer
            optax.adam(lr, eps=1e-5),
        ),
        # gradient accumilation happens every num_tasks -> the opt will step every n_tasks
        every_k_schedule= len(envs),
        # if true -> first we take the average of the graidents and then step using that avg -> emulate large batch size
        # if false -> gradients for each num_task step will be summed 
        use_grad_mean=True 
    )

 
    # getting the initial observations and hidden states so we can initialize the params of the network
    inti_obs_list = []
    init_hs_list = []
    for idx in range(len(envs)):
        # shape is Observation(agents_view=(10, 64), action_mask=(10, 5), step_count=(10,)) for the first task
        # WARNING, it is different from task to task
        init_obs = envs[idx].observation_spec.generate_value()
        # adding a batch axis 
        # Observation(agents_view=(1, 10, 64), action_mask=(1, 10, 5), step_count=(1, 10))
        init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
        inti_obs_list.append(init_obs)
    
        # HiddenStates(encoder=(64, 1, 4, 64, 64), decoder_self_retn=(64, 1, 4, 64, 64), decoder_cross_retn=(64, 1, 4, 64, 64))
        init_hs = get_init_hidden_state(config.network.net_config, config.arch.num_envs)


        # HiddenStates(encoder=(1, 1, 4, 64, 64), decoder_self_retn=(1, 1, 4, 64, 64), decoder_cross_retn=(1, 1, 4, 64, 64))
        # TODO why this is happening ?
        init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)
        init_hs_list.append(init_hs)

    # Initialise params and optimiser state using the custom function
    params = sable_network.init(
        net_key,
        inti_obs_list,
        init_hs_list,
        net_key,
        method="init_all_tasks",
    )

    # init the optizer chain with the params
    opt_state = optim.init(params)

    # now apply function is a tuple containing 2 functions -> get actions and call 
    apply_fns = (
        partial(sable_network.apply, method="get_actions"),
        sable_network.apply,
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(envs, apply_fns, optim.update, config)
    # replicate learn function across devices
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )

    
    states_list, timesteps_list = [], []
    # calculate the number of keys needed
    # num_envs = 64
    num_envs_per_task_batch = config.arch.get('num_envs', 1) 
    # update_batch_size = 2
    num_update_batches = config.system.get('update_batch_size', 1) 
    # now this code means that we are running 64 env in parralel -> first mini batch
    # then after that run them again in parallel -> second mini batch
    # so overall we have 128 env per task -> 5 tasks means 640 env overall
    # we need a seperate key for each one of these
    total_keys_needed = num_tasks * n_devices * num_update_batches * num_envs_per_task_batch
    key, init_env_key_base = jax.random.split(key)
    all_env_keys = jax.random.split(init_env_key_base, total_keys_needed)
    # why flatten? vmap can not operate on 2 axises, we need to five it one acces to work on
    # all_env_keys.shape[-1] -> size of each key is 2 
    keys_per_task_flat = all_env_keys.reshape(num_tasks, -1, all_env_keys.shape[-1])

    for i,env in enumerate(envs):
        # 128*2 
        task_keys_flat = keys_per_task_flat[i]
        # reset all the parallel envs
        # shapes should be (num_env*update_batch_size , rest of shapes)
        env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(task_keys_flat)
        # reshape each timesteps and env_states in the parallel envs
        # now each of the timesteos and env_states has the shape of  [ (n_devices * num_envs * update_batch_size), ..rest of shapes ]
        # we want to to be [ n_devices , update_batch_size , num_envs , ..rest of shapes ]
        reshape_states = lambda x: x.reshape(
            (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
        )
        # (devices, update batch size, num_envs, ...)
        env_states = tree.map(reshape_states, env_states)
        timesteps = tree.map(reshape_states, timesteps)

        states_list.append(env_states)
        timesteps_list.append(timesteps)

    joint_hstates = []

    for _ in range(len(envs)):
        # get initial hidden state
        init_hstates = get_init_hidden_state(config.network.net_config, config.arch.num_envs)

        joint_hstates.append(init_hstates)

    # generate a unique key for each device and each batch
    key, step_keys = jax.random.split(key)


    # replicate params and opt state through devices
    replicate_learner = (params, opt_state,step_keys) 
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))

    # make identical copies of params and opt_state 
    # now each batch and each device will have the same params and opt_state
    replicate_learner = tree.map(broadcast, replicate_learner)

    # copy the replicated_learner to the physical devices
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # do the same for the hidden state
    # joint_replicated_hstates = []
    # for hs in joint_hstates:
    #     h_task_broadcasted = tree.map(broadcast, hs)
        
    #     h_task_replicated = flax.jax_utils.replicate(h_task_broadcasted, devices=jax.devices())
    #     joint_replicated_hstates.append(h_task_replicated)
    broadcasted_hs = tree.map(broadcast,joint_hstates)
    replicated_hs = flax.jax_utils.replicate(broadcasted_hs,devices=jax.devices())

    # Initialise learner state.
    params, opt_state,step_keys = replicate_learner

    # now this learner state has its params replicated through the devices and envs 
    init_learner_state = LearnerState(
        params=params,
        opt_states=opt_state,
        key=step_keys,
        env_state=states_list,
        timestep=timesteps_list,
        hstates=replicated_hs,
    )

    return learn, apply_fns[0], init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_sable"
    # deep copy the config -> as it enter inside the config tree and extract each and every leave and copy it
    config = copy.deepcopy(_config) #

    n_devices = len(jax.devices())
    
    envs = []
    eval_envs = []
    print(f"\n{Fore.CYAN}--- Creating Environments from Config Tasks ---{Style.RESET_ALL}")

    # config contains logger, arch, system, network, env -> each one of these contains the configuration of something 
    # config.env contains the configs in the multiagent.yaml 
    # config.env.task contains the tasks there -> specific configs for each task like the scenario file name .. etc
    tasks_list = OmegaConf.select(config, "env.tasks", default=None) 
    
    # get the scenario root path to access the scenario folder
    scenario_base_path = OmegaConf.select(config, "env.scenario_config_path", default=None)

    # i will use this to store the task specific config and feed each one of these to the env.make
    # doing so will enable me to give different task config in the same loop without changing the original files
    task_cfg_list = []

    # looping through the tasks list in the multiagent.yaml file so i can get the task config for each env/task there
    for i, task_spec in enumerate(tasks_list): 
            
            # get the task name
            task_name = task_spec.get('name', f'Unnamed Task {i+1}')

            # init -> each task is a copy of the original config -> same logger, arch, system, network, env
            task_cfg = copy.deepcopy(config)

            # this will allow me to change inside the task_cfg and make it changeable
            OmegaConf.set_struct(task_cfg, False)

            # get the env name -> will help to know what env i am dealing with because different envs may have differet configs in the logger, arch, system, network, env
            env_name = task_cfg["env"]['envs_name'][i]['name']

            print(f"  Processing Task {i+1}/{len(tasks_list)}: {Style.BRIGHT}{task_name}{Style.RESET_ALL}")

            # these if statements are specific for each env -> a better way to do it is to make functions that deals with each env 
            # these functions can be here in this file -> then use switch so depending on the env name i will use the function
            # the swtich should return the specific task_cfg of the specific env/task (probably env because i am switching based on the env_name)  
            if env_name == "VectorConnector":
                # get scenario file name , join it with the scenarios path and load the scenario file
                scenario_file_stem = task_spec.get('scenario_file_name')
                scenario_file_path = os.path.join(scenario_base_path, f"{scenario_file_stem}.yaml")
                scenario_cfg = OmegaConf.load(scenario_file_path)

                # now scenario_cfg will contain the scenatio file , we want to add this scenario file to our task_cfg in a way env.make is comfortable with
                task_cfg['env']['scenario']['task_config'] = scenario_cfg['task_config']

                # scenario key is a way to access env.scenario.name 
                # scenario value is the name of the env
                scenario_key = task_spec.get('scenario_key') 
                scenario_value = task_spec.get('scenario_value')

                # here we are updating the value of the env.scenario.name with the env name
                # it can also be done using a normal task_cfg['env']['scenario']['name] = env name but i got errors here so i relied on OmegaConf to do this for me
                OmegaConf.update(task_cfg, scenario_key, scenario_value, merge=True)
                OmegaConf.update(task_cfg.env.scenario, "env_kwargs", {}, merge=True)

                # env specific configs 
                task_cfg['env']['eval_metric'] = task_spec['eval_metric']
                task_cfg['env']['log_win_rate'] = task_spec['log_win_rate']
                task_cfg['env']['implicit_agent_id'] = task_spec['implicit_agent_id']
                task_cfg['env']['aggregate_rewards'] = task_spec['aggregate_rewards']
                task_cfg['env']['kwargs'] = task_spec['task_kwargs']

            elif env_name == "Smax":
                
                # the same is done here
                # this specific env has a different way to create the config than the one before it
                task_cfg['env']['scenario']['name'] = "HeuristicEnemySMAX"
                task_cfg['env']['scenario']['task_name'] = task_name

                task_cfg['env']['eval_metric'] = task_spec['eval_metric']
                task_cfg['env']['log_win_rate'] = task_spec['log_win_rate']
                task_cfg['env']['implicit_agent_id'] = task_spec['implicit_agent_id']
                task_cfg['env']['kwargs'] = task_spec['task_kwargs']
               
                OmegaConf.update(task_cfg.env.scenario, "env_kwargs", {}, merge=True)


            else:
                # same is done here
                scenario_file_stem = task_spec.get('scenario_file_name')
                scenario_file_path = os.path.join(scenario_base_path, f"{scenario_file_stem}.yaml")
                scenario_cfg = OmegaConf.load(scenario_file_path)


                task_cfg['env']['scenario']['task_config'] = scenario_cfg['task_config']
                scenario_key = task_spec.get('scenario_key')
                scenario_value = task_spec.get('scenario_value')
                OmegaConf.update(task_cfg, scenario_key, scenario_value, merge=True)
                OmegaConf.update(task_cfg.env.scenario, "env_kwargs", {}, merge=True)


                task_cfg['env']['eval_metric'] = task_spec['eval_metric']
                task_cfg['env']['log_win_rate'] = task_spec['log_win_rate']
                task_cfg['env']['implicit_agent_id'] = task_spec['implicit_agent_id']
                task_cfg['env']['kwargs'] = task_spec['task_kwargs']


                    
            # in the end i populated the env_name key with the env_name from the list of env_names in the task_cfg itself using an index
            task_cfg['env']['env_name'] = task_cfg['env']['envs_name'][i]['name'] 
                    

            # create the envs append the train_env -> used to train and collect traj
            # eval_env -> used for evaluation
            train_env, eval_env = environments.make(task_cfg)
            envs.append(train_env)
            eval_envs.append(eval_env)
            # save the config of the tasks because these will be used later for evaluation also
            task_cfg_list.append(task_cfg)
            print(f"    {Fore.GREEN}Successfully created envs for task '{task_name}'.{Style.RESET_ALL}")
# PRNG keys.
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=3)

    # Setup learner.
    # learn -> i will explain it when its used down the code
    # sable_execution_fn -> get actions 
    # learner_state -> replicated learner state through device and envs
    learn, sable_execution_fn, learner_state = learner_setup(envs, (key, net_key), config)

    # Setup evaluator.
    def make_rec_sable_act_fn(actor_apply_fn: ActorApply, task_id:int) -> EvalActFn:
        _hidden_state = "hidden_state"

        def eval_act_fn(
            params: Params, timestep: TimeStep, key: chex.PRNGKey, actor_state: ActorState
        ) -> Tuple[Action, Dict]:
            hidden_state = actor_state[_hidden_state]
            output_action, _, _, hidden_state = actor_apply_fn(  # type: ignore
                params,
                timestep.observation,
                hidden_state,
                key,
                task_id = task_id
            )
            return output_action, {_hidden_state: hidden_state}

        return eval_act_fn

    # One key per device for evaluation.

    total_eval_keys_needed = n_devices * len(eval_envs)
    key_e, *eval_keys_flat_list = jax.random.split(key_e, total_eval_keys_needed + 1)
    eval_keys_flat = jnp.stack(eval_keys_flat_list)
    eval_keys_per_task_device = eval_keys_flat.reshape(len(eval_envs), n_devices, -1)

    eval_keys = jax.random.split(key_e, n_devices)
    evaluators_list = []
    for i in range(len(eval_envs)):
        eval_act_fn = make_rec_sable_act_fn(sable_execution_fn,i)

        eval_env_instance = eval_envs[i]
        task_cfg_for_eval = task_cfg_list[i]
        task_name = tasks_list[i].get('name', f'Unnamed Task {i+1}')
        # task_name = f'task_{i}'


        evaluator_fn = get_eval_fn(eval_env_instance, eval_act_fn, task_cfg_for_eval, absolute_metric=False)

        keys_for_this_task = eval_keys_per_task_device[i]


        evaluators_list.append({
            "name": task_name,
            "eval_fn": evaluator_fn, 
            "keys": keys_for_this_task
        })


    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_hs_list = []

    for _ in range(len(eval_envs)):
        eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
        eval_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
        eval_hs = flax.jax_utils.replicate(eval_hs, devices=jax.devices())
        eval_hs_list.append(eval_hs)

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    total_return = 0
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        # for task in range(len(envs)):
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time








        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)

    # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        # trained_params = unreplicate_batch_dim(learner_state.params)
        trained_params = unreplicate_batch_dim(learner_output.learner_state.params)
        all_eval_metrics = {}
        total_eval_return = 0.0



        # key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        # eval_keys = jnp.stack(eval_keys)
        # eval_keys = eval_keys.reshape(n_devices, -1)
        # # Evaluate.
        # for i in range(len(eval_keys)):
        for idx,evaluator_info in enumerate(evaluators_list):
            task_name = evaluator_info["name"]
            evaluator_fn = evaluator_info["eval_fn"]
            eval_task_keys = evaluator_info["keys"]

            eval_metrics = evaluator_fn(trained_params, eval_task_keys, {"hidden_state": eval_hs_list[idx]})
            eval_metrics = tree.map(lambda x: jnp.mean(x), eval_metrics)
            all_eval_metrics[task_name] = eval_metrics
            prefixed_eval_metrics = {}
            base_prefix = "evaluator"

            for metric_key, metric_value in eval_metrics.items():
                new_key = f"{base_prefix}/{task_name}/{metric_key}"
                prefixed_eval_metrics[new_key] = metric_value

            logger.log(prefixed_eval_metrics, t, eval_step, LogEvent.EVAL)
            episode_return = float(eval_metrics.get("episode_return", jnp.nan)) 
            total_eval_return += episode_return

        avg_eval_return = total_eval_return / len(evaluators_list)
        total_return = total_return + avg_eval_return
        logger.log({"eval_average/episode_return": avg_eval_return}, t, eval_step, LogEvent.EVAL)

            # eval_metrics = evaluator_instance(trained_params, eval_keys, {"hidden_state": eval_hs})
            # logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
            # episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        abs_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
        abs_hs = tree.map(lambda x: x[jnp.newaxis], abs_hs)
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": abs_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return total_return / (config.arch.num_evaluation * 2)


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_sable.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Rec Sable experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()