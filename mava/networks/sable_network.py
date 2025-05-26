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

from functools import partial
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal
from jax import tree
from omegaconf import DictConfig

from mava.networks.retention import MultiScaleRetention
from mava.networks.torsos import SwiGLU
from mava.networks.utils.sable import (
    act_encoder_fn,
    continuous_autoregressive_act,
    continuous_train_decoder_fn,
    discrete_autoregressive_act,
    discrete_train_decoder_fn,
    train_encoder_fn,
)
from mava.systems.sable.types import HiddenStates, SableNetworkConfig
from mava.types import Observation
from mava.utils.network_utils import _CONTINUOUS, _DISCRETE
from typing import Sequence 


class EncodeBlock(nn.Module):   
    """Sable encoder block."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    all_n_agents_list: list[int]

    def setup(self) -> None:
        self.ln1 = nn.RMSNorm()
        self.ln2 = nn.RMSNorm()
        
        # pass a list of number of agents for all tasks so we can access the task specific n_agents inside the multiscaleretention to create
        # task specific decay matrix
        self.retn = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            all_n_agents_list=self.all_n_agents_list, 
            masked=False,  
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )

        self.ffn = SwiGLU(self.net_config.embed_dim, self.net_config.embed_dim)

    def __call__(
        self, x: chex.Array, hstate: chex.Array, dones: chex.Array, step_count: chex.Array, task_id: int,
    ) -> chex.Array:
        """Applies Chunkwise MultiScaleRetention."""
        ret, updated_hstate = self.retn(
            key=x, query=x, value=x, hstate=hstate, dones=dones, step_count=step_count, task_id=task_id
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate

    def recurrent(self, x: chex.Array, hstate: chex.Array, step_count: chex.Array,) -> chex.Array:
        """Applies Recurrent MultiScaleRetention."""
        ret, updated_hstate = self.retn.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hstate, step_count=step_count
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate


class Encoder(nn.Module):
    """Multi-block encoder consisting of multiple `EncoderBlock` modules."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    all_n_agents_list: list[int]


    def setup(self) -> None:
        self.ln = nn.RMSNorm()

        #
        
        self.obs_encoder = nn.Sequential(
            [
                nn.RMSNorm(),
                nn.Dense(
                    self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2)), use_bias=False
                ),
                nn.gelu,
            ],
            name="shared_obs_encoder" 
        )
        
        
        self.head = nn.Sequential(
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.RMSNorm(),
                nn.Dense(1, kernel_init=orthogonal(0.01)), 
            ],
            name="shared_value_head" 
        )

        self.blocks = [
            EncodeBlock(
                self.net_config,
                self.memory_config,
                self.all_n_agents_list, 
                name=f"encoder_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]


    def __call__(
        self, obs: chex.Array, hstate: chex.Array, dones: chex.Array, step_count: chex.Array, task_id: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply chunkwise encoding."""
        # routing to the task MLP for obs
        obs_rep = self.obs_encoder(obs) 


        updated_hstate = jnp.zeros_like(hstate)
        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            hs = hstate[:, :, i]  # Get the hidden state for the current block
            # Apply the chunkwise encoder block
            obs_rep, hs_new = block(self.ln(obs_rep), hs, dones, step_count,task_id)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)
        
        value = self.head(obs_rep) 



        return value, obs_rep, updated_hstate

    def recurrent(
        self, obs: chex.Array, hstate: chex.Array, step_count: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply recurrent encoding."""


        obs_rep = self.obs_encoder(obs) 


        updated_hstate = jnp.zeros_like(hstate)

        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            hs = hstate[:, :, i]  # Get the hidden state for the current block
            # Apply the recurrent encoder block
            # passing the task id here is not necessary TODO removed later
            obs_rep, hs_new = block.recurrent(self.ln(obs_rep), hs, step_count)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        value = self.head(obs_rep) 


        return value, obs_rep, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    all_n_agents_list: list[int]

    def setup(self) -> None:
        self.ln1, self.ln2, self.ln3 = nn.RMSNorm(), nn.RMSNorm(), nn.RMSNorm()

        self.retn1 = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            all_n_agents_list=self.all_n_agents_list,
            # the decoder need to be masked
            masked=True,  
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )
        self.retn2 = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            all_n_agents_list=self.all_n_agents_list,
            # same here
            masked=True, 
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )

        self.ffn = SwiGLU(self.net_config.embed_dim, self.net_config.embed_dim)

    def __call__(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
        step_count: chex.Array,
        task_id: int,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Chunkwise MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1(
            key=x, query=x, value=x, hstate=hs1, dones=dones, step_count=step_count, task_id=task_id
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2(
            key=ret,
            query=obs_rep,
            value=ret,
            hstate=hs2,
            dones=dones,
            step_count=step_count,
            task_id=task_id
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)

    def recurrent(
        self,
        x: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        step_count: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Recurrent MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1.recurrent(
            key_n=x, query_n=x, value_n=x, hstate=hs1, step_count=step_count,
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2.recurrent(
            key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2, step_count=step_count,
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder consisting of multiple `DecoderBlock` modules."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    all_n_agents_list: list[int]
    max_action_dims: int
    tasks_action_space_type: list[str]

    def setup(self) -> None:
        self.ln = nn.RMSNorm()

        use_bias = self.tasks_action_space_type == _CONTINUOUS 
        self.action_encoder = nn.Sequential( 
            [
                nn.Dense(
                    self.net_config.embed_dim,
                    use_bias=use_bias,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                ),
                nn.gelu,
            ],
            name="shared_action_encoder" 
        )

        
        self.log_std = ( 
            self.param("log_std", nn.initializers.zeros, (self.max_action_dims,)) 
            if self.tasks_action_space_type == _CONTINUOUS
            else None
        )

        
        
        self.head = nn.Sequential( 
            [
                nn.Dense(self.net_config.embed_dim, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                nn.RMSNorm(),
                nn.Dense(self.max_action_dims, kernel_init=orthogonal(0.01)), 
            ],
            name="shared_policy_head" 
        )

        self.blocks = [
            DecodeBlock(
                self.net_config,
                self.memory_config,
                self.all_n_agents_list, 
                name=f"decoder_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]


    def __call__(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        dones: chex.Array,
        step_count: chex.Array,
        task_id: int,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply chunkwise decoding."""
        updated_hstates = tree.map(jnp.zeros_like, hstates)
        # same here -> select the task specific action encoder
        action_embeddings = self.action_encoder(action) 

        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = tree.map(lambda x, j=i: x[:, :, j], hstates)
            x, hs_new = block(x=x, obs_rep=obs_rep, hstates=hs, dones=dones, step_count=step_count, task_id=task_id)
            updated_hstates = tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        logit = self.head(x) 

        return logit, updated_hstates

    def recurrent(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        step_count: chex.Array,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply recurrent decoding."""
        updated_hstates = tree.map(jnp.zeros_like, hstates)

        action_embeddings = self.action_encoder(action) 


        x = self.ln(action_embeddings)


        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = tree.map(lambda x, i=i: x[:, :, i], hstates)
            x, hs_new = block.recurrent(x=x, obs_rep=obs_rep, hstates=hs, step_count=step_count,)
            updated_hstates = tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        logit = self.head(x)


        return logit, updated_hstates


class SableNetwork(nn.Module):
    """Sable network module."""

    all_n_agents: tuple
    n_agents_per_chunk: tuple
    max_action_dim: int
    net_config: SableNetworkConfig
    memory_config: DictConfig
    task_action_space_types: list
    num_tasks:int 

    def setup(self) -> None:

        # just to make sure that the decay scaling factor is between 0 and 1
        assert (
            self.memory_config.decay_scaling_factor >= 0
            and self.memory_config.decay_scaling_factor <= 1
        ), "Decay scaling factor should be between 0 and 1"

        # Decay kappa for each head
        self.decay_kappas = 1 - jnp.exp(
            jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.net_config.n_head)
        )

        self.decay_kappas = self.decay_kappas * self.memory_config.decay_scaling_factor
        self.decay_kappas = self.decay_kappas[None, :, None, None, None]

        # create the encoder and decoder
        self.encoder = Encoder(
            self.net_config,
            self.memory_config,
            self.n_agents_per_chunk,
        )
        self.decoder = Decoder( 
            self.net_config,
            self.memory_config,
            self.n_agents_per_chunk,
            self.max_action_dim,
            self.task_action_space_types,
        )

        # Set the actor and trainer functions
        # partial is like baking or pre-setting certain arguments into the function.
        # now i dont need to pass chunk size list each time i am calling train_encoder_fn and act_encoder_fn
        self.train_encoder_fn = partial(
            train_encoder_fn,
            chunk_size=self.memory_config.chunk_size,
        )
        self.act_encoder_fn = partial(
            act_encoder_fn,
            chunk_size=self.n_agents_per_chunk,
        )
        
        # here i am making multiple functions (depending on the num_tasks) and each function will handle a set of num_agents
        # this will allow me to have 3 functions for example, each one will deal with an env (task) and each env (task) will have different 
        # num of agents. 
        # I am also using partial to bake the chunck size
        # I am making it a list because i need multiple decay matrices 
        self.train_decoder_fn = [
            partial(
                discrete_train_decoder_fn,
                n_agents=self.all_n_agents[_],
                chunk_size=self.memory_config.chunk_size,
            )
            for _ in range(self.num_tasks)
        ]

        # having different copies of discrete_autoregressive_act function, each copy is dedicated for a task
        self.autoregressive_act = discrete_autoregressive_act
           


    def __call__(
        self,
        obs,
        legal_actions,
        step_count,
        action: chex.Array,
        hstates: HiddenStates,
        dones: chex.Array,
        task_id: int,
        rng_key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Training phase."""
        # obs, legal_actions, step_count = (
        #     observation.agents_view,
        #     observation.action_mask,
        #     observation.step_count,
        # )
        
        # already got chunk_sizes list baked in 
        value, obs_rep, _ = self.train_encoder_fn(
            encoder=self.encoder, obs=obs, hstate=hstates[0], dones=dones, step_count=step_count,task_id=task_id
        )    

        # access the specific function needed for the task -> which have the num of agents for that task and chunk size list baked in 
        action_log, entropy = self.train_decoder_fn[task_id](
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            legal_actions=legal_actions,
            hstates=hstates[1:],
            dones=dones,
            step_count=step_count,
            rng_key=rng_key,
            task_id=task_id ,
            max_action_dim= self.max_action_dim
        )

        value = jnp.squeeze(value, axis=-1)
        return value, action_log, entropy

    def get_actions(
        self,
        obs: Observation,
        legal_actions,
        step_count,
        hstates: HiddenStates,
        key: chex.PRNGKey,
        task_id:int
    ) -> Tuple[chex.Array, chex.Array, chex.Array, HiddenStates]:
        """Inference phase."""

        # Decay the hidden states: each timestep we decay the hidden states once
        decayed_hstates = tree.map(lambda x: x * self.decay_kappas, hstates)

        # same here
        value, obs_rep, updated_enc_hs = self.act_encoder_fn(
            encoder=self.encoder,
            obs=obs,
            decayed_hstate=decayed_hstates[0],
            step_count=step_count,
            task_id=task_id
        )

        # this function doesnt need the num of agents or the chunk size list -> that why i did not bake them inside
        output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act(
            decoder=self.decoder,
            obs_rep=obs_rep,
            legal_actions=legal_actions,
            hstates=decayed_hstates[1:],
            step_count=step_count,
            key=key,
            task_id = task_id,
        )

        updated_hs = HiddenStates(
            encoder=updated_enc_hs,
            decoder_self_retn=updated_dec_hs[0],
            decoder_cross_retn=updated_dec_hs[1],
        )

        value = jnp.squeeze(value, axis=-1)
        return output_actions, output_actions_log, value, updated_hs

    # custom init function to initalize all tasks
    # it has the same functionality as get_action (which was the previous init task) 
    # for multi tasking we have a list of obs, hs -> we loop through them and initialize each task dedicated function there
    @nn.compact
    def init_all_tasks(
      self,
        padded_obs: list[Observation],
        hstates: list[HiddenStates],
        key: chex.PRNGKey,
        ) -> None: 
        for i in range(len(padded_obs)):
            

            obs, legal_actions, step_count = (
                padded_obs[i].agents_view,
                padded_obs[i].action_mask,
                padded_obs[i].step_count,
            )

            decayed_hstates = tree.map(lambda x: x * self.decay_kappas, hstates[i])

            value, obs_rep, updated_enc_hs = self.act_encoder_fn(
                    encoder=self.encoder,
                    obs=obs,
                    decayed_hstate=decayed_hstates[0],
                    step_count=step_count,
                    task_id=i
                )
            
            output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act(
                decoder=self.decoder,
                obs_rep=obs_rep,
                legal_actions=legal_actions,
                hstates=decayed_hstates[1:],
                step_count=step_count,
                key=key,
                task_id = i,
            )

            updated_hs = HiddenStates(
                encoder=updated_enc_hs,
                decoder_self_retn=updated_dec_hs[0],
                decoder_cross_retn=updated_dec_hs[1],
            )

            