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


class TaskPolicyHeadMLP(nn.Module):
    embed_dim: int 
    action_dim: int 
    name: Optional[str] = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:

        y = nn.Dense(
                features=self.embed_dim, 
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"{self.name}_dense1"
            )(x)
        y = nn.gelu(y)
        y = nn.RMSNorm(name=f"{self.name}_rmsnorm")(y)
        logits = nn.Dense(
                    features=self.action_dim, 
                    kernel_init=orthogonal(0.01),
                    name=f"{self.name}_dense2"
                 )(y)
        return logits


class TaskObsEncoderMLP(nn.Module):
    embed_dim: int
    name: Optional[str] = None



    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        x = nn.RMSNorm(name=f"{self.name}_rmsnorm")(obs)
        x = nn.Dense(
                features=self.embed_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                use_bias=False, 
                name=f"{self.name}_dense"

            )(x)
        x = nn.gelu(x)
        return x


class TaskValueHeadMLP(nn.Module):

    embed_dim: int
    name: Optional[str] = None

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:

        y = nn.Dense(
                features=self.embed_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                name=f"value_{self.name}_dense1"
            )(x)
        y = nn.gelu(y)
        y = nn.RMSNorm(name=f"value_{self.name}_rmsnorm")(y)
        value = nn.Dense(
                    features=1,
                    kernel_init=orthogonal(0.01),
                    name=f"value_{self.name}_dense2"
                )(y)
        return value


class TaskActionEncoderMLP(nn.Module):
    embed_dim: int  
    action_dim: int 
    name: Optional[str] = None

    def setup(self):
        self.action_embedding_layer = nn.Dense(
                    self.embed_dim,
                    use_bias=True,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                )


    @nn.compact
    def __call__(self, action: chex.Array) -> chex.Array:
        x = self.action_embedding_layer(action)
        x = nn.gelu(x)
        return x
    
class EncodeBlock(nn.Module):   
    """Sable encoder block."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int

    def setup(self) -> None:
        self.ln1 = nn.RMSNorm()
        self.ln2 = nn.RMSNorm()

        self.retn = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            n_agents=self.n_agents,
            masked=False,  # Full retention for the encoder
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )

        self.ffn = SwiGLU(self.net_config.embed_dim, self.net_config.embed_dim)

    def __call__(
        self, x: chex.Array, hstate: chex.Array, dones: chex.Array, step_count: chex.Array
    ) -> chex.Array:
        """Applies Chunkwise MultiScaleRetention."""
        ret, updated_hstate = self.retn(
            key=x, query=x, value=x, hstate=hstate, dones=dones, step_count=step_count
        )
        x = self.ln1(x + ret)
        output = self.ln2(x + self.ffn(x))
        return output, updated_hstate

    def recurrent(self, x: chex.Array, hstate: chex.Array, step_count: chex.Array) -> chex.Array:
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
    n_agents: int
    num_tasks: int


    def setup(self) -> None:
        self.ln = nn.RMSNorm()

        self.task_obs_encoders = [
        TaskObsEncoderMLP(self.net_config.embed_dim, name=f"task_obs_encoder_{i}")
        for i in range(self.num_tasks)
        ]

        self.task_value_heads = [ 
            TaskValueHeadMLP(
                embed_dim=self.net_config.embed_dim,
                name=f"value_head_{task_id}"
            ) for task_id in range(self.num_tasks) 
        ]

        self.blocks = [
            EncodeBlock(
                self.net_config,
                self.memory_config,
                self.n_agents,
                name=f"encoder_block_{block_id}",
            )
            for block_id in range(self.net_config.n_block)
        ]

    def __call__(
        self, obs: chex.Array, hstate: chex.Array, dones: chex.Array, step_count: chex.Array, task_id: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply chunkwise encoding."""
        
        selected_encoder = self.task_obs_encoders[task_id] 
        obs_rep = selected_encoder(obs)


        updated_hstate = jnp.zeros_like(hstate)
        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            hs = hstate[:, :, i]  # Get the hidden state for the current block
            # Apply the chunkwise encoder block
            obs_rep, hs_new = block(self.ln(obs_rep), hs, dones, step_count)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)
        
        
        selected_value_head = self.task_value_heads[task_id] 
        value = selected_value_head(obs_rep)

        # value = self.head(obs_rep)

        return value, obs_rep, updated_hstate

    def recurrent(
        self, obs: chex.Array, hstate: chex.Array, step_count: chex.Array, task_id: int
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Apply recurrent encoding."""

        # dummy_h = jnp.zeros_like(hstate)
        # for enc in self.task_obs_encoders:
        #     _rep, _hs = enc.recurrent(obs, dummy_h, step_count)


        selected_encoder = self.task_obs_encoders[task_id] 
        obs_rep = selected_encoder(obs)

        updated_hstate = jnp.zeros_like(hstate)

        # Apply the encoder blocks
        for i, block in enumerate(self.blocks):
            hs = hstate[:, :, i]  # Get the hidden state for the current block
            # Apply the recurrent encoder block
            obs_rep, hs_new = block.recurrent(self.ln(obs_rep), hs, step_count)
            updated_hstate = updated_hstate.at[:, :, i].set(hs_new)

        # Compute the value function
        # value = self.head(obs_rep)

        selected_value_head = self.task_value_heads[task_id] 
        value = selected_value_head(obs_rep) 


        return value, obs_rep, updated_hstate


class DecodeBlock(nn.Module):
    """Sable decoder block."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int

    def setup(self) -> None:
        self.ln1, self.ln2, self.ln3 = nn.RMSNorm(), nn.RMSNorm(), nn.RMSNorm()

        self.retn1 = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            n_agents=self.n_agents,
            masked=True,  # Masked retention for the decoder
            memory_config=self.memory_config,
            decay_scaling_factor=self.memory_config.decay_scaling_factor,
        )
        self.retn2 = MultiScaleRetention(
            embed_dim=self.net_config.embed_dim,
            n_head=self.net_config.n_head,
            n_agents=self.n_agents,
            masked=True,  # Masked retention for the decoder
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
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Applies Chunkwise MultiScaleRetention."""
        hs1, hs2 = hstates

        # Apply the self-retention over actions
        ret, hs1_new = self.retn1(
            key=x, query=x, value=x, hstate=hs1, dones=dones, step_count=step_count
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
            key_n=x, query_n=x, value_n=x, hstate=hs1, step_count=step_count
        )
        ret = self.ln1(x + ret)

        # Apply the cross-retention over obs x action
        ret2, hs2_new = self.retn2.recurrent(
            key_n=ret, query_n=obs_rep, value_n=ret, hstate=hs2, step_count=step_count
        )
        y = self.ln2(obs_rep + ret2)
        output = self.ln3(y + self.ffn(y))

        return output, (hs1_new, hs2_new)


class Decoder(nn.Module):
    """Multi-block decoder consisting of multiple `DecoderBlock` modules."""

    net_config: SableNetworkConfig
    memory_config: DictConfig
    n_agents: int
    tasks_action_dims: list[int]
    num_tasks: int 
    tasks_action_space_type: list[str]

    def setup(self) -> None:
        self.ln = nn.RMSNorm()


        self.task_action_encoders = [ 
            TaskActionEncoderMLP(
                embed_dim=self.net_config.embed_dim, 
                action_dim=self.tasks_action_dims[task_id],       
                name= f"action_encoder_{task_id}"
            ) for task_id in range(self.num_tasks) 
        ]

        #TODO I should change this later
        #Optional: out of the scope 
        self.log_std = (
            self.param("log_std", nn.initializers.zeros, (self.tasks_action_dims[0],))
            if self.tasks_action_space_type[0] == _CONTINUOUS
            else None
        )


        self.task_policy_heads = [ 
            TaskPolicyHeadMLP(
                embed_dim=self.net_config.embed_dim, 
                action_dim=self.tasks_action_dims[task_id],       
                name=f"policy_head_{task_id}"
            ) for task_id in range(self.num_tasks) 
        ]

        self.blocks = [
            DecodeBlock(
                self.net_config,
                self.memory_config,
                self.n_agents,
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
        # action_embeddings = self.action_encoder(action)

        selected_action_encoder = self.task_action_encoders[task_id] 
        action_embeddings = selected_action_encoder(action)

        x = self.ln(action_embeddings)

        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = tree.map(lambda x, j=i: x[:, :, j], hstates)
            x, hs_new = block(x=x, obs_rep=obs_rep, hstates=hs, dones=dones, step_count=step_count)
            updated_hstates = tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        # logit = self.head(x)
        selected_policy_head = self.task_policy_heads[task_id]
        logit = selected_policy_head(x)

        return logit, updated_hstates

    def recurrent(
        self,
        action: chex.Array,
        obs_rep: chex.Array,
        hstates: Tuple[chex.Array, chex.Array],
        step_count: chex.Array,
        task_id: int,
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Apply recurrent decoding."""
        updated_hstates = tree.map(jnp.zeros_like, hstates)
        # action_embeddings = self.action_encoder(action)

        selected_action_encoder = self.task_action_encoders[task_id] 
        action_embeddings = selected_action_encoder(action)

        x = self.ln(action_embeddings)


        # Apply the decoder blocks
        for i, block in enumerate(self.blocks):
            hs = tree.map(lambda x, i=i: x[:, :, i], hstates)
            x, hs_new = block.recurrent(x=x, obs_rep=obs_rep, hstates=hs, step_count=step_count)
            updated_hstates = tree.map(
                lambda x, y, j=i: x.at[:, :, j].set(y), updated_hstates, hs_new
            )

        # logit = self.head(x)
        selected_policy_head = self.task_policy_heads[task_id] 
        logit = selected_policy_head(x) 


        return logit, updated_hstates


class SableNetwork(nn.Module):
    """Sable network module."""

    n_agents: int
    n_agents_per_chunk: int
    task_action_dims: list
    net_config: SableNetworkConfig
    memory_config: DictConfig
    task_action_space_types: list
    num_tasks:int 

    def setup(self) -> None:
        # if self.action_space_type not in [_DISCRETE, _CONTINUOUS]:
        #     raise ValueError(f"Invalid action space type: {self.action_space_type}")

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

        self.encoder = Encoder(
            self.net_config,
            self.memory_config,
            self.n_agents_per_chunk,
            self.num_tasks,
        )
        self.decoder = Decoder( 
            self.net_config,
            self.memory_config,
            self.n_agents_per_chunk,
            self.task_action_dims,
            self.num_tasks,
            self.task_action_space_types,
        )

        # Set the actor and trainer functions
        self.train_encoder_fn = partial(
            train_encoder_fn,
            chunk_size=self.memory_config.chunk_size,
        )
        self.act_encoder_fn = partial(
            act_encoder_fn,
            chunk_size=self.n_agents_per_chunk,
        )
        # TODO what if the action space is cont ?  or what if i got mixed action spaces? probably move this condions to the __call function 
        # if self.task_action_space_types[0] == _CONTINUOUS:
        #     self.train_decoder_fn = partial(
        #         continuous_train_decoder_fn,
        #         n_agents=self.n_agents,
        #         chunk_size=self.memory_config.chunk_size,
        #         action_dim=self.task_action_dims[0],
        #     )
        #     self.autoregressive_act = partial(
        #         continuous_autoregressive_act, action_dim=self.task_action_dims[0]
        #     )
        # else:
        #TODO: define this per task

        self.train_decoder_fn = [
            partial(
                discrete_train_decoder_fn,
                n_agents=self.n_agents,
                chunk_size=self.memory_config.chunk_size,
            )
            for _ in range(self.num_tasks)
        ]

        self.autoregressive_act = [
            discrete_autoregressive_act
            for _ in range(self.num_tasks)
        ]



    def __call__(
        self,
        observation: Observation,
        action: chex.Array,
        hstates: HiddenStates,
        dones: chex.Array,
        task_id: int,
        rng_key: Optional[chex.PRNGKey] = None,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Training phase."""
        obs, legal_actions, step_count = (
            observation.agents_view,
            observation.action_mask,
            observation.step_count,
        )

        value, obs_rep, _ = self.train_encoder_fn(
            encoder=self.encoder, obs=obs, hstate=hstates[0], dones=dones, step_count=step_count,task_id=task_id
        )

        action_log, entropy = self.train_decoder_fn[task_id](
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            legal_actions=legal_actions,
            hstates=hstates[1:],
            dones=dones,
            step_count=step_count,
            rng_key=rng_key,
            task_id=task_id  
        )

        value = jnp.squeeze(value, axis=-1)
        return value, action_log, entropy

    def get_actions(
        self,
        observation: Observation,
        hstates: HiddenStates,
        key: chex.PRNGKey,
        task_id:int
    ) -> Tuple[chex.Array, chex.Array, chex.Array, HiddenStates]:
        """Inference phase."""
        obs, legal_actions, step_count = (
            observation.agents_view,
            observation.action_mask,
            observation.step_count,
        )

        # Decay the hidden states: each timestep we decay the hidden states once
        decayed_hstates = tree.map(lambda x: x * self.decay_kappas, hstates)

        value, obs_rep, updated_enc_hs = self.act_encoder_fn(
            encoder=self.encoder,
            obs=obs,
            decayed_hstate=decayed_hstates[0],
            step_count=step_count,
            task_id=task_id
        )

        output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act[task_id](
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

    @nn.compact
    def init_all_tasks(
      self,
        observation: list[Observation],
        hstates: list[HiddenStates],
        key: chex.PRNGKey,
        ) -> None: 
        for i in range(len(observation)):

            obs, legal_actions, step_count = (
                observation[i].agents_view,
                observation[i].action_mask,
                observation[i].step_count,
            )

            decayed_hstates = tree.map(lambda x: x * self.decay_kappas, hstates[i])
        # for task_id in range(len(self.encoder.task_obs_encoders)):

            value, obs_rep, updated_enc_hs = self.act_encoder_fn(
                    encoder=self.encoder,
                    obs=obs,
                    decayed_hstate=decayed_hstates[0],
                    step_count=step_count,
                    task_id=i
                )
            
            output_actions, output_actions_log, updated_dec_hs = self.autoregressive_act[i](
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

            