from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.dict import Dict as DictSpace
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from torch import nn
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Config
from transformers.modeling_utils import unwrap_model

from rl4lms.algorithms.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
from rl4lms.algorithms.common.maskable.logits_processor import (
    MaskLogitsProcessorCasualLM,
)
from rl4lms.envs.text_generation.hf_generation_utils import override_generation_routines
from rl4lms.envs.cube.policy.base_policy import (
    EvaluateActionsOutput,
    GenerationInputs,
    GenerationOutputs,
    LMActorCriticPolicy,
    PolicyOutput,
    PolicyType,
    RefPolicyOutput,
    ValueOutput,
)
from rl4lms.envs.text_generation.warm_start import (
    ActorCriticWarmStartMixin,
    MaskableActorCriticWarmStartMixin,
)


class CausalLMActorCriticPolicy(LMActorCriticPolicy, ActorCriticWarmStartMixin):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        lr_schedule: Schedule,
        model_name: str,
        architecture: Dict[str, Any] = {},
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
        state_dict: Dict[str, Any] = None,
    ):
        self.architecture = architecture
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
        )
        self.load_from_dict(state_dict)

    def _build_model_heads(self, model_name: str):
        # Hacky lol.
        assert self.architecture is not None

        vocab_size = self.architecture["vocab_size"]
        config = GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=self.architecture["bos_token_id"],
            eos_token_id=self.architecture["eos_token_id"],
            pad_token_id=self.architecture["pad_token_id"],
            n_positions=self.architecture["max_length"],
            n_embd=int(self.architecture["n_embd"]),
            n_layer=int(self.architecture["n_layer"]),
            n_head=int(self.architecture["n_head"]),
        )
        self._policy_model = GPT2LMHeadModel(config=config)
        self._policy_model.resize_token_embeddings(vocab_size)
        self._policy_model.__class__ = override_generation_routines(
            type(self._policy_model)
        )

        self._value_model = GPT2LMHeadModel(config=config)
        self._ref_model = deepcopy(self._policy_model).eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False
        )

        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
            else:  # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._ref_model = torch.nn.DataParallel(self._ref_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(
                    self._value_head.to(self.device)
                )

    def _prepare_inputs_for_model(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.tensor,
        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ):
        model_inputs = unwrap_model(model).prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )

        if self._apply_model_parallel and unwrap_model(model).is_parallelizable:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {
                key: (
                    value.to(model.transformer.first_device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in model_inputs.items()
            }
        return model_inputs

    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            # take attention mask only for the first step
            # for subsequent steps, update_model_kwargs will handle it
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._policy_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._policy_model(output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        entropy = dist.entropy()

        # sample act
        log_prob = dist.log_prob(actions)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._policy_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._policy_model
            ).config.is_encoder_decoder,
        )

        policy_outputs = PolicyOutput(
            actions=actions,
            raw_log_probs=log_prob,
            log_probs=log_prob,
            entropy=entropy,
            past_model_kwargs=past_model_kwargs,
        )

        return policy_outputs

    def forward_value(
        self,
        obs: TensorDict,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        # prepare inputs
        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._value_model, input_ids, past_model_kwargs
        )

        # forward pass to transformers
        output = self._value_model(output_hidden_states=True, **model_inputs)

        # pool the hidden states ?
        last_tokens_hidden = output.hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._value_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._value_model
            ).config.is_encoder_decoder,
        )

        value_outputs = ValueOutput(values=values, past_model_kwargs=past_model_kwargs)

        return value_outputs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:

        policy_outputs = self.forward_policy(obs=obs, actions=actions)
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def get_log_probs_ref_model(
        self,
        obs: TensorDict,
        action: torch.tensor,
        past_model_kwargs: Dict[str, Any] = None,
    ) -> RefPolicyOutput:
        self._ref_model = self._ref_model.eval()

        input_ids = obs["input_encoded_pt"]
        attention_mask = obs["input_attention_mask_pt"]

        if not past_model_kwargs:
            past_model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(
            self._ref_model, input_ids, past_model_kwargs
        )
        output = self._ref_model(output_hidden_states=True, **model_inputs)
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._ref_model
        )._update_model_kwargs_for_generation(
            output,
            past_model_kwargs,
            is_encoder_decoder=self.is_encoder_decoder(self._ref_model),
        )

        ref_policy_outputs = RefPolicyOutput(log_prob, past_model_kwargs)
        return ref_policy_outputs

    def get_policy_first_device(self):
        return (
            self._policy_model.transformer.first_device
            if self._apply_model_parallel
            and unwrap_model(self._policy_model).is_parallelizable
            else "cuda"
        )

    def get_inputs_for_generation(self, obs: TensorDict):
        gen_inputs = GenerationInputs(
            obs["input_encoded_pt"], obs["input_attention_mask_pt"]
        )
        return gen_inputs

    def get_policy_type(self):
        return PolicyType.CAUSAL

    def to(self, device: str):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)

    def get_distribution(self, obs: TensorDict, detach=False):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        past_model_kwargs = {
            "attention_mask": attention_mask,
        }

        if detach:
            with torch.no_grad():
                model_inputs = self._prepare_inputs_for_model(
                    self._policy_model, input_ids, past_model_kwargs
                )

                # forward pass to transformers
                output = self._policy_model(output_hidden_states=True, **model_inputs)
        else:
            model_inputs = self._prepare_inputs_for_model(
                self._policy_model, input_ids, past_model_kwargs
            )

            # forward pass to transformers
            output = self._policy_model(output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        return dist

    def predict_values(self, obs: TensorDict):
        return self.forward_value(obs).values
