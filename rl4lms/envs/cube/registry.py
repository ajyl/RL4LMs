from typing import Any, Dict, Type, Union

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rl4lms.algorithms.ppo.ppo import PPO
from rl4lms.data_pools.cube import CubeData, CubeDataPool
from rl4lms.envs.cube.alg_wrappers import wrap_onpolicy_alg
from rl4lms.envs.text_generation.metric import (
    BaseMetric,
    LearnedRewardMetric,
    Perplexity,
)
from rl4lms.envs.text_generation.policy.base_policy import LMActorCriticPolicy
from rl4lms.envs.cube.policy.causal_policy import (
    CausalLMActorCriticPolicy,
)
from rl4lms.envs.text_generation.reward import (
    LearnedBatchedRewardFunction,
    RewardFunction,
)
from rl4lms.envs.text_generation.test_datapool import TestTextGenPool


class DataPoolRegistry:
    _registry = {
        "dummy_pool": TestTextGenPool,
        "cube": CubeData,
    }

    @classmethod
    def get(cls, datapool_id: str, kwargs: Dict[str, Any]) -> CubeDataPool:
        datapool_cls = cls._registry[datapool_id]
        datapool = datapool_cls.prepare(**kwargs)
        return datapool

    @classmethod
    def add(cls, id: str, datapool_cls: Type[CubeDataPool]):
        DataPoolRegistry._registry[id] = datapool_cls


class RewardFunctionRegistry:
    _registry = {
        "learned_reward": LearnedBatchedRewardFunction,
    }

    @classmethod
    def get(cls, reward_fn_id: str, kwargs: Dict[str, Any]) -> RewardFunction:
        reward_cls = cls._registry[reward_fn_id]
        reward_fn = reward_cls(**kwargs)
        return reward_fn

    @classmethod
    def add(cls, id: str, reward_fn_cls: Type[RewardFunction]):
        RewardFunctionRegistry._registry[id] = reward_fn_cls


class MetricRegistry:
    _registry = {
        "learned_reward": LearnedRewardMetric,
        "causal_perplexity": Perplexity,
    }

    @classmethod
    def get(cls, metric_id: str, kwargs: Dict[str, Any]) -> BaseMetric:
        metric_cls = cls._registry[metric_id]
        metric = metric_cls(**kwargs)
        return metric

    @classmethod
    def add(cls, id: str, metric_cls: Type[BaseMetric]):
        MetricRegistry._registry[id] = metric_cls


class PolicyRegistry:
    _registry = {
        "causal_lm_actor_critic_policy": CausalLMActorCriticPolicy,
    }

    @classmethod
    def get(cls, policy_id: str) -> Type[LMActorCriticPolicy]:
        policy_cls = cls._registry[policy_id]
        return policy_cls

    @classmethod
    def add(cls, id: str, policy_cls: Type[LMActorCriticPolicy]):
        PolicyRegistry._registry[id] = policy_cls


class AlgorithmRegistry:
    _registry = {
        "ppo": PPO,
    }

    @classmethod
    def get(cls, alg_id: str) -> Union[Type[OnPolicyAlgorithm]]:
        try:
            alg_cls = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return alg_cls

    @classmethod
    def add(cls, id: str, alg_cls: Union[Type[OnPolicyAlgorithm]]):
        AlgorithmRegistry._registry[id] = alg_cls


class WrapperRegistry:
    _registry = {
        "ppo": wrap_onpolicy_alg,
    }

    @classmethod
    def get(cls, alg_id: str):
        try:
            wrapper_def = cls._registry[alg_id]
        except KeyError:
            raise NotImplementedError
        return wrapper_def

    @classmethod
    def add(cls, id: str, wrapper_def):
        WrapperRegistry._registry[id] = wrapper_def
