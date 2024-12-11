from functools import partial
from typing import Any, Dict, List
import numpy as np

from rl4lms.data_pools.cube import Sample
from rl4lms.envs.cube.env import CubeEnv
from rl4lms.envs.cube.evaluation_utils import evaluate_on_samples
from rl4lms.envs.cube.logging_utils import Tracker
from rl4lms.envs.cube.registry import (
    DataPoolRegistry,
    MetricRegistry,
    RewardFunctionRegistry,
    PolicyRegistry,
    AlgorithmRegistry,
    WrapperRegistry,
)
from rl4lms.envs.text_generation.reward import RewardFunction
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from rl4lms.envs.cube.tokenizer import CubeTokenizer
from rl4lms.envs.text_generation.warm_start import TrainerWarmStartMixin


def build_tokenizer(tokenizer_config):
    return CubeTokenizer(tokenizer_config.get("max_length"))


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(
        reward_config["id"], reward_config.get("args", {})
    )
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [
        MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
        for metric_config in metric_configs
    ]
    return metrics


def build_datapool(
    datapool_config: Dict[str, Any], tokenizer: CubeTokenizer, dev=False
):

    def _get_datapool_by_split(split: str, tokenizer, dev=False):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        kwargs["dev"] = dev
        kwargs["tokenizer"] = tokenizer
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train", tokenizer, dev=dev)
    val_datapool = _get_datapool_by_split("val", tokenizer, dev=dev)
    test_datapool = _get_datapool_by_split("test", tokenizer, dev=dev)

    samples_by_split = {
        "train": [(sample, weight) for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool][:256],
        "test": [sample for sample, _ in test_datapool][:256],
    }
    return samples_by_split


def build_env(
    env_config: Dict[str, Any],
    reward_fn: RewardFunction,
    tokenizer: CubeTokenizer,
    train_samples: List[Sample],
):
    # vectoried env
    env_kwargs = {
        "reward_function": reward_fn,
        "tokenizer": tokenizer,
        "samples": train_samples,
    }
    env_kwargs = {**env_kwargs, **env_config.get("args", {})}
    env = make_vec_env(
        CubeEnv,
        n_envs=env_config.get("n_envs", 1),
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
    )
    return env


def build_alg(
    alg_config: Dict[str, Any],
    env: CubeEnv,
    tokenizer: CubeTokenizer,
    tracker: Tracker,
    policy_state: Dict[str, Any],
    alg_state: Dict[str, Any],
):
    # TBD - move these to a registry once the experimentation is done
    # Also switch to Sb3 algos when possible with minimal code adaptations
    policy_config = alg_config["policy"]
    policy_cls = PolicyRegistry.get(policy_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])

    policy_args = policy_config["args"]
    policy_args["state_dict"] = policy_state
    policy_args["architecture"]["vocab_size"] = tokenizer.vocab_size
    policy_args["architecture"]["max_length"] = tokenizer.max_length
    policy_args["architecture"]["bos_token_id"] = tokenizer.bos_token_id
    policy_args["architecture"]["eos_token_id"] = tokenizer.eos_token_id
    policy_args["architecture"]["pad_token_id"] = tokenizer.pad_token_id
    alg_kwargs = {
        "policy": policy_cls,
        "env": env,
        "policy_kwargs": policy_args,
    }
    alg_kwargs = {**alg_kwargs, **alg_config.get("args")}
    wrapper = WrapperRegistry.get(alg_config["id"])
    alg = wrapper(
        alg_cls,
        alg_kwargs,
        alg_config["kl_div"]["coeff"],
        tracker,
        alg_config["kl_div"].get("target_kl", None),
        alg_config["kl_div"].get("norm_reward", False),
    )
    alg.load_from_dict(alg_state)
    return alg


class OnPolicyTrainer(TrainerWarmStartMixin):
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        datapool_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        env_config: Dict[str, Any],
        on_policy_alg_config: Dict[str, Any],
        train_eval_config: Dict[str, Any],
        tracker: Tracker = None,
        experiment_name: str = "",
        dev: bool = False,
    ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._on_policy_alg_config = on_policy_alg_config
        self._train_eval_config = train_eval_config
        self._tracker = tracker
        self._experiment_name = experiment_name
        self._dev = dev
        self._setup()

    def _setup(self):
        # load trainer state from available previous checkpoint if available
        self.load_trainer_state(self._tracker)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(
            self._datapool_config, self._tokenizer, dev=self._dev
        )
        self._env = build_env(
            self._env_config,
            self._reward_fn,
            self._tokenizer,
            self._samples_by_split["train"],
        )
        self._alg = build_alg(
            self._on_policy_alg_config,
            self._env,
            self._tokenizer,
            self._tracker,
            self._policy_state_dict,
            self._alg_state_dict,
        )

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._n_steps_per_iter = self._env.num_envs * self._alg.n_steps

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get("generation_kwargs", None)

    def _evaluate_on_datapools(self, epoch: int, splits: List[str] = ["val", "test"]):
        for split in splits:
            evaluate_on_samples(
                policy=self._alg.policy,
                tokenizer=self._tokenizer,
                samples=self._samples_by_split[split],
                batch_size=self._eval_batch_size,
                metrics=self._metrics,
                epoch=epoch,
                split_name=split,
                tracker=self._tracker,
                gen_kwargs=self._eval_gen_kwargs,
            )

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=iter_start, splits=["val"])

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            # current state
            self._trainer_state["current_iter"] = epoch

            # inner rollout and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            # save the policy checkpoint
            if (epoch + 1) % self._train_eval_config.get("save_every", 20) == 0:
                self.save_trainer_state(
                    self._tracker, self._alg.policy, self._trainer_state
                )

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                self._evaluate_on_datapools(epoch=epoch, splits=["val"])

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch, splits=["val"])

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(self._alg.policy.get_language_model())
