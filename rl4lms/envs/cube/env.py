from cmath import inf
from typing import Dict, Tuple, Optional, List, Any, Callable

import torch
from gymnasium import Env, spaces
from gymnasium.spaces.dict import Dict as DictSpace
from gymnasium.spaces.discrete import Discrete
from gymnasium.core import ObsType
from rl4lms.data_pools.cube import Sample
from rl4lms.envs.cube.reward import BatchedRewardFunction, RewardFunction
from rl4lms.envs.cube.observation import Observation
from rl4lms.envs.cube.tokenizer import CubeTokenizer
from rl4lms.core_components.sampler import PrioritySampler


class CubeEnv(Env):
    def __init__(
        self,
        tokenizer: CubeTokenizer,
        reward_function: RewardFunction,
        # vocab_size: int,
        max_prompt_length: int,
        samples: Tuple[List[Sample], float],
        max_episode_length: int = 12,
        priority_scale: float = 0.0,
        terminate_on_eos: bool = False,
        context_start_token: Optional[int] = None,
        prompt_truncation_side: str = "left",
    ):
        """
        A generic RL environment to generate textual sequences.
        For eg: text generation, summarization, machine translation, text simplification
        Args:
            tokenizer (CubeTokenizer): encoder
            reward_function (RewardFunction): reward functiom
            max_prompt_length (int): maximum prompt length.
            samples (Tuple[List[Sample], float]): list of samples
            max_episode_length (int, optional): Max steps to the model Defaults to 512.
            priority_scale (float, optional): weight for the priority sampler Defaults to 0.0.
            terminate_on_eos (bool, optional): whether to terminate on EOS. Defaults to False.
            context_start_token (bool, optional): start token for the context (For Encoder-Decoder models! )
            prompt_truncation_side (str): truncation side for prompt text (Defaults to "left")
        """
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.max_steps = max_episode_length
        self._max_text_length = max_prompt_length
        self._terminate_on_eos = terminate_on_eos
        self._context_start_token = context_start_token
        self._prompt_truncation_side = prompt_truncation_side
        super().__init__()

        # set the observation and action space here
        self._vocab_size = self.tokenizer.vocab_size
        self.observation_space = DictSpace(
            {
                # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
                # while creating rollout buffers, observations are concatenated for each key
                "prompt_or_input_encoded_pt": spaces.Box(
                    low=0, high=self._vocab_size, shape=(self._max_text_length,)
                ),
                "prompt_or_input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length,)
                ),
                "context_encoded_pt": spaces.Box(
                    low=0, high=self._vocab_size, shape=(self.max_steps,)
                ),
                "context_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self.max_steps,)
                ),
                "input_encoded_pt": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length + self.max_steps,),
                ),
                "input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length + self.max_steps,)
                ),
            }
        )
        self.action_space = Discrete(n=self._vocab_size)
        self.sampler_for_replaying = PrioritySampler(priority_scale=priority_scale)
        for sample, weight in samples:
            self.sampler_for_replaying.add(sample, weight)

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__previous_obs = None
        self.__time_step = None

    def _get_info(self):
        prev_output = None
        meta_info = None
        if self.__previous_obs is not None:
            prev_output = self.__previous_obs.context_text
            meta_info = self.__previous_obs.meta_info
        return {
            "output": self.__current_obs.context_text,
            "action_history": self.__current_obs.action_history,
            "reference_text": self.__current_obs.target_or_reference_texts,
            "prompt_text": self.__current_obs.prompt_or_input_text,
            "prev_output": prev_output,
            "meta_info": meta_info,
        }

    def step(
        self, action: int
    ) -> Tuple[Dict[str, torch.tensor], int, bool, bool, dict]:
        self.__time_step += 1

        # previous obs
        self.__previous_obs = self.__current_obs

        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self.tokenizer)

        # decide if the episode is finished or not
        done = (action == self.tokenizer.eos_token_id and self._terminate_on_eos) or (
            self.__time_step == self.max_steps
        )

        # compute reward
        if not isinstance(self.reward_function, BatchedRewardFunction):
            reward = (
                None
                if self.reward_function is None
                else self.reward_function(
                    self.__previous_obs,
                    action,
                    self.__current_obs,
                    done,
                    self.__current_obs.meta_info,
                )
            )
        else:
            reward = -inf  # will be overridden later

        # populate additional info
        info = self._get_info()
        return self.__current_obs.to_dict(), reward, done, info

    def reset(
        self, seed=None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment and starts a new episode
        """
        super().reset(seed=seed)

        sample = None
        if options is not None:
            sample = options.get("sample")

        # gets a new sample if not provided
        if sample is None:
            sample = self.sampler_for_replaying.sample(size=1)[0]
        self.__current_sample = sample

        # init the observation
        self.__current_obs = Observation.init_from_sample(
            sample,
            self.tokenizer,
            self._max_text_length,
            self.max_steps,
            self._prompt_truncation_side,
            self._context_start_token,
            sample.meta_data,
        )

        # start the time step counter
        self.__time_step = 0

        dict_observation = self.__current_obs.to_dict()
        info = self._get_info()
        return dict_observation, info

    def render(self):
        pass

    def close(self):
        pass

    def add_sample(self, sample: Sample, weight: int = 1.0):
        self.sampler_for_replaying.add(sample, weight)
