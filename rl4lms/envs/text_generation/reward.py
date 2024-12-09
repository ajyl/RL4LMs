from abc import ABC, abstractclassmethod

import torch
from rl4lms.envs.text_generation.observation import Observation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rl4lms.envs.text_generation.metric import (
    IntentAccuracyDailyDialog,
)
import numpy as np
from typing import List, Dict, Any


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        """
        Callable for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
            done (bool): whether the episode is finished or not
            meta_info (dict) - other information regarding textual sample
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class BatchedRewardFunction(ABC):
    """
    Computes rewards for several instances at once
    """

    @abstractclassmethod
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        An abstract class for batched reward functions for text generation
        """
        raise NotImplementedError


### Automated reward functions ###########################


class CommonGenPenaltyShapingFunction(RewardFunction):
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            prompt_text = next_observation.prompt_or_input_text
            prefix = "generate a sentence with: "
            concept_n_grams = prompt_text.split(prefix)[1][:-1]

            if (
                concept_n_grams.lower() in next_observation.context_text.lower()
                or prefix in next_observation.context_text.lower()
                or "generate" in next_observation.context_text.lower()
                or "sentence" in next_observation.context_text.lower()
            ):
                penalty_score = -1
            else:
                penalty_score = 0
            return penalty_score
        return 0


class BatchedCommonGenPenaltyShapingFunction(BatchedRewardFunction):
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_info: Dict[str, Any] = None,
    ) -> List[float]:
        scores = []
        for done, prompt_text, gen_text in zip(dones, prompt_texts, gen_texts):
            if done:
                prefix = "generate a sentence with: "
                concept_n_grams = prompt_text.split(prefix)[1][:-1]

                if (
                    concept_n_grams.lower() in gen_text.lower()
                    or prefix in gen_text.lower()
                    or "generate" in gen_text.lower()
                    or "sentence" in gen_text.lower()
                ):
                    penalty_score = -1
                else:
                    penalty_score = 0
                scores.append(penalty_score)
        return scores


class LearnedBatchedRewardFunction(BatchedRewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        # collect generations at done steps
        gens = []
        indices_with_done = []
        rewards = torch.zeros(len(prompt_texts))
        for ix, (prompt, gen, ref, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, dones)
        ):
            if done:
                generated_text = prompt if self._include_prompt_for_eval else ""
                generated_text += gen
                gens.append(generated_text)
                indices_with_done.append(ix)

        # compute rewards at once
        with torch.no_grad():
            encoded = self._metric_tokenizer(
                gens, return_tensors="pt", truncation=True, padding=True
            )
            outputs = self._metric_model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            scores = torch.softmax(outputs.logits, dim=1)
            scores = scores[:, self._label_ix].flatten().cpu()
            rewards[indices_with_done] = scores

        return rewards



#############################################################################

########## Learned Reward Functions##########################################


class LearnedRewardFunction(RewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_text = (
                current_observation.prompt_or_input_text
                if self._include_prompt_for_eval
                else ""
            )
            generated_text += next_observation.context_text

            with torch.no_grad():
                encoded = self._metric_tokenizer(
                    generated_text, return_tensors="pt", truncation=True, padding=True
                )
                outputs = self._metric_model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits.flatten(), dim=0)
                score = scores[self._label_ix].item()
                return score
        return 0


class PARENTRewardFunction(RewardFunction):
    """
    PARENT F1 score as the reward
    """

    def __init__(self) -> None:
        super().__init__()
        self._metric = ParentToTTo()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_texts = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, generated_texts, None, meta_infos)
            reward = scores["table_to_text/parent_overall_f_score"][0][0]
            return reward
        return 0


class RougeLMaxRewardFunction(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = RougeLMax(**args)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, predicted, references, meta_infos)
            reward = scores["lexical/rouge_l_max"][0][0]
            return reward
        return 0


class TER(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = TERMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/ter"][1]
            score = 1 - score
            return score
        return 0


class chrF(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = chrFmetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute score at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/chrf"][1]
            return score
        return 0


class IntentAccuracy(BatchedRewardFunction):
    def __init__(
        self, shape: bool = True, intent_coeff: float = 1.0, auto_coeff: float = 1.0
    ) -> None:
        super().__init__()
        self._metric = None
        self._shape = shape
        self._intent_coeff = intent_coeff
        self._auto_coeff = auto_coeff
        breakpoint()
        # self._shaping_metric = MeteorMetric()

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        if self._metric is None:
            self._metric = IntentAccuracyDailyDialog()

        # compute rewards for finished episodes only
        rewards = np.zeros(len(gen_texts))

        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []
        done_meta_infos = []
        done_ixs = []
        for ix, (prompt, gen, ref, meta_info, done) in enumerate(
            zip(prompt_texts, gen_texts, ref_texts, meta_infos, dones)
        ):
            if done:
                done_prompt_texts.append(prompt)
                done_gen_texts.append(gen)
                done_ref_texts.append(ref)
                done_meta_infos.append(meta_info)
                done_ixs.append(ix)

                if self._shape:
                    score = self._shaping_metric.compute(
                        done_prompt_texts, done_gen_texts, done_ref_texts
                    )
                    rewards[ix] = self._auto_coeff * score["lexical/meteor"][1]

        scores = self._metric.compute(
            done_prompt_texts, done_gen_texts, done_ref_texts, done_meta_infos
        )["intent/accuracy"][0]
        rewards[done_ixs] += self._intent_coeff * np.array(scores)
        return rewards.tolist()


if __name__ == "__main__":
    predictions = "hello there general kenobi"
    references = ["hello there general kenobi", "hello there!!"]
    observation = Observation(
        None, None, None, None, None, predictions, references, None, None, None, None
    )

    reward_fn = MeteorRewardFunction()
    print(reward_fn(None, None, observation, True))

    reward_fn = chrF()
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeCombined()
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rouge1")
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rouge2")
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rougeL")
    print(reward_fn(None, None, observation, True))

    reward_fn = BERTScoreRewardFunction(language="en")
    print(reward_fn(None, None, observation, True))

    reward_fn = BLEURewardFunction()
    print(reward_fn(None, None, observation, True))

    reward_fn = BLEURTRewardFunction()
    print(reward_fn(None, None, observation, True))
