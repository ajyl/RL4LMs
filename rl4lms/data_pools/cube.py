import random
import torch
from abc import abstractclassmethod
from dataclasses import dataclass
from typing import Any, List, Dict
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from rl4lms.envs.cube.tokenizer import CubeTokenizer


@dataclass(init=True)
class Sample:
    id: str
    cube_state: torch.Tensor
    cube_state_str: str
    solution: torch.Tensor
    solution_str: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    meta_data: Dict[str, Any] = None


class CubeDataPool:
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> "CubeDataPool":
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List["CubeDataPool"]:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix:end_ix]))
            start_ix = end_ix
        return pools


def encode(examples, tokenizer=None):
    combined_text = []
    for i in range(max(len(examples["scramble"]), len(examples["solution"]))):
        examples["scramble"][i] = examples["scramble"][i].lower()
        if not (examples["solution"][i]):
            examples["solution"][i] = ""
        combined_text.append(examples["scramble"][i] + ":" + examples["solution"][i])
    return tokenizer.encode(combined_text)


class CubeData(CubeDataPool):

    @classmethod
    def prepare(
        cls, split: str, tokenizer: CubeTokenizer, seed: int, dev=False
    ) -> "CubeDataPool":
        data_filepath = (
            f"/n/home01/ajyl/RL4LMs/data/data222_complete_faces_threshold_2_train_2starts.csv",
        )
        train_split_ratio = 0.0006
        if dev:
            data_filepath = (f"/n/home01/ajyl/RL4LMs/data/dev_data.csv",)
            train_split_ratio = 0.01
        dataset = load_dataset(
            "csv",
            data_files=data_filepath,
        )
        dataset = dataset.map(encode, fn_kwargs={"tokenizer": tokenizer}, batched=True)

        train_testvalid = dataset["train"].train_test_split(test_size=train_split_ratio)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5)

        dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "val": test_valid["train"],
                "test": test_valid["test"],
            }
        )

        ds_split = dataset[split]
        if split == "train":
            ds_split = ds_split.shuffle(seed)
        n_samples = len(ds_split)
        samples = []
        for ix, item in tqdm(enumerate(ds_split), desc="Loading data", total=n_samples):

            # item: {
            #    scramble: str (len 24): 'ulrurdrubffflddrldlufbbb'
            #    solution: str (len *): 'RRFRR'
            #    input_ids: List[int] (len max_length)
            #    attention_mask: List[int] (len max_length) (1s, 0s).
            #    labels: same as input_ids?
            # }
            sample = Sample(
                id=f"{split}_{ix}",
                cube_state=torch.tensor(item["input_ids"][:24]),
                cube_state_str=item["scramble"],
                solution=torch.tensor(item["input_ids"][25:]),
                solution_str=item["solution"],
                input_ids=torch.tensor(item["input_ids"]),
                attention_mask=torch.tensor(item["attention_mask"]),
                labels=torch.tensor(item["labels"]),
            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance
