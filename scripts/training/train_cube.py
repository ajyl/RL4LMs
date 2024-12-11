import os
from argparse import ArgumentParser

import yaml

from rl4lms.envs.cube.logging_utils import Tracker
from rl4lms.envs.cube.training_utils import (
    OnPolicyTrainer,
)


def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool,
    dev: bool,
):

    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )

    # instantiate the trainer here
    trainer = OnPolicyTrainer(
        tokenizer_config=config["tokenizer"],
        datapool_config=config["datapool"],
        reward_config=config["reward_fn"],
        env_config=config["env"],
        on_policy_alg_config=config["alg"],
        train_eval_config=config["train_evaluation"],
        tracker=tracker,
        experiment_name=experiment_name,
        dev=dev,
    )
    trainer.train_and_eval()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--exp",
        type=str,
        help="WANDB experiment name",
        default="rl4lm_experiment",
    )
    parser.add_argument(
        "--entity_name", type=str, help="WANDB entity name", default=None
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true", help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--dev", action="store_true"
    )
    args = parser.parse_args()
    project_name = "cube_ppo"

    base_path = "/n/holylabs/LABS/wattenberg_lab/Lab/ajyl_tmp/cube_ppo"

    main(
        args.config_path,
        project_name,
        args.exp,
        base_path,
        args.entity_name,
        args.log_to_wandb,
        args.dev,
    )
