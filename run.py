import argparse
import datetime

from rl_algorithms import build_agent
from rl_algorithms.common import helper_functions as common_utils
from rl_algorithms.utils import Config

from env import PhantomDummyEnv

def parse_args() -> argparse.Namespace:
    """Set input arguments."""
    parser = argparse.ArgumentParser(description="Demo code of 'Deep reinforcement learning for guidewire navigation in coronary artery phantom' paper")

    parser.add_argument("--seed", type=int, default=777, help="random seed for reproducibility")
    parser.add_argument(
        "--load-from",
        type=str,
        default="data/ckpt.pt",
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument("--episode-num", type=int, default=3, help="total episode num")
    parser.add_argument(
        "--cfg-path", type=str, default="config.py", help="config path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.cfg_path)
    env = PhantomDummyEnv()

    # Set a random seed
    common_utils.set_random_seed(args.seed, env)

    cfg.agent.env_info = dict(
        name=env.name,
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_discrete=True,
    )
    # Initialize agent
    args.test = True
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    cfg.agent["log_cfg"] = dict(agent=cfg.agent.type, curr_time=curr_time)
    build_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, build_args)

    agent.test()


if __name__ == "__main__":
    main()