"""Run one real ALFWorld + Groq episode and write its trajectory (T13 sanity check).

Needs ALFWorld installed (Colab/WSL2 — not native Windows) and GROQ_API_KEY set.
"""

import argparse

import yaml

from tripwire.agent.groq_client import make_groq_llm
from tripwire.agent.react_agent import run_episode
from tripwire.env.alfworld_env import make_env, reset_episode, step_episode
from tripwire.trajectory.logger import TrajectoryLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base_config.yaml", help="ALFWorld env config")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--out-dir", default="data/raw/pilot")
    parser.add_argument("--episode-id", default="ep_manual_0001")
    args = parser.parse_args()

    with open(args.agent_config, "r") as f:
        cfg = yaml.safe_load(f)

    env = make_env(args.config)
    llm_call = make_groq_llm(cfg["model"])

    steps, summary = run_episode(
        reset_fn=lambda: reset_episode(env),
        step_fn=lambda action: step_episode(env, action),
        llm_call=llm_call,
        episode_id=args.episode_id,
        model=cfg["model"],
        max_steps=cfg["max_steps"],
        max_retries=cfg["max_retries"],
        backoff_base_seconds=cfg["backoff_base_seconds"],
        backoff_max_seconds=cfg["backoff_max_seconds"],
    )

    with TrajectoryLogger(args.out_dir, args.episode_id) as logger:
        for step in steps:
            logger.log_step(step)
        logger.log_episode_summary(summary)

    print(f"episode_id={summary.episode_id} outcome={summary.outcome} num_steps={summary.num_steps}")


if __name__ == "__main__":
    main()
