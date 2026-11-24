"""Collect a batch of pilot trajectories (T14).

Resumable: safe to re-run after a Groq free-tier interruption — already-finished episodes are
skipped, and errored/partial ones are retried. Needs ALFWorld installed (Colab/WSL2 — not
native Windows) and GROQ_API_KEY set.
"""

import argparse

import yaml

from tripwire.agent.groq_client import make_groq_llm
from tripwire.env.alfworld_env import make_env, reset_episode, step_episode
from tripwire.paths import resolve_config_path
from tripwire.runner.batch_runner import run_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="number of pilot episodes (30-50)")
    parser.add_argument("--config", default="base_config.yaml", help="ALFWorld env config")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument("--out-dir", default="data/raw/pilot")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(resolve_config_path(args.agent_config), "r") as f:
        cfg = yaml.safe_load(f)

    env = make_env(args.config)
    llm_call = make_groq_llm(cfg["model"])

    summaries = run_batch(
        reset_fn=lambda: reset_episode(env),
        step_fn=lambda action: step_episode(env, action),
        llm_call=llm_call,
        model=cfg["model"],
        output_dir=args.out_dir,
        num_episodes=args.n,
        max_steps=cfg["max_steps"],
        max_retries=cfg["max_retries"],
        backoff_base_seconds=cfg["backoff_base_seconds"],
        backoff_max_seconds=cfg["backoff_max_seconds"],
        seed=args.seed,
        history_window_steps=cfg.get("history_window_steps", 10),
        max_consecutive_errors=cfg.get("max_consecutive_errors", 3),
    )

    counts: dict[str, int] = {}
    for summary in summaries:
        counts[summary.outcome] = counts.get(summary.outcome, 0) + 1
    print(f"ran {len(summaries)} episode(s) this invocation: {counts}")


if __name__ == "__main__":
    main()
