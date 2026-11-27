"""Split the cleaned T27 corpus into calibration/test sets (T28).

50/50, stratified by outcome so each half has the same success/failure mix as the whole
corpus, with a fixed seed so re-running this script reproduces the exact same split.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20250601  # fixed - never change without deliberately re-splitting


def load_summaries(dir_path: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(dir_path.glob("ep_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") == "episode_summary":
                out[rec["episode_id"]] = rec
                break
    return out


def make_splits(summaries: dict[str, dict], seed: int = SEED) -> tuple[list[str], list[str]]:
    by_outcome: dict[str, list[str]] = defaultdict(list)
    for episode_id, rec in summaries.items():
        by_outcome[rec["outcome"]].append(episode_id)

    calibration_ids: list[str] = []
    test_ids: list[str] = []
    rng = random.Random(seed)

    for _outcome, ids in sorted(by_outcome.items()):
        ids = sorted(ids)  # deterministic starting order before shuffling
        rng.shuffle(ids)
        half = len(ids) // 2
        calibration_ids.extend(ids[:half])
        # an odd count gives its extra episode to test, so test is never smaller
        test_ids.extend(ids[half:])

    return sorted(calibration_ids), sorted(test_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="data/raw/full")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    summaries = load_summaries(Path(args.corpus_dir))
    calibration_ids, test_ids = make_splits(summaries, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_ids.txt").write_text("\n".join(calibration_ids) + "\n", encoding="utf-8")
    (out_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")

    cal_counts = Counter(summaries[i]["outcome"] for i in calibration_ids)
    test_counts = Counter(summaries[i]["outcome"] for i in test_ids)
    print(f"loaded {len(summaries)} episodes from {args.corpus_dir}")
    print(f"calibration: {len(calibration_ids)} episodes, outcomes: {dict(cal_counts)}")
    print(f"test:        {len(test_ids)} episodes, outcomes: {dict(test_counts)}")


if __name__ == "__main__":
    main()
