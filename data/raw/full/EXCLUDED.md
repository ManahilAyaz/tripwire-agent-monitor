# Excluded episodes

These 11 episodes are moved aside into `_excluded/`, not deleted, from the 283-episode
full-corpus run (`corpus_full (283).zip`). The remaining 272 files directly in `data/raw/full/`
are the clean corpus. See react_agent.py's `example_task_inputs`/`avoid_example_tasks` and
`env/alfworld_env.py::gamefile_to_task_id` for what `task_id` means and how contamination
avoidance already works at generation time; this file documents a post-hoc check of what still
got through due to ALFWorld/TextWorld's own game-sampling behavior repeating across Kaggle
session restarts (see T27 sampling-bug fix in `src/tripwire/runner/batch_runner.py`).

| file | task_id | reason |
|---|---|---|
| ep_0024.jsonl | pick_and_place_simple-CellPhone-None-Drawer-302/trial_T20190907_235359_470335 | pilot overlap (also in `data/raw/pilot/`) |
| ep_0124.jsonl | pick_and_place_simple-Tomato-None-Microwave-6/trial_T20190907_235735_175320 | pilot overlap (also in `data/raw/pilot/`) |
| ep_0192.jsonl | pick_heat_then_place_in_recep-Egg-None-GarbageCan-2/trial_T20190909_101128_479012 | pilot overlap (also in `data/raw/pilot/`) |
| ep_0207.jsonl | pick_cool_then_place_in_recep-Apple-None-Microwave-12/trial_T20190908_221148_854032 | pilot overlap (also in `data/raw/pilot/`) |
| ep_0122.jsonl | pick_cool_then_place_in_recep-Potato-None-CounterTop-30/trial_T20190908_111718_406246 | intra-corpus duplicate (first copy kept: ep_0003.jsonl) |
| ep_0126.jsonl | pick_heat_then_place_in_recep-Plate-None-Fridge-13/trial_T20190907_035716_975895 | intra-corpus duplicate (first copy kept: ep_0021.jsonl) |
| ep_0144.jsonl | pick_cool_then_place_in_recep-Bowl-None-DiningTable-23/trial_T20190908_140248_782127 | intra-corpus duplicate (first copy kept: ep_0020.jsonl) |
| ep_0252.jsonl | pick_and_place_simple-SprayBottle-None-CounterTop-417/trial_T20190909_091308_390069 | intra-corpus duplicate (first copy kept: ep_0178.jsonl) |
| ep_0268.jsonl | pick_and_place_simple-Statue-None-CoffeeTable-228/trial_T20190906_185513_812051 | intra-corpus duplicate (first copy kept: ep_0087.jsonl) |
| ep_0271.jsonl | pick_two_obj_and_place-Pencil-None-Desk-314/trial_T20190907_044915_976409 | intra-corpus duplicate (first copy kept: ep_0142.jsonl) |
| ep_0278.jsonl | pick_and_place_simple-Pencil-None-GarbageCan-316/trial_T20190907_165839_449092 | intra-corpus duplicate (first copy kept: ep_0247.jsonl) |

For each intra-corpus duplicate, the *later* copy (higher episode number) is the one excluded;
the first occurrence stays in the clean 272.

## Root cause

Not random-sampling-with-replacement collisions (that was an earlier, incorrect guess). ALFWorld
delegates per-episode game selection to TextWorld's `TextworldBatchGymEnv`, which samples
**without** replacement from a shuffled cycle of all game files — but that cycle is seeded with
a hardcoded `1234` local to the environment object, unrelated to this project's own
`random.seed()`/`--seed` argument. Every time a Kaggle session restart forced `run_batch.py` to
re-create its `env` from scratch, that cycle reset back to position 0 of the *same* fixed
1234-shuffle — so the first episodes run after a restart could replay games already consumed
earlier in this corpus, or in the pilot's own earlier (separate-process) run. Fixed for future
runs by `batch_runner.sequential_game_order` (persists our own shuffled order and cursor to disk
instead of relying on the env's internal, restart-unsafe cycle).

## Clean corpus summary (272 episodes)

- Outcomes: 206 failure, 66 success.
- 0 contaminated (none match the 12 ReAct few-shot example task_inputs).
- 0 overlap with the 40-episode pilot corpus (`data/raw/pilot/`).
- All 272 task_ids are unique.
