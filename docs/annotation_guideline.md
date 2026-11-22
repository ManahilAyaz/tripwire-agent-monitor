# Annotation Guideline v1.1 (T09)

| Item | Value |
|---|---|
| Project | Tripwire: Mid-Trajectory Failure Detection for Long-Horizon AI Agents |
| Task | T09, Week 1 (November 16-22, 2026) |
| Author | Aqsa Zaib |
| Status | Draft v1.1, updated for Trajectory Log Schema v1.0. Not yet reviewed or approved by the team or the supervisor. Not tested on real trajectories. |
| Related files | `docs/taxonomy.md` (T07), `docs/candidate_signals.md` (T08), `configs/rules.yaml` (T10, **not to be consulted while annotating**) |

All examples in this document are made up for illustration. None of them come from real agent runs. Step numbers are 1-indexed as in schema v1.0.

## A. Annotation objectives

We annotate trajectories by hand so that we have a reference to compare Tripwire against. The annotations are used to:

1. Check whether the monitor flags the right steps (is the alarm on a real pattern?).
2. Measure how early the monitor flags a failing episode compared with the first defensible failure point (section E).
3. Count false alarms, such as flags on episodes that recover and succeed.
4. Find missing categories and weak rules, so the taxonomy and signals can be improved.

Annotators use only what is observable in the log: `task_input`, `initial_observation`, and for each step `step`, `action`, `observation` and `parse_ok`; plus the raw `outcome`, but only in Pass 2. Annotators do **not** use:

- `thought` (see section L, decision pending);
- `retry_count` and `timestamp` (not evidence of agent behaviour);
- hidden or imagined reasoning of the agent.

Annotation covers failure detection only. We record what happened. We do not annotate what a corrector should have done (FYP-II).

### Workflow for one episode

1. **Pass 1 (outcome hidden):** read the steps in order without seeing `outcome`. Mark steps that look suspicious and note the category. Use the team's loader (`load_prefix(path, k)` from T12) rather than opening the raw file, because the raw file's last line is the summary that contains `outcome`. Whether `load_prefix` with k = full length is the agreed annotator view is for the T12 owners to confirm. Do not write a separate loader.
2. **Pass 2:** look at the raw `outcome`. Set the episode label (section B). Decide which patterns count as failure-related (sections E and F).
3. Fill in the annotation records (section I).

Reading in order matters. The outcome is used only to decide which pattern counts as the failure pattern, not to decide whether an early step looks suspicious.

A log file with no `episode_summary` line is unfinished. The batch runner re-runs it, so it is **not annotated**.

## B. Episode-level labels

The raw `outcome` is written by the logger. The annotation `episode_label` is derived from it by this fixed mapping. Annotators do not re-decide it.

| Raw `outcome` (schema v1.0) | Annotation `episode_label` | `termination_reason` | Meaning |
|---|---|---|---|
| `success` | `success` | `task_completed` | The environment reported the task won. |
| `failure` | `failure` | `environment_end` | The environment ended the episode without a win. |
| `max_steps` | `failure` | `step_limit` | The agent hit the step cap without a win. |
| `error` | `inconclusive` | `infrastructure_error` | Infrastructure problem (API down, environment crash). **Not a task result.** |

Notes:

- `termination_reason` is derived from the raw outcome, so the distinction between `failure` and `max_steps` is not lost even though both are `failure` here. The episode record also stores `raw_outcome`.
- `infrastructure_error` is deliberately generic. The schema does not say *what* went wrong, and an `error` episode is not necessarily a crash (it may be an API outage). Do not write "crash" unless evidence outside the log (for example the runner's own log) says so, and then only in `notes`.
- `validate_logs.py` excludes `error` episodes from the corpus, so `inconclusive` episodes will be rare. They are still defined so that the mapping is complete.
- A raw `outcome` outside the four values is a schema violation. Do not annotate it; report it to the logger owner.
- An episode with many suspicious steps can still be `success`. An episode with no suspicious steps can still be `failure`.

## C. Step-level labels

Each step gets one label. The label describes the observable behaviour at that step, in the context of earlier steps only.

| Label | Meaning | `category_id` | Allowed when episode label is |
|---|---|---|---|
| `normal` | The action is reasonable given what has been observed so far. Includes ordinary exploration, a single `look` or `inventory`, and retries after the situation changed. | `null` | any |
| `suspicious` | The step shows part of a taxonomy pattern, but the evidence is not yet enough (or the pattern was left behind by recovery). | required | any |
| `failure_related` | The step is at or after the first defensible failure point of a failure pattern and still belongs to that pattern. | required | `failure` only |
| `uncertain` | The annotator cannot decide between labels because of ambiguity or missing information (section G). | optional | any |

Rules:

1. Steps not listed in the annotation file count as `normal`, but only if the episode record says `all_steps_reviewed` is true.
2. `failure_related` is used only in episodes labelled `failure`. In `success` and `inconclusive` episodes the strongest label is `suspicious`.
3. A step before the first defensible failure point is at most `suspicious`, even if the episode later fails.
4. One action with clear evidence of environment rejection, followed by a different sensible action, is labelled `suspicious` (TX-04). It is a watch label, not a verdict. Do not assign TX-04 when the only evidence is an ambiguous response such as "Nothing happens."
5. Do not use `suspicious` for ordinary exploration, harmless double-checks, or actions that are only unusual in style.
6. **Parse failures.** A step with `parse_ok = false` is not an agent action: the logger substituted `look`. It is never evidence for TX-01, TX-03, TX-04 or TX-05, and the substituted `look` is never a "repeated look". Label such a step `suspicious` with `category_id` = `TX-99` and note `parse_failure`, so that how often it happens can be counted. TX-99 has no evidence bar, so parse failures alone never create a first defensible failure point (record no FDFP and explain in the notes). Whether they deserve their own category is an open team decision (section L).
7. **Environment response and action validity.** A well-formed action (`parse_ok = true`) is not automatically invalid just because its observation says "Nothing happens." TX-04 should be assigned only when the observation or available context provides sufficient evidence that the environment rejected the action as invalid or impossible. If the observation is ambiguous, use `uncertain` with `ambiguous_action` or `insufficient_evidence`. A repeated action with no effect may support TX-03 only when the evidence shows that the same action was attempted again without a relevant state change. Do not treat `parse_ok` as proof that an action was valid in the environment.

## D. Failure categories

Use the IDs from `docs/taxonomy.md`: TX-01 to TX-08, and TX-99 for anything else. IDs are unchanged from v1.

Choosing the category:

1. List every category whose observable evidence is visible in the steps.
2. The main category is the one that best explains why the behaviour is a problem. Precedence guide:
   - A repeated action that keeps getting a no-effect reply: TX-03 (secondary TX-01, TX-04 if they also fit).
   - A repeated action with meaningful, changing observations: TX-01.
   - A repeating block of different actions: TX-02 (before TX-01).
   - Same observation again and again while actions vary: TX-05.
   - A semantic problem you can name (wrong goal, ignored evidence): TX-07 or TX-08 (before TX-06).
   - TX-06 only when nothing more specific fits.
3. Add at most two secondary categories in `secondary_category_ids`.
4. Use TX-99 with a note if nothing fits. Never stretch a category to cover something different.
5. All steps of one pattern share the same main category and `pattern_id`.

Annotators judge TX-02 by reading the steps themselves. The GRAPHECTORY `cycle_length` value is a detector input and is not shown to annotators.

## E. First defensible failure point

The first defensible failure point (FDFP) is the earliest step at which the observable evidence for a failure pattern is strong enough. It is recorded once per failure-related pattern, and only in episodes labelled `failure` (raw `failure` or `max_steps`).

How to find it:

1. Identify the failure pattern: the pattern that the agent never left before the episode ended (section F covers patterns that were left).
2. Walk forward through the steps of that pattern and find the first step t where the evidence condition for its category (table below) is met.
3. Check using only steps 1 to t. Ask: if a reviewer saw only steps 1 to t (exactly what `load_prefix(path, t)` returns), would they agree the pattern is established? If not, move later.
4. That step is the FDFP. The steps before it in the same pattern are `suspicious` (record the first one as `first_suspicious_step`). The FDFP step and later pattern steps are `failure_related`.

The first unusual action is not the FDFP unless it alone meets the condition, and most single actions do not.

### E.1 Evidence conditions (provisional, judgement-based)

Each category has a **floor** (the minimum evidence before an FDFP can exist at all, so one event never counts) and a **condition** that a reviewer must judge to be true using steps 1 to t only. Only steps with `parse_ok = true` count as agent steps.

| Category | Floor (minimum) | Condition: the FDFP is the first step where a reviewer agrees that... |
|---|---|---|
| TX-01 | The same normalized action at 2 or more consecutive agent steps | ...the repeated actions provide no meaningful new information, and there is no visible reason why repeating the action should now produce a different result. |
| TX-02 | A repeating block of different actions and corresponding observations, observed at least twice | ...the same action-observation sequence repeats without meaningful new information, indicating that the agent is cycling through the same behaviour. |
| TX-03 | The same action attempted twice with no effect | ...nothing that could change the outcome happened between the attempts, and the agent is trying the same thing again. |
| TX-04 | At least two actions with clear evidence of environment rejection | ...the actions show a repeated pattern of attempting actions that the environment explicitly indicates are invalid or impossible, rather than a one-off mistake. |
| TX-05 | The same normalized observation at 2 consecutive steps | ...the repeated observation reflects a lack of meaningful state change despite the agent taking actions. |
| TX-06 | A stretch of steps with no new information (reviewer's judgement; write the length in `note`) | ...the agent is circling through places and objects already seen, with nothing new learned or carried. |
| TX-07 | 2 consecutive actions unrelated to `task_input` | ...there is no plausible link to a sub-goal. |
| TX-08 | 2 steps after an observation that showed what the task needed | ...the agent has moved on without using it and has no visible reason to. |

For consecutive-action patterns (TX-01), steps must have consecutive logged step numbers, and every step in the sequence must have `parse_ok = true`. A step with `parse_ok = false` breaks the sequence. Do not skip a parse-failure step and join the valid actions on either side as if they were consecutive.

### E.2 Why these are conditions and not counts (independence from the detector)

If the annotation bars were set equal to what the rules detect (for example "3 in a row" in both places), the evaluation would be circular: the detector would look perfect by construction. To prevent that:

1. The conditions above are judgements about the evidence. They contain no detector threshold, and detector thresholds (`configs/rules.yaml`, T10/T19) are not derived from them or copied from them.
2. Annotators do not look at `configs/rules.yaml`, detector output or `cycle_length` before finishing an episode.
3. Any floor or condition change goes into the guideline version note, with the reason. It is never changed to match a detector value.
4. Counts seen while annotating (how long was the run at the FDFP?) are recorded in `note`. After the pilot the team may decide to fix counts for better agreement, and should then keep those pilot episodes out of the detector's threshold tuning. That decision is open (section L).

If no pattern in a failing episode meets its condition (for example the agent behaved sensibly but ran out of steps), record no FDFP and write a note. Do not invent one.

## F. Recovery handling

A pattern is recovered when the agent leaves it by its own behaviour and continues sensibly.

How to annotate:

1. Keep the steps of the pattern as `suspicious` with their category. Even if the pattern reached its evidence condition, steps in a pattern that was left are not `failure_related`.
2. The **break step** is the first step after the pattern in which the agent does something different that changes the state or provides new information. Set `recovery_step` to this break step. Mark the pattern as recovered if it does not resume during the next 3 available steps after the break step. If the episode ends before 3 further steps are available, record the recovery as provisional in `notes` and explain how many subsequent steps were observed. Do not treat the break step alone as sufficient evidence of recovery.
3. If the pattern resumes within the next 3 available steps, treat it as one pattern with a short interruption, not a recovery. If fewer than 3 subsequent steps are available because the episode ends, follow rule 2 and document the limited evidence.
4. In a `failure` episode, an earlier pattern that was recovered stays `suspicious` and `recovered` is true. A later unrecovered pattern is the failure pattern.
5. A recovered pattern in a `success` episode is a useful example of a possible false alarm for the monitor. Annotate it carefully.
6. Write only what happened. Do not annotate why the agent recovered or what could have been done.

## G. Uncertain cases

Use `uncertain` when you honestly cannot choose a label from the logged evidence. Always give an `uncertainty_reason`:

| Code | Situation |
|---|---|
| `ambiguous_action` | The action could be reasonable or not (for example `look` to re-check a room). |
| `truncated_observation` | The logged observation is cut off or too short to judge. |
| `insufficient_evidence` | A pattern might be starting, but the evidence so far is too thin. |
| `missing_context` | Needed information is not available (for example the annotator view does not show `task_input`). |
| `log_gap` | Steps appear to be missing or out of order (step numbers are not 1, 2, 3, ...). |

Guidance:

- Prefer `uncertain` over a guess, but do not use it to skip hard steps. Add a note saying what would settle the question.
- An `uncertain` step cannot be an FDFP. If the whole pattern is uncertain, record no FDFP and explain in the notes.
- Do not use hidden or imagined agent reasoning to settle uncertainty, and do not use `thought` unless the team has decided it is allowed (section L).
- Uncertain steps go to the review step (section J) first.
- If the episode is `inconclusive`, do not label any step `failure_related`.

## H. Illustrative examples

Task for all examples: "put a mug in the coffee machine". All text is invented. Unless stated, `parse_ok` is true.

### H.1 Normal behaviour (raw outcome `success`, episode label `success`)

| Step | Action | Observation | Step label |
|---|---|---|---|
| 1 | go to countertop 1 | You arrive at countertop 1. On the countertop 1, you see a mug 1 and a knife 1. | normal |
| 2 | take mug 1 from countertop 1 | You pick up the mug 1 from the countertop 1. | normal |
| 3 | go to coffeemachine 1 | You arrive at coffeemachine 1. On the coffeemachine 1, you see nothing. | normal |
| 4 | put mug 1 in/on coffeemachine 1 | You put the mug 1 in/on the coffeemachine 1. | normal |

### H.2 Suspicious behaviour, not a failure (raw outcome `success`, episode label `success`)

| Step | Action | Observation | Step label | Category |
|---|---|---|---|---|
| 1 | go to countertop 1 | You arrive at countertop 1. On the countertop 1, you see a mug 1. | normal | |
| 2 | put mug 1 in/on coffeemachine 1 | Nothing happens. | suspicious | TX-04 |
| 3 | take mug 1 from countertop 1 | You pick up the mug 1 from the countertop 1. | normal | |
| 4 | go to coffeemachine 1 | You arrive at coffeemachine 1. On the coffeemachine 1, you see nothing. | normal | |
| 5 | put mug 1 in/on coffeemachine 1 | You put the mug 1 in/on the coffeemachine 1. | normal | |

Why: The task context shows that the mug is on the countertop and has not been picked up. The agent attempts to put it into the coffee machine anyway, and the environment reports no effect. In this illustrative example, the task context and observation together support treating this as a likely invalid action (TX-04). It is a single occurrence, and the agent then takes the mug and proceeds sensibly. Therefore, it remains `suspicious`, not `failure_related`. In a real trajectory, if the context does not clearly establish that the action was invalid, annotate it as `uncertain` instead.

### H.3 Failure-related behaviour (raw outcome `max_steps`, episode label `failure`, termination_reason `step_limit`)

| Step | Action | Observation | Step label | Category |
|---|---|---|---|---|
| 4 | go to shelf 1 | You arrive at shelf 1. On the shelf 1, you see nothing. | normal | |
| 5 | take mug 1 from shelf 1 | Nothing happens. | suspicious | TX-03 |
| 6 | take mug 1 from shelf 1 | Nothing happens. | suspicious | TX-03 |
| 7 | take mug 1 from shelf 1 | Nothing happens. | failure_related | TX-03 |
| 8 | take mug 1 from shelf 1 | Nothing happens. | failure_related | TX-03 |
| ... | same action repeated until the step limit | Nothing happens. | failure_related | TX-03 |

Why: Step 4 shows that the shelf is empty. The agent then repeatedly attempts to take a mug from that shelf, and the environment reports no effect. The task context and observations together support TX-03 (repeated unsuccessful action), with TX-01 as a secondary category because the same action is repeated. TX-04 may also be recorded as a secondary category if the annotator can clearly establish from the logged context that the action is impossible. Step 5 is the first unsuccessful attempt and is `suspicious`. Step 6 meets the minimum evidence floor, but the reviewer does not yet consider the pattern established. At step 7, the reviewer judges that the repeated unsuccessful action has become a clear failure pattern, so step 7 is the FDFP. The third attempt is a judgement in this example, not a fixed detector threshold. The episode ends at the step limit without success.

### H.4 Suspicious behaviour followed by recovery (raw outcome `success`, episode label `success`)

| Step | Action | Observation | Step label | Category |
|---|---|---|---|---|
| 3 | go to cabinet 1 | You arrive at cabinet 1. The cabinet 1 is closed. | normal | |
| 4 | look | You are in the middle of a room. Looking quickly around you, you see a cabinet 1 and a countertop 1. | normal | |
| 5 | look | You are in the middle of a room. Looking quickly around you, you see a cabinet 1 and a countertop 1. | suspicious | TX-01 |
| 6 | look | You are in the middle of a room. Looking quickly around you, you see a cabinet 1 and a countertop 1. | suspicious | TX-01 |
| 7 | go to countertop 1 | You arrive at countertop 1. On the countertop 1, you see a mug 1. | normal | |
| 8 | take mug 1 from countertop 1 | You pick up the mug 1 from the countertop 1. | normal | |

Why: a single `look` (step 4) is normal. Steps 5 and 6 repeat it with the same observation (TX-01, secondary TX-05), so they are `suspicious` and `first_suspicious_step` is 5. The agent then does something different that gives new information, so step 7 is the break step and the recovery step, and the pattern does not resume in steps 8 to 10 (or before the episode ends). The episode succeeds, so the steps stay `suspicious` and are not `failure_related`, and the pattern is marked `recovered`. This kind of case tells us where the monitor might raise a false alarm.

### H.5 Parse failures (raw outcome `success`, episode label `success`)

| Step | `parse_ok` | Action (as logged) | Observation | Step label | Category |
|---|---|---|---|---|---|
| 3 | true | go to cabinet 1 | You arrive at cabinet 1. The cabinet 1 is closed. | normal | |
| 4 | **false** | look | You are in the middle of a room. Looking quickly around you, you see a cabinet 1 and a countertop 1. | suspicious | TX-99 (`parse_failure`) |
| 5 | **false** | look | You are in the middle of a room. Looking quickly around you, you see a cabinet 1 and a countertop 1. | suspicious | TX-99 (`parse_failure`) |
| 6 | true | open cabinet 1 | You open the cabinet 1. In it, you see a mug 1. | normal | |

Why: steps 4 and 5 show `look` twice, but the agent never chose `look`: the model replies had no valid `Action:` and the logger substituted it. This is not TX-01. It is recorded as TX-99 `parse_failure` so we can count how often it happens. It is different from H.2 step 2, where the agent sent a well-formed command that the environment rejected (TX-04).

## I. Annotation format

Annotations are stored separately from the raw logs (schema rule 6: ground-truth labels are not in the raw files and live in `data/annotations/`). Format: JSONL, one JSON object per line, no line breaks inside an object. Two record types are used. They link to the raw log through `episode_id` and `step`. Use the `step` value exactly as written in the log (1-indexed).

Episode record fields:

| Field | Meaning |
|---|---|
| `record_type` | `episode_annotation` |
| `guideline_version` | `"1.1"` for this guideline |
| `episode_id` | As in the log |
| `annotator` | Annotator ID such as `A1` |
| `raw_outcome` | Copied from the summary: `success`, `failure`, `max_steps` or `error` |
| `episode_label` | `success`, `failure` or `inconclusive`, from the mapping in section B |
| `termination_reason` | `task_completed`, `environment_end`, `step_limit` or `infrastructure_error`, from the mapping in section B |
| `all_steps_reviewed` | true if unlisted steps may be read as `normal` |
| `patterns` | List of pattern records (below). Empty list if none. |
| `notes` | Free text, may be empty |

Pattern record fields: `pattern_id`, `category_id`, `secondary_category_ids`, `first_suspicious_step`, `first_defensible_failure_point` (`null` if none), `last_step`, `recovered`, `recovery_step` (`null` if not recovered).

Step record fields: `record_type` (`step_annotation`), `episode_id`, `step`, `annotator`, `step_label`, `category_id`, `pattern_id`, `uncertainty_reason`, `note`.

Example (illustrative, matches example H.3 with the episode shortened to 10 steps):

```jsonl
{"record_type": "episode_annotation", "guideline_version": "1.1", "episode_id": "ep_0007", "annotator": "A1", "raw_outcome": "max_steps", "episode_label": "failure", "termination_reason": "step_limit", "all_steps_reviewed": true, "patterns": [{"pattern_id": "p1", "category_id": "TX-03", "secondary_category_ids": ["TX-01", "TX-04"], "first_suspicious_step": 5, "first_defensible_failure_point": 7, "last_step": 10, "recovered": false, "recovery_step": null}], "notes": "Same failed take action repeated to the end of the episode."}
{"record_type": "step_annotation", "episode_id": "ep_0007", "step": 5, "annotator": "A1", "step_label": "suspicious", "category_id": "TX-03", "pattern_id": "p1", "uncertainty_reason": null, "note": "First failed take."}
{"record_type": "step_annotation", "episode_id": "ep_0007", "step": 6, "annotator": "A1", "step_label": "suspicious", "category_id": "TX-03", "pattern_id": "p1", "uncertainty_reason": null, "note": "Second attempt, not yet established."}
{"record_type": "step_annotation", "episode_id": "ep_0007", "step": 7, "annotator": "A1", "step_label": "failure_related", "category_id": "TX-03", "pattern_id": "p1", "uncertainty_reason": null, "note": "Third unsuccessful attempt, first defensible failure point."}
{"record_type": "step_annotation", "episode_id": "ep_0007", "step": 8, "annotator": "A1", "step_label": "failure_related", "category_id": "TX-03", "pattern_id": "p1", "uncertainty_reason": null, "note": ""}
{"record_type": "step_annotation", "episode_id": "ep_0007", "step": 9, "annotator": "A1", "step_label": "failure_related", "category_id": "TX-03", "pattern_id": "p1", "uncertainty_reason": null, "note": ""}
{"record_type": "step_annotation", "episode_id": "ep_0007", "step": 10, "annotator": "A1", "step_label": "failure_related", "category_id": "TX-03", "pattern_id": "p1", "uncertainty_reason": null, "note": ""}
```

Proposed file naming (to be agreed by the team): `data/annotations/<annotator>/<episode_id>.jsonl` for each annotator, and `data/annotations/adjudicated/<episode_id>.jsonl` for the final reviewed version. Raw logs stay in `data/raw/<pilot|full>/<episode_id>.jsonl`.

## J. Annotation consistency

Process for two or more annotators:

1. Each annotator labels the same episodes independently, before any discussion.
2. Compare automatically: episode label, step label per step, main category, FDFP, and recovery status.
3. Record every difference as a disagreement record (below). Do not overwrite the original annotation files.
4. Hold a review: the annotators go through each disagreement together using this guideline and decide. If they still disagree, ask a third team member or the supervisor.
5. Save the agreed result in the adjudicated file, with `resolved_by` noted.
6. If a disagreement happened because the guideline was unclear, write down the change needed. Update the guideline and its version note before the next batch.
7. Do a small pilot round first (a few episodes) to test the guideline, the evidence conditions and the format.

Agreement to report after each batch: percentage agreement on episode labels (expected to be 100% because the label is derived from `outcome`) and on step labels. Cohen's kappa for step labels can be added if enough steps exist. For FDFP, report the exact-match rate and the rate within 1 and 2 steps. No target value is set yet because there is no data.

Disagreement record (illustrative):

```jsonl
{"record_type": "disagreement", "episode_id": "ep_0007", "step": 6, "field": "step_label", "annotator_a": "A1", "value_a": "suspicious", "annotator_b": "A2", "value_b": "failure_related", "disagreement_type": "evidence_bar_interpretation", "resolution": "suspicious", "resolved_by": "review", "guideline_change_needed": true, "note": "TX-03 condition should say what 'nothing changed' means for a second attempt."}
```

`field` is one of `episode_label`, `step_label`, `category_id`, `first_defensible_failure_point`, `recovered`, `recovery_step`. `disagreement_type` examples: `evidence_bar_interpretation`, `category_precedence`, `ambiguous_observation`, `pattern_boundary`, `other`.

## K. Schema v1.0 assumptions and how this guideline handles them

| ID | Schema v1.0 fact | Effect on annotation |
|---|---|---|
| S1 | Step lines have `episode_id`, `step`, `action`, `observation`, `parse_ok`, `retry_count`, `thought`, `timestamp`. | Annotations link by `episode_id` and `step`. |
| S2 | Steps are 1-indexed. | Annotators copy `step` as logged. Non-consecutive steps are `log_gap`. |
| S3 | `task_input` and `initial_observation` are only in the summary line. | Available when replaying a finished episode. A live monitor cannot rely on them (`docs/candidate_signals.md`, section 5). |
| S4 | `outcome` has four values; `error` is infrastructure, not a task result. | Mapped in section B. |
| S5 | No tool-call or tool-result fields. | Tool errors cannot be annotated. |
| S6 | No field says whether an action was valid. `parse_ok` is about the model reply, not about the environment. | Validity is judged from the observation text. Parse failures are handled by rule C.6. |
| S7 | Observation truncation is not mentioned in the schema. | If an observation looks cut off: `uncertain` with `truncated_observation`. Full observations are preferred. |
| S8 | `retry_count` is an API retry count. | Not used for annotation. |
| S9 | `thought` is logged and described as observable. Permission to use it is undecided. | Hidden from annotators (section L). |
| S10 | A file without a summary line is unfinished. | Not annotated. |
| S11 | Ground-truth labels live in `data/annotations/`. | Section I. |

## L. Decisions pending (team / supervisor)

1. **`thought`.** May annotators read it, and may monitors use it? Not decided. Default in this guideline: annotators do not read it. A reason to keep it hidden even if monitors may use it: reading the agent's stated plan can bias the judgement of what was observable behaviour. Thought repetition itself belongs to LivePlan and LANGUTORY, not to this guideline. The annotator view must hide `thought`, which needs support in the T12 loader.
2. **Parse failures.** Is TX-99 `parse_failure` the right place, or does the team want a new category?
3. **Counts versus conditions** for the evidence table, after the pilot (section E.2).
4. **Annotator view.** Confirmation that `load_prefix` with k = full length gives annotators steps and `task_input` without `outcome` and, if needed, without `thought`.
5. **File layout** under `data/annotations/` (section I).
