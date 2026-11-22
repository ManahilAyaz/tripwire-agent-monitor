# Candidate Observable Signals (T08)

| Item | Value |
|---|---|
| Project | Tripwire: Mid-Trajectory Failure Detection for Long-Horizon AI Agents |
| Task | T08, Week 1 (November 16-22, 2026) |
| Author | Aqsa Zaib |
| Status | Draft v1.1, updated for Trajectory Log Schema v1.0 and the GRAPHECTORY design decisions. Candidate list only. Not reviewed or approved by the team or the supervisor. |
| Related files | `docs/taxonomy.md` (T07), `docs/annotation_guideline.md` (T09), `configs/rules.yaml` and `SRS/Tripwire/monitors/rules.py` (T10) |

## 1. Purpose and scope

This document lists measurable signals that Tripwire could use to notice possible failure while an agent is still running. These are candidates. Only SIG-01 is implemented (T10). No threshold here has been validated on data. Every symbol such as K, K_o, K_u, C, W_s, W is a placeholder for a value that T10/T19 will choose from pilot data; the numbers shown in examples only illustrate the arithmetic.

Rules for every signal:

- It uses only data in the schema v1.0 log that a monitor is allowed to see: a prefix (`task_input`, `initial_observation`, steps 1..k) obtained from `load_prefix(path, k)` (T12). Never `outcome`, never steps after k.
- It does not use `thought` until the team/supervisor decide that monitors may (section 6).
- A signal firing means "suspicious pattern". It does not mean the episode has failed (see `docs/taxonomy.md`, section 2.1).
- Detection only. Nothing here corrects or steers the agent (FYP-II).

## 2. Notation and conventions

- Steps are numbered t = 1, ..., N in execution order (**1-indexed**, as in schema v1.0).
- a_t is the `action` at step t. o_t is the `observation` returned after a_t.
- norm(x) trims whitespace, collapses runs of whitespace to one space and lowercases. Same definition as `docs/taxonomy.md`, section 2.2, and `normalize_action()` in `rules.py`.
- An **agent step** is a step line with `parse_ok = true`. A step with `parse_ok = false` is a logger substitution (the logger sent `look`); the action was not chosen by the agent. By default, steps with `parse_ok = false` are excluded from all behavioural signals. They do not count as agent actions or attempts, and their observations are excluded from observation-based signals. They also break consecutive sequences: signals must not skip a parse-failure step and join the valid steps on either side. This convention requires team confirmation (open point 5).
- `retry_count` (API retries) is never used by any signal. One step line is one agent action.
- W is a window size (number of most recent agent steps).

## 3. Summary table

| ID | Name | Taxonomy | Input needed (schema v1.0) | Deterministic | Layer | Status |
|---|---|---|---|---|---|---|
| SIG-01 | Consecutive Action Run Length | TX-01 | `action`, `step`, `parse_ok` | Yes | 1 | Implemented (T10), threshold provisional |
| SIG-02 | Cyclic Action Pattern | TX-02 | GRAPHECTORY `cycle_length` per step (not in the raw JSONL) | Depends on GRAPHECTORY | 1 | **Conditional** on the confirmed GRAPHECTORY feature contract |
| SIG-03 | Repeated Observation Run Length | TX-05 | `observation`, `parse_ok` | Yes | 1 | Candidate |
| SIG-04 | No-Effect / Error Observation Indicator | TX-03, TX-04 (supporting evidence only) | `observation`, `parse_ok` | Mostly (needs message list and contextual review) | 1 | Candidate |
| SIG-05 | Repeated Unsuccessful Action Count | TX-03 | `action`, `observation`, `parse_ok` | Mostly | 1 | Candidate |
| SIG-06 | Action Diversity in a Window | TX-01, TX-02, TX-06 | `action`, `parse_ok` | Yes | 1 | Candidate |
| SIG-07 | Observation Novelty / Stagnation Length | TX-06 | `observation`, `initial_observation`, `parse_ok` | Partly (proxy only) | 1 proxy, 2 | Candidate |
| SIG-08 | Goal Relevance Check | TX-07 | `task_input`, recent `action` and `observation` | No | 2 | Candidate |
| SIG-09 | Tool-Call Failure Rate | TX-04 | tool name, arguments, result or error | Yes, if fields existed | 1 | **Unavailable**: no tool fields in schema v1.0 |
| SIG-10 | Ignored-Observation Check | TX-08 | `task_input`, `observation`, later `action`s | No | 2 | Candidate |
| SIG-11 | Step Budget Consumption | TX-06 (context) | `step` and a step limit | Yes | 1 (context only) | Candidate, **step limit not available in the prefix** |

Manual annotation (T09) is not a signal, but it is the reference against which these signals are evaluated.

### 3.1 Who computes what (responsibility split)

GRAPHECTORY supplies features (facts). The rule monitor decides when a signal reaches its alert threshold. Rule thresholds belong to my T10/T19 work and live in `configs/rules.yaml`.

| Signal | Feature (the fact) comes from | Alert threshold decided by |
|---|---|---|
| SIG-01 | Rule monitor itself (T10, from `action`, `step`, `parse_ok`) | Rule monitor (T10/T19). Provisional 3. |
| SIG-02 | GRAPHECTORY `cycle_length` per step | Rule monitor (T10/T19). **No value proposed.** |
| SIG-03 to SIG-07, SIG-11 | Not decided. To be confirmed with Manahil whether GRAPHECTORY already exposes an equivalent feature. If it does, the rule monitor must not recompute it. | Rule monitor (T10/T19) |
| SIG-08, SIG-10 | Layer 2 semantic monitor | Layer 2 / combination design (not designed in Week 1) |
| SIG-09 | Not available | n/a |

### 3.2 Not a signal here: repeated thoughts

Repeated reasoning is outside this list. In Manahil's GRAPHECTORY design, thoughts are stored on graph nodes (not as separate nodes), LivePlan adds thought nodes to detect repeated reasoning, and LANGUTORY (T06) covers text analysis. A thought-repetition signal here would duplicate that work. SIG-08 and SIG-10 are about the task and the environment, not about the agent's own wording.

## 4. Signal definitions

### SIG-01 Consecutive Action Run Length

| Field | Content |
|---|---|
| What it measures | How many times the same action is issued by the agent in a row. |
| Required input | Ordered agent steps: `action`, `step`, `parse_ok`. |
| Calculation | R_t = 1 for the first agent step. For the next agent step with step number t: R_t = R_(t-1) + 1 if norm(a_t) = norm(a_(t-1)) **and** the step numbers are consecutive, else R_t = 1. Signal value = max over t of R_t. An alert is raised at the first step where R_t >= K (total occurrences, first one included). Provisional K = 3 (`configs/rules.yaml`). |
| Example | Steps 1 to 5: go to desk 1, look, look, look, take pen 1 from desk 1. R = 1, 1, 2, 3, 1. Max = 3. At K = 3 the alert is raised at step 4. At K = 4 nothing is flagged. |
| Suspicious pattern | Long runs, especially when the observation does not change or says nothing happened. |
| Deterministic | Yes. |
| Limitations and false positives | A run of 2 is common and harmless. It cannot see loops with more than one action (SIG-02). A step that cannot be used (missing action, `parse_ok` not true, bad step number) ends the run, so a gap in the log can hide a real run; skipped records are listed in the result. A gap in the step numbers also ends a run. API retries cannot create false runs (one record per step; retries are only in `retry_count`), but logger-substituted `look` steps could, which is why `parse_ok = false` steps are excluded. |
| Layer | Layer 1. Implemented in `SRS/Tripwire/monitors/rules.py`, configured in `configs/rules.yaml`. |

### SIG-02 Cyclic Action Pattern (conditional on GRAPHECTORY)

| Field | Content |
|---|---|
| What it measures | Whether the agent's recent steps form a repeating cycle, as reported by GRAPHECTORY through its per-step `cycle_length` value. |
| Required input | `cycle_length` for step t, supplied by GRAPHECTORY. It is **not** in the raw JSONL log and is **not** added to schema v1.0. |
| Calculation | None defined in this document. GRAPHECTORY does not do full graph-wide cycle detection and this project does not duplicate it. Once the feature contract is confirmed, the rule monitor compares the supplied value with an alert threshold. No threshold is proposed here. |
| Example | Cannot be written yet, because the format of `cycle_length` is not confirmed. (v1 had a self-made "block of p actions repeated C times" formula. It is removed because it would duplicate GRAPHECTORY.) |
| Suspicious pattern | Going round in a circle without new information. |
| Deterministic | Depends on GRAPHECTORY. |
| Limitations and false positives | Unknown until the contract is known. Known risk from the taxonomy: legitimate alternation exists (open, take, close for several items), and "go to cabinet 1 / go to cabinet 2" are different actions. |
| Layer | Layer 1, conditional. |

Questions to settle with Manahil before SIG-02 can be built (nothing below is assumed):

1. What is the type and range of `cycle_length`, and what value means "no cycle"?
2. What does the length count: actions, graph nodes, or something else?
3. Is it computed from steps 1..t only (prefix-safe), so it contains no future information?
4. How are `parse_ok = false` steps treated when it is computed?
5. How is it delivered to the rule monitor (function call, object, file)? It is not in the raw JSONL.

### SIG-03 Repeated Observation Run Length

| Field | Content |
|---|---|
| What it measures | How many consecutive steps return the same observation. |
| Required input | Ordered observations (full text, not truncated). |
| Calculation | Same as SIG-01 applied to normalized observations, using only consecutive steps with `parse_ok = true`. The signal value is the maximum consecutive observation run length in the available prefix. An alert may be considered when the run reaches a provisional threshold K_o, which has not yet been selected or validated. |
| Example | Observations: obs1, obs1, obs1, obs1, obs2 give a maximum consecutive run of 4. With an illustrative threshold K_o = 4, the signal reaches that threshold at the fourth observation. A-B-A-B alternation is not detected by this consecutive-run signal and must be treated separately from TX-05. |
| Suspicious pattern | Agent acts but the environment shows nothing new. |
| Deterministic | Yes. |
| Limitations and false positives | Actions such as `inventory` or `look` repeat on purpose. Truncated observations break exact matching. Tiny differences in the text hide repeats. |
| Layer | Layer 1. |

### SIG-04 No-Effect / Error Observation Indicator

| Field | Content |
|---|---|
| What it measures | Whether an agent step produced an observation matching a known no-effect or error-message pattern. This is supporting evidence only and does not independently establish that the action was invalid. |
| Required input | Observation text of an agent step. Schema v1.0 has no error or validity field. |
| Calculation | I_t = 1 if `parse_ok = true` and the normalized observation matches an entry in the provisional no-effect/error message list; otherwise I_t = 0. Derived values may include the consecutive count of I_t = 1 and the proportion of matching observations in the last W eligible agent steps. The message list and its interpretation must be checked against real ALFWorld trajectories. |
| Example | Observations: ok, "Nothing happens.", "Nothing happens.", ok. I = 0, 1, 1, 0. For W = 4 the rate is 0.5. |
| Suspicious pattern | Several no-effect steps close together. A single one is weak evidence. |
| Deterministic | Mostly. It depends on a message list that must be checked against real ALFWorld observations. |
| Limitations and false positives | One invalid action is normal. "Nothing happens." can also follow a valid but useless action, so it does not prove the action was invalid. `parse_ok = false` is a different thing and is not an I_t = 1 event. A message list built from memory may miss other texts. |
| Layer | Layer 1. |

### SIG-05 Repeated Unsuccessful Action Count

| Field | Content |
|---|---|
| What it measures | How often the same action is retried after it had no effect. |
| Required input | `action` and `observation` of agent steps. |
| Calculation | For each distinct norm(a): U(a) = number of agent steps t with norm(a_t) = a and I_t = 1. Signal value = max over a of U(a). Flag when U >= K_u (provisional, not chosen yet). Attempts do not need to be neighbours. |
| Example | `take mug 1 from countertop 1` fails at steps 12, 14, 16 (look at 13 and 15). U = 3. |
| Suspicious pattern | The agent does not learn from the failure. |
| Deterministic | Mostly (same dependency as SIG-04). |
| Limitations and false positives | A retry can be sensible after the situation changes (for example, the agent opens a container before trying to take an object again). This count does not check what changed between attempts, so it is only a candidate indicator of repeated unsuccessful behaviour. It must not independently establish TX-03 or trigger a definitive failure judgement without contextual review. Judging whether the situation changed requires additional logic, Layer 2, or manual annotation. |
| Layer | Layer 1, with Layer 2 or manual confirmation for the "nothing changed" part. |

### SIG-06 Action Diversity in a Window

| Field | Content |
|---|---|
| What it measures | How varied the recent actions are. |
| Required input | `action` of agent steps. |
| Calculation | D_W = unique(norm(a) over the last W agent steps) / W. Low D_W means little variety. |
| Example | W = 6, actions A, A, B, A, A, B: unique = 2, D_6 = 2/6 = 0.33. For six different actions D_6 = 1.0. |
| Suspicious pattern | Low diversity together with no progress. |
| Deterministic | Yes. |
| Limitations and false positives | Short windows are noisy. Some tasks naturally reuse actions. Low diversity alone says nothing about progress. Good as a supporting signal, weak as a standalone alarm. |
| Layer | Layer 1 (supporting signal). |

### SIG-07 Observation Novelty / Stagnation Length

| Field | Content |
|---|---|
| What it measures | How long it has been since the agent last saw something new. |
| Required input | `observation` of agent steps, and `initial_observation` as the first thing already seen. |
| Calculation | An observation is novel if `norm(o_t)` has not appeared earlier in the available prefix. The `initial_observation` counts as already seen. Define S_t = 0 when the current observation is novel; otherwise S_t = S_(t-1) + 1. A step with `parse_ok = false` does not update the novelty calculation. Flag when S_t >= W_s (provisional, not chosen yet). This is an observation-novelty proxy, not a direct measurement of task progress. |
| Example | Novel observations at steps 3 and 8, none through step 18, so S_18 = 10. |
| Suspicious pattern | The agent is moving around without discovering anything. |
| Deterministic | Only as a rough proxy. True progress needs meaning. |
| Limitations and false positives | Slight text changes make an old observation look novel. Real work such as carrying an object to a goal may produce no novel text. Early exploration can look stagnant. |
| Layer | Layer 1 as a proxy. Layer 2 or manual for real progress. |

### SIG-08 Goal Relevance Check

| Field | Content |
|---|---|
| What it measures | Whether the recent actions still serve the task input. |
| Required input | `task_input`, recent `action`s and `observation`s. |
| Calculation | Not a formula. The semantic monitor returns a label such as aligned / unclear / off-task for a window of recent steps, with a short reason. (The proposal names Gemini for this.) |
| Example | Task "put a mug in the coffee machine", recent actions all about knives and forks: off-task. |
| Suspicious pattern | Off-task or unclear for several windows in a row. |
| Deterministic | No. |
| Limitations and false positives | LLM judgements vary and can be wrong. Sub-goals look off-task. It costs API calls, so it should run selectively, for example after a Layer 1 signal fires. `task_input` is only in the summary line (see section 5). |
| Layer | Layer 2. |

### SIG-09 Tool-Call Failure Rate (unavailable)

| Field | Content |
|---|---|
| What it measures | How often tool calls fail, and whether the same failing call repeats. |
| Required input | Tool name, tool arguments, tool result or error status per step. **None of these exist in schema v1.0.** |
| Calculation | F_W = failed calls in the last W steps / tool calls in the last W steps. Also count identical failing calls. Cannot be computed now. |
| Example | 4 tool calls in a window, 3 return errors, F = 0.75. |
| Suspicious pattern | High failure rate, or the same failing call repeated. |
| Deterministic | Yes, once the fields exist. |
| Limitations and false positives | ALFWorld actions are text commands, not tool calls, so this signal stays unusable unless tool fields are added to the schema by team agreement (schema change = PR agreed by all three, with a `schema_version` bump). `parse_ok` is not a tool-error flag. |
| Layer | Layer 1, unavailable. |

### SIG-10 Ignored-Observation Check

| Field | Content |
|---|---|
| What it measures | Whether the agent ignored useful information from an observation. |
| Required input | `task_input`, an observation, and the next several actions. |
| Calculation | Semantic question to the monitor: "Does this observation contain information relevant to the task, and do the subsequent actions available in the current prefix use that information?" Output: yes / no / unclear. The monitor must use only actions already present in steps 1..k. If the required observation window has not yet become available, the signal must return pending or not evaluated rather than using future steps. |
| Example | Observation shows "a mug 1" on a countertop for a mug task and the next five actions never take it. |
| Suspicious pattern | Needed information is present and not used for several steps. |
| Deterministic | No. |
| Limitations and false positives | The visible object may not match the task (a different object). Long or truncated observations. LLM error. Same `task_input` availability issue as SIG-08. |
| Layer | Layer 2. |

### SIG-11 Step Budget Consumption

| Field | Content |
|---|---|
| What it measures | How much of the allowed step limit has been used. |
| Required input | CCurrent `step` and the step limit. In schema v1.0, `max_steps` is stored only in the episode summary, which is written after the episode finishes. It is therefore unavailable to a live monitor under the current prefix contract. SIG-11 cannot be used for live monitoring unless the step limit is provided separately through an agreed configuration or the T12 prefix contract is updated. |
| Calculation | B_t = t / max_steps. With 1-indexed steps, B_t is in (0, 1]. |
| Example | Step 30 with a limit of 50 gives B = 0.6. |
| Suspicious pattern | Not suspicious alone. It makes other signals more worrying as B grows. |
| Deterministic | Yes. |
| Limitations and false positives | It says nothing about behaviour. Different tasks need different numbers of steps. Context only. |
| Layer | Layer 1 (context for combining signals). |

## 5. Data needed versus Schema v1.0

| Data | In schema v1.0? | Used by | Note |
|---|---|---|---|
| `step` (1-indexed int) | Yes, step line | all | Numbering base is fixed: 1. Runs require consecutive numbers. |
| `action` | Yes, step line | SIG-01, 05, 06 | The command actually sent to the environment. |
| `observation` | Yes, step line | SIG-03, 04, 05, 07, 08, 10 | Signals that compare text need the full observation. Whether it is ever truncated in the logs is not stated in the schema (open point 4). |
| `parse_ok` | Yes, step line | all action-based signals | `false` means the logger substituted `look`. Not a validity flag for the environment. |
| `retry_count` | Yes, step line | none | API retries (T13). Not an agent action. Never mixed into SIG-01. |
| `timestamp` | Yes, step line | none | Not needed by any signal listed here. |
| `thought` | Yes, step line | none | **Pending team/supervisor decision** (section 6). |
| `task_input` | Yes, **summary line only** | SIG-08, 10 | Available through `load_prefix(path, k)` when replaying a finished episode. The summary is written only when the episode finishes (schema rule 3), so a monitor reading a still-growing file would not find it. Question for the T12 owners (open point 3). |
| `initial_observation` | Yes, summary line only | SIG-07 | Same availability question as `task_input`. |
| `max_steps` | Yes, summary line only | SIG-11 | Not available to a live monitor under the current prefix contract. Requires an agreed external configuration or an updated T12 contract. |
| `outcome` | Yes, summary line | never by monitors | Evaluation code only. |
| Tool name, arguments, result or error | **No** | SIG-09 | Unavailable. |
| Validity / no-effect flag from the environment | **No** | SIG-04, 05 | Use observation text (message list). |
| GRAPHECTORY `cycle_length` | **No** (not in raw JSONL, by design) | SIG-02 | Format not confirmed. Not to be added to the raw schema. |

## 6. Use of `thought` (decision needed)

Schema v1.0 logs `thought` and calls it observable. The team has **not** decided whether monitors and annotators may use it, and private chain-of-thought must not be assumed allowed. Until the team/supervisor decide:

- no signal in this document reads `thought`;
- the rule monitor (`rules.py`) ignores it;
- the annotation guideline keeps it hidden from annotators.

If it is later allowed, thought analysis stays with LivePlan (thought nodes) and LANGUTORY (T06), not here (section 3.2).

## 7. Combining signals (not designed yet)

How Layer 1 outputs trigger Layer 2, and how they become a risk assessment, is not decided in FYP-I Week 1. A sensible starting point to discuss is to let Layer 1 signals decide when the semantic monitor is called. This needs team agreement and data.

## 8. Open points for team review

1. All thresholds (K, K_o, K_u, W_s, W, and any SIG-02 threshold) need pilot data and belong to T10/T19. Nothing is validated.
2. The no-effect message list for SIG-04 and SIG-05 has to be confirmed on real ALFWorld observations.
3. How `task_input`, `initial_observation` and `max_steps` reach a monitor, since they sit in the summary line (T12 loader owners).
4. Whether observation text can be truncated in the log, and whether tool fields will ever be added.
5. Convention for `parse_ok = false` steps: exclude them from behavioural signals, as described above. The team must decide whether parse failures should be measured separately as a data-quality signal or retained as TX-99 annotations. They must not automatically be counted as behavioural failures.
6. GRAPHECTORY: the `cycle_length` contract (section SIG-02), and which of SIG-03 to SIG-07 and SIG-11 it already supplies.
7. Whether monitors and annotators may use `thought` (supervisor/team).
