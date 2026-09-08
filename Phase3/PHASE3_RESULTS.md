# Phase3 report — RAG abstention experiment

Original run and post-review ambiguity amendment completed locally on
7 September 2026 under the revised school-project plan.

## Summary

This experiment tested whether the selected Phase2 RAG system knows when not to
answer, and whether a structured abstention prompt or a reranker-score gate
improves that behavior. It used **200 cases, including all 44 authored questions**,
split into 140 development and 60 final cases.

A small post-review amendment clarified the linked wildfire control and two
negative questions. It retained 394 exact-request outcomes and generated six
new responses. Both policies now answer that control correctly and abstain on
both clarified negatives. Original measurements remain below for transparency;
the [amended results](#post-review-ambiguity-amendment-results) are reported
separately. This is not a new independent held-out test.

**B0, using the exact selected Phase2 p2 prompt, remains the selected policy.**
B1 reduced development negative-case errors but did not improve the final
comparison and lowered control token-F1. B2 changed no answer/abstention outcomes
relative to B1. These findings support retaining B0 for this experiment; they
do not show that structured prompts or score gates are universally worse.

There are **400 B0/B1 outcomes: 396 successful responses and four generation
failures**. Failures are retained in the evaluation, not treated as correct
abstentions. No RAGAS or external judge was used. Case evidence was AI-reviewed;
the project owner authorized finalization, not independent per-case human
certification. The limitations and original-plan simplifications below are part
of the conclusion.

## Research questions and conclusions

The table summarizes the original run. The amendment improves the two wildfire
negative cases but leaves the selected policy and all final measurements unchanged.

| Original research question | Evidence from this experiment | Conclusion |
|---|---|---|
| RQ1: How well does the Phase2 baseline abstain when evidence is insufficient? | B0 correctly abstained on 73/79 development negatives and 32/34 final negatives, with no false abstentions on controls. Remaining negative errors include generation failures. | The baseline usually followed the abstention contract on this challenge set, but was not error-free. This is not a population-level safety estimate. |
| RQ2: Does an output-contract prompt reduce unsupported answers? | B1 reduced development negative errors from 6/79 to 3/79, but final errors were 3/34 versus B0's 2/34. Its development control-F1 drop exceeded the selection allowance. | No consistent improvement was demonstrated. This measures whole-prompt abstention behavior, not an isolated JSON-format effect or a fact-checked hallucination rate. |
| RQ3: Does a reranker-score gate improve behavior without excessive false abstention? | The development-selected B2 gate rejected no development cases and one final case that B1 had already abstained on. It caused no control abstentions and changed no answer/abstention outcomes. | No additional benefit was demonstrated at the locked threshold. This is not evidence that all possible gates are ineffective. |

## Experimental design

### Policies and shared configuration

| Policy | What changes |
|---|---|
| B0 — Phase2 baseline | Exact p2 prompt from the supplied registry; a short cited answer or the canonical abstention sentence. No rejection gate. |
| B1 — structured abstention | Same question and numbered contexts as B0; the original Phase3 prompt requests JSON with an answer and citations, or `insufficient_evidence` with a null answer and empty citations. It explicitly forbids repairing false premises. |
| B2 — B1 plus score gate | Reuses B1 predictions. A successful response is changed to abstention when an eligible case's top reranker score is below the development-selected threshold. Failures remain failures. No additional generation calls. |

The selected components are BGE-M3 sparse retrieval, BGE-reranker-large and
`gemini-3.1-flash-lite`, with temperature 0, a 512-token output limit and minimal
reasoning. Retrieval uses depth 5; controlled-context ablation/weak-evidence
cases use their reviewed remaining passages. B0/B1 receive identical contexts
for each case. Local CUDA settings are fixed across policies.

This is a fresh local Phase3 comparison, not a bit-identical reproduction of
historical Phase2 scores. It uses the supplied new review bundle and corpus,
not the old resolved testset as a correctness gate. The active corpus contains
19,259 of the 19,263 uploaded chunks: four external replacement source articles
are withheld globally. Removed-article cases separately exclude their source
article for that case. See the [local guide](LOCAL_RUN.md) for exact setup and
the [prompt registry](phase2_generation_prompts.yaml) for p2.

### Dataset, partitions and evaluation scope

There are 87 answerable controls and 113 negatives across seven case types.
Development contains 61 controls and 79 negatives; final contains 26 controls
and 34 negatives. The subtype table below gives the complete 140/60 allocation.
The 44 authored questions are **included in the 200**, not an extra test set.

Source-related variants remain in the same partition. The dataset has 82
distinct internal source articles and 22 distinct external source articles.
All 87 controls have their source gold in the top three retrieved passages.
The original 80-control/10-natural-miss target became 87/3 after checking actual
retrieval; incorrect negative labels were not retained to force the quota.

There are 153 `full_corpus` cases eligible for B2 when actual reranker scores
are available, and 47 `provided_context` cases: 22 ablations, 22 weak-evidence
cases and three natural misses. The natural-miss contexts were actually
retrieved, not manually ablated, but retain their original provided-context
measurement scope and are not gated. Separate scope-level aggregate scores
are not reported.

### Development selection and final evaluation

The seven-case smoke run was reused in development. Threshold calibration and
policy selection used development only; the selected decision was locked before
final generation. Final reports all policies without reselecting or tuning them.

Development eligibility required generation success at least 98%, false
abstention at most 10%, control token-F1 no more than 0.02 below B0 and
citation-index validity no more than 0.01 below B0. Eligible policies were
compared by negative-case error rate, preferring the simpler policy within the
plan's 0.02 tolerance.

**Only B0 satisfied all original development guardrails.** B1/B2 control token-F1 was
0.3434 versus B0's 0.4487, a drop of about 0.1053. Their generation-success,
false-abstention and citation-index checks passed; the token-F1 guardrail
excluded them. This is the saved decision, not a choice made from final results.
See the [locked development decision](runs/development/decision.json).

B2's threshold was -7.839843750000001 on raw reranker scores. It changed no
answer/abstention outcome. The threshold is not a calibrated probability or a
recommendation for another dataset.

## Original quantitative results

The tables below use the saved
[development results](runs/development/results.json) and
[final results](runs/final/results.json). Values are rounded for display;
full precision remains in those files. Development and final are kept separate
because development was used for selection.

### Metric definitions

- **Negative errors:** a non-abstaining response or a generation failure on a
  negative case, divided by all negatives. This is the saved
  `false_answer_rate`, not a fact-checked hallucination rate.
- **Abstention F1:** insufficient evidence is the positive class, with
  `2TP / (2TP + FP + FN)`. A successful abstention on a negative is TP;
  a negative non-abstention or failure is FN. Under the runner's conservative
  failure convention, a control abstention or failure is FP. There were no
  control failures in this run.
- **Control exact match (EM) and token-F1:** answer-text similarity to the best
  accepted reference after citation extraction, lowercasing, punctuation removal
  and whitespace normalization. Token-F1 averages per-control token-overlap F1.
  These are lexical metrics, not semantic correctness scores.
- **False abstentions:** refusals on answerable controls. The saved guardrail
  also counts control generation failures as errors; none occurred here.
- **Citation-index validity:** the mean valid-reference fraction on controls.
  It checks indices against the available contexts, not whether the citations
  support the answer.
- **Generation success:** a usable, parsed response, divided by all cases.
  It does not mean the answer is factually correct. Failures are also shown
  separately and are never removed from denominators.

### Abstention and generation

| Split / policy | Negative errors | Abstention F1 | Generation success | Generation failures |
|---|---:|---:|---:|---:|
| Development B0 | 6/79 (7.59%) | 0.9605 | 99.29% | 1/140 |
| Development B1 | 3/79 (3.80%) | 0.9806 | 99.29% | 1/140 |
| Development B2 | 3/79 (3.80%) | 0.9806 | 99.29% | 1/140 |
| Final B0 | 2/34 (5.88%) | 0.9697 | 98.33% | 1/60 |
| Final B1 | 3/34 (8.82%) | 0.9538 | 98.33% | 1/60 |
| Final B2 | 3/34 (8.82%) | 0.9538 | 98.33% | 1/60 |

### Answerable-control guardrails

| Split / policy | Control EM | Control token-F1 | False abstentions | Citation-index validity |
|---|---:|---:|---:|---:|
| Development B0 | 5/61 (8.20%) | 0.4487 | 0/61 (0%) | 100% |
| Development B1 | 0/61 (0.00%) | 0.3434 | 0/61 (0%) | 100% |
| Development B2 | 0/61 (0.00%) | 0.3434 | 0/61 (0%) | 100% |
| Final B0 | 3/26 (11.54%) | 0.4766 | 0/26 (0%) | 100% |
| Final B1 | 1/26 (3.85%) | 0.2996 | 0/26 (0%) | 100% |
| Final B2 | 1/26 (3.85%) | 0.2996 | 0/26 (0%) | 100% |

All policies answered all 87 controls and used valid citation indices on them.
That does not certify every answer or citation's evidential support. A separate
B0 negative answer used an invalid citation.

The low EM values, including B1's 0/61 development exact matches, do **not**
mean all those answers were factually wrong. A longer correct sentence can fail
exact match against a short reference. B1 has weaker brevity instructions than
p2, so answer length and valid paraphrases also affect the token-F1 comparison.
This is a whole-policy comparison, not an isolated test of JSON formatting.

### Per-type diagnostic breakdown

Each error entry is a count out of that split's case count. On negative types,
errors mean non-abstention or failure; on controls, they mean false abstention
or failure, **not incorrect answer content**. B1 and B2 have identical subtype
counts and therefore share a column.

| Case type | Dev cases | Dev B0 errors | Dev B1/B2 errors | Final cases | Final B0 errors | Final B1/B2 errors |
|---|---:|---:|---:|---:|---:|---:|
| Answerable control | 61 | 0 | 0 | 26 | 0 | 0 |
| Natural retrieval miss | 2 | 0 | 0 | 1 | 0 | 0 |
| Context ablation | 16 | 1 | 1 | 6 | 0 | 0 |
| Removed article | 16 | 2 | 0 | 6 | 0 | 0 |
| Weak evidence | 15 | 1 | 1 | 7 | 0 | 0 |
| Counterfactual | 15 | 2 | 1 | 7 | 1 | 2 |
| External-unanswerable | 15 | 0 | 0 | 7 | 1 | 1 |
| All cases | 140 | 6 | 3 | 60 | 2 | 3 |

Of these errors, each policy has one development counterfactual failure and one
final external-unanswerable failure. They are included above, not additional
errors. B1's development improvement over B0 is concentrated in the two
removed-article cases and one counterfactual decision; B1 does not resolve the
paired ablation/weak-evidence negative-case errors. In final, B1's extra error
is a useful false-premise correction under a strict abstention contract, as
explained below.

The three natural misses all elicited abstention, but three observations cannot
establish a reliable subtype success rate. Other subtype samples are also small.
These counts are diagnostic, not evidence of statistical significance.

## Behavioral findings: what went wrong and what the scores miss

In the original run, successful-response inspection covered both policies' answers for all 87
controls and every negative non-abstention. All 396 raw successful responses
also reparse consistently with their saved fields. Across B0/B1 there are 174
control answers, 212 negative abstentions, ten negative non-abstentions and four
failures. Most control answers preserve the expected fact; parsing success is
not certification of every claim or citation.

- **Related facts substituted for the missing fact.** For the paired San Diego
  wildfire ablation/weak-evidence questions, both policies answered with damage
  or shelter information instead of abstaining. The intended source finding
  was four bodies. Their broad "what was found" wording makes these less clean
  negative tests; interpret them conservatively, not as certified hallucinations.
  B0 also cited `[6]` when the weak case contained only one numbered passage.
- **Unsupported relationship inference.** B0 called Fiorentina the winner
  against Sporting Lisbon even though the cited passage was a future fixture
  list. It also attributed the requested orphan-help interview to the Malawi
  Nation newspaper; the context named an interview but did not establish that
  requested subject. B1 abstained on both removed-article cases.
- **False-premise handling is not the same as factual truth.** B0 supplied
  Atlanta's location despite the FAA question's false radar-outage premise.
  Both policies answered the final Times Square "parade" question from ball-drop
  evidence. B1 also corrected the prison-sentence question with the actual
  possible $627 fine. That correction is substantively useful, but violates this
  experiment's strict abstention contract. Its extra final error is therefore
  not evidence of an extra hallucinated fact.
- **Lexical scores penalize valid wording.** B0's "nineteen" received zero
  overlap against `19`. B1's "flipped and landed on its right side" received zero
  overlap against `rolled over`. Many B1 answers retained the expected fact but
  added words, lowering token-F1. Do not equate those decreases with factual loss.
- **Minor precision loss.** On the control asking how many people were expected
  at the Times Square ball drop, B0 used "at least one million" from a passage
  about actual attendance alongside the separate forecast of "about a million."
  The core quantity agrees, but the qualifier conflates attendance with the
  forecast. B1 retained the expected "about a million" wording.
- **Provider failures are not abstentions.** Two questions each exhausted
  three attempts under both policies, producing the four recorded failures.
  Unchanged B0 diagnostic replays later returned an explicit provider
  content-filter reason. This supports filtering as the explanation, but the
  original finish reasons were lost and B1 was not replayed. The exact
  triggering passage was not isolated. See Appendix B for the evidence and
  post-run logging fix; the original four outcomes were not replaced.

## Data review and limitations

### The 44 authored questions

All 44 were individually AI-reviewed against source evidence and actual
retrieved contexts. Twelve counterfactual and ten external slots were repaired
or replaced, with reasons retained in [review_decisions.json](review_decisions.json).
All 22 external questions now have full retained source passages: 18 recovered
from the pinned source dataset and four locally sourced replacements. The four
local source articles are globally withheld, leaving 19,259 active chunks.

For each policy, 21/22 external questions produced the intended abstention;
the remaining question was the no-content failure, not a generated answer.
For counterfactuals, each policy had 19 abstentions, two non-abstentions and
one failure, although the non-abstaining questions differed. These are small,
descriptive challenge sets, not population-level safety estimates.

The per-policy counts above combine development and final only to describe
these authored cases; they are not additional held-out evaluation results.

### Repairs and evidence quality

Before the separate ambiguity amendment, 79 slots were changed, including source-answer corrections,
leakage/ambiguity repairs and evidence-based reclassifications. Eight proposed
natural misses actually retrieved answers and became controls. The Barnett
control became a natural miss because the requested statement was absent from
the actual retrieved chunk. This last correction occurred during development,
after 153 successful responses but **before any final generation**. The prior
inputs and exact-request reuse audit remain in `runs/label_audit/`.

All seven original bundle files remain byte-identical to their backups, and
shared Phase2 code is unchanged. The saved raw responses reproduce their parsed
fields, per-case scores and aggregate measurements. Frozen context identities
and the retained request ledger establish artifact consistency, not independent
semantic certification.

### Limits on interpretation

- **Review independence:** each case has an AI evidence review. Project-owner
  approval authorizes use of the experiment, but does not establish independent
  per-case human inspection or a second review.
- **Dataset size and dependence:** the 200 cases are a deliberately constructed
  challenge set. Related variants share sources, and there are only three
  natural misses. No grouped confidence intervals or population-level claims
  are supplied.
- **Uneven negative-case difficulty:** some weak-evidence passages are short
  article endings rather than close-to-answer distractors. They test missing
  information but do not establish uniformly difficult partial-evidence tests.
  The short ICE-lawsuit query also measures query underspecification, not only
  retriever quality. These examples should not be generalized to every case of
  their subtype.
- **Ambiguity and scoring:** the original broad wildfire questions are imperfect negative
  tests. A helpful correction of a false premise can violate the strict
  abstention contract without inventing a fact.
- **Prompt and metric confounding:** B1 differs in instruction wording and
  brevity, not just output format. EM/token-F1 penalize extra words and some
  valid paraphrases; citation-index validity is not semantic citation evaluation.
- **Operational failures:** four benchmark outcomes remain failures. Later
  filtering diagnostics support a cause but do not recover every original
  provider response detail.

The original labels, prompts, thresholds and selection rules remain unchanged.
The separately identified amendment below corrects question ambiguity after
inspection of the original outputs; it is not an untouched test. Aligned
answer-length instructions or a different gate would be a separate experiment,
not a rewrite of the original findings.

## Post-review ambiguity amendment results

### Method and evidence

The broad wildfire question could reasonably elicit damage or shelter details.
Original queue rows 13, 63 and 87 now ask: "What was found in a canyon east of
San Diego in the path of the Harris Fire?" The same wording is used for the
answerable control, context ablation and weak-evidence case. It identifies the
event without supplying the answer. The existing counterfactual already uses
this wording with San Francisco substituted for San Diego.

All three changed cases are in development. Labels, reference answers, case
IDs, the 200-case allocation, all 44 authored questions, corpus, model and
prompts are unchanged. The two negatives retain their exact provided passages,
which lack the requested finding. Fresh selected retrieval for the control
places the answering source first; its caption and text identify four bodies
in a canyon east of San Diego. The five retrieved passages were AI-reviewed
before generation. No contexts were removed to avoid provider filtering.

All 400 old/new request payloads were compared: 394 were identical and their
outcomes were reused, including the four historical failures. The six changed
requests were generated with the current credential; all succeeded without
retries. B2 calibration and policy selection were recomputed using revised
development only, before producing the revised final comparison. Final B0/B1
responses were reused unchanged. See the [request-reuse audit](runs/revision/reuse_audit.json)
and [revised cases](runs/revision/reviewed/cases.jsonl).

### Changed-case behavior

| Case | Original behavior | Revised B0 | Revised B1 |
|---|---|---|---|
| Answerable control, row 13 | Both policies identified the four bodies. | Four bodies, citing the source. | Four people’s bodies, with source-supported detail and citation. |
| Context ablation, row 63 | Both supplied other wildfire-related information. | Abstains. | Abstains. |
| Weak evidence, row 87 | Both supplied crop-damage information; B0 also used an invalid citation. | Abstains. | Abstains. |

The clearer negatives no longer invite those alternate readings. This supports
the value of precise case construction; it is not evidence that either model
policy was improved. The control also received fresh retrieval, so its new
wording and context changes are not isolated causal effects. Each changed
request has one successful generation, not a repeated-trial estimate.

### Revised development measurements

| Policy | Original negative errors | Revised negative errors | Revised abstention F1 | Original control F1 | Revised control F1 |
|---|---:|---:|---:|---:|---:|
| B0 | 6/79 | 4/79 (5.06%) | 0.9740 | 0.4487 | 0.4490 |
| B1 | 3/79 | 1/79 (1.27%) | 0.9936 | 0.3434 | 0.3420 |
| B2 | 3/79 | 1/79 (1.27%) | 0.9936 | 0.3434 | 0.3420 |

Development generation success remains 139/140 per policy, with no control
abstentions and 100% control citation-index validity. Control EM remains 5/61
for B0 and 0/61 for B1/B2. Each policy now has zero ablation and weak-evidence
decision errors. The remaining development errors are three non-abstentions
and one failure for B0, and one failure for B1/B2.

**B0 remains selected.** B1/B2 still exceed the allowed 0.02 control-F1 drop
relative to B0 (about 0.1070). This lexical guardrail is not a semantic verdict:
the new B1 control response contains the correct fact but is longer. The
development-selected B2 threshold is unchanged at -7.839843750000001, and the
gate again changes no answer/abstention outcomes.

All final predictions and measurements are unchanged from the original tables:
B0 has 2/34 negative errors, while B1/B2 have 3/34. Across the revised B0/B1
outcomes there are 174 control answers, 216 negative abstentions, six negative
non-abstentions and four retained failures. The two questions associated with
provider failures were not replaced simply to improve completion rates.

The revised [development results](runs/revision/development/results.json),
[locked decision](runs/revision/development/decision.json) and
[final results](runs/revision/final/results.json) contain the full measurements.
The amendment uses previously inspected data and reused responses. Its results
are a descriptive post-review comparison, not additional independent held-out
evidence or a fresh 400-response experiment.

## Plan alignment and completion

The references checked were the repository's
[original plan](../docs/Detailed%20Test%20Plans/phase_3_abstention_test_plan.md),
[original 14a](../notebooks/Tests/14a_phase_3_abstention_preparation_colab.ipynb),
[original 14b](../notebooks/Tests/14b_phase_3_abstention_evaluation_colab.ipynb),
the seven original bundle files and the supplied
[Phase2 prompt registry](phase2_generation_prompts.yaml).
Separate programmatic checks confirm exact p2 use for B0 and exact original
`scripts/collect_abstention_predictions.py` prompt use for B1.

The three research questions remain intact: baseline abstention, the structured
policy's effect, and any additional benefit from a development-calibrated gate.
The revised school plan deliberately uses the supplied new Phase3 corpus/cases,
local execution and AI evidence review, rather than resampling the historical
reserve or claiming independent human approval. All 87 controls have their
source gold in top three; 82 distinct internal articles and 22 distinct external
source articles satisfy the original diversity minima.

Completion is **against the revised school plan**, not literal completion of
every original reporting requirement:

| Original requirement | What this experiment retains / simplifies |
|---|---|
| Human and secondary review | Individual AI evidence reviews and project-owner authorization to finalize; no independent human per-case or secondary review. |
| Full metric suite | Retains abstention F1, negative/false-abstention errors, control EM/token-F1, citation-index validity, generation success and subtype counts. Separate precision/recall, confusion-matrix, coverage and selective-risk summaries are not exported. |
| Citation F1 | Uses citation-index validity, not semantic citation precision/recall or entailment certification. |
| Grouped 95% bootstrap intervals | Not computed; results are descriptive. Source-related cases are not assumed independent, and no significance or population claim is made. |
| Separate controlled-context/end-to-end summaries | Scope remains in each case and governs B2 applicability; separate scope-level aggregates are not exported. |
| Latency, token and cost summaries | Token totals and per-success generation timings are retained; no dedicated latency study or billed-cost estimate. |
| Separate finalized split JSONL files | One reviewed JSONL with explicit partitions, frozen traces and separate development/final result files. |

These omissions are disclosed limitations, not completed analyses. No labels,
prompts, thresholds, eligibility rules or final measurements were changed in
order to close the experiment or improve its reported outcome.

## Conclusion and use of the results

Under the revised school-project plan, **Phase3 is complete and these results
are ready to use in the project report**. The baseline usually abstained when
the reviewed evidence was insufficient, but related facts, unsupported
relationships, false-premise handling and provider failures exposed distinct
failure modes. B1 did not demonstrate a consistent improvement over B0, and B2
added no answer/abstention benefit at the development-selected threshold.
Accordingly, the development-selected B0 remains the preferred policy for this
experiment. These are bounded findings, not universal factual-accuracy or
production-safety claims.

The original experiment and the bounded ambiguity amendment are complete.
No further Gemini generation, relabeling or threshold tuning is needed for
these comparisons. Both versions' frozen data and outputs, and the shared
request ledger, are retained. Further methodological changes belong to a
separate future experiment.

Supporting artifacts:

- [Development results](runs/development/results.json) and [comparison CSV](runs/development/comparison.csv)
- [Locked development decision](runs/development/decision.json)
- [Final results](runs/final/results.json) and [comparison CSV](runs/final/comparison.csv)
- [Revised development results](runs/revision/development/results.json) and [comparison CSV](runs/revision/development/comparison.csv)
- [Revised final results](runs/revision/final/results.json) and [comparison CSV](runs/revision/final/comparison.csv)
- [All 200 case reviews](phase3_full_review_bundle/results/phase3/reviewed/CASE_REVIEW.md)
- [Actual retrieval evidence findings](runs/retrieval_review.jsonl)
- [Exact-request reuse and earlier usage](runs/label_audit/reuse_audit.json)
- [Revised plan](phase_3_abstention_test_plan.md) and [local operating guide](LOCAL_RUN.md)

## Appendix A — API accounting

The benchmark required **409 physical API attempts**: 396 successes, 12
invalid/no-content responses and one earlier interrupted attempt. This is below
the local 475-attempt safety ceiling (500/day with 25 reserved; 15 RPM).
Provider-reported benchmark totals are 822,776 input tokens and 7,665 output
tokens, or 830,441 total. The interrupted request may have unreported usage.

The original 153 responses were reused only after verifying all 400 old/new
request payloads were identical. Physical usage combines the old run's 154
attempts with the corrected run's 255; the latter alone is not the total.

Two later diagnostic B0 replays are separate from the benchmark: ledger
requests 410/411, with 4,038 additional input tokens and zero output tokens.
The original run and diagnostics account for 411 physical attempts, not 411
benchmark generations. The amendment adds six successful requests (IDs 412–417),
with 10,069 input and 111 output tokens, or 10,180 total. The shared ledger now
contains **417 attempts**: 411 on September 6 and six on September 7 in the
provider's Pacific-time quota days. No usage was reset when the credential
changed. B2 adds no API calls. Token totals do not constitute a billed-cost
estimate or a dedicated latency study.

## Appendix B — Provider diagnostics and historical runtime

A **provider block** means Gemini's safety system withheld a usable response.
It is different from the model returning the experiment's insufficient-evidence
sentence or JSON abstention. It is recorded as a generation failure, not a
correct abstention or a demonstrated factual error.

- **Provider filtering reproduced on two questions.** Authored row 9 (the
  teacher/student counterfactual) and row 38 (the Harrison Ford conservation
  video external question) exhausted three attempts in each policy. All 12
  original attempts reported zero output tokens; the old logger lost the finish
  reason. One unchanged B0 diagnostic request per question subsequently returned
  `content_filter: PROHIBITED_CONTENT`, zero output tokens and no `message` field.
  This directly confirms filtering on both diagnostic replays. It strongly
  supports the same explanation for the original failures, but does not recover
  all 12 original finish reasons or independently replay B1.
  Sensitive retrieved distractors are plausible triggers; the exact triggering
  passage was not isolated. Google documents non-adjustable child-safety
  protections and safety feedback in its [safety guide](https://ai.google.dev/gemini-api/docs/safety-settings).
  No filters, prompts or contexts were weakened to force a successful response.

The raw diagnostic evidence is retained in `api_usage.sqlite`, request IDs
410 and 411, under run
`4b5973e59c8b3c7dbd64bef3b7eedb2325e5dd67231d0f6f66b0a6ecac127ac2`.
Each record includes a request-payload hash and `benchmark_unchanged: true`.
They are not replacement predictions or additional final-set measurements.

### Retrieved-context evidence and causal limits

The two questions remain suitable for their intended categories. Row 9 changes
the Hebron victim from a student to a teacher; the retrieved source supports the
student incident, not the altered premise. Row 38 asks about Harrison Ford's
conservation video; the answering source is withheld and none of its five
retrieved passages supplies that fact.

The frozen contexts expose a plausible reason for provider filtering:

| Case | Unrelated sensitive passages sent to Gemini |
|---|---|
| Authored row 9, counterfactual | Contexts 2 and 3 (`acc543941d16_chunk_0` and `_chunk_1`) concern a child-exploitation investigation; context 5 (`18c5e0cc4469_chunk_0`) concerns another child-abuse criminal case. |
| Authored row 38, external-unanswerable | Context 4 (`3812dd0eb7d1_chunk_1`) concerns a child-abuse video investigation, not Harrison Ford's conservation video. |

Across all 200 frozen context sets, these four passages appear only in these
two cases. The linked Hebron student control succeeds under both policies with
the same relevant source but different distractors. The Ford chest-waxing fact
appears only in withheld source evidence, not in the generation request.
These observations strengthen the retrieved-content explanation, but do not
isolate an individual triggering passage or establish a specific safety
category. The compatibility response also does not establish whether blocking
occurred during input screening or response generation.

Original failed attempts are ledger IDs 240–245 and 392–397. Each returned zero
output tokens; neighboring requests succeeded, and these twelve attempts were
not recorded as HTTP authentication or quota errors. The diagnostic B0 payload
hashes match the frozen original requests. Provider filtering is directly
confirmed for the two diagnostic responses and strongly supported for the
original failures, whose finish reasons remain unavailable. The retrieved
passages are preserved in [the original traces](runs/retrievals.json) and remain
unchanged for these cases in [the amended traces](runs/revision/retrievals.json).

### Runner correction and experimental interpretation

The post-run fix in `phase3_api.py` reads `finish_reason` before accessing
`message`, records content-filter results as terminal `provider_blocked`
failures, and recovers them after restart without another request. It retains
the provider reason and token usage. Other invalid responses keep bounded
retries. Tests confirm that failures cannot become correct abstentions through
scoring or the B2 gate. The original four exhausted results remain untouched;
this post-run runtime fix does not retroactively change the experiment.

Offline replay of the two saved diagnostic responses reproduces the old
runner's three-attempt exhaustion and lost reason. The corrected runner records
one terminal block and reuses it on restart without another request. This check
passes through both policy paths, but it is a simulation using the saved B0
responses, not an independent live B1 diagnostic or a successful regeneration.

The findings distinguish provider availability from evidence-based abstention:
irrelevant retrieved material can prevent any usable response to an otherwise
suitable question. The four failures remain in both versions' denominators;
no sensitive passage was removed or question replaced to force completion.
Further replacement or retrieval-policy experiments would require a separately
reported amendment, not an overwrite of these outcomes.

For reproducibility, the exact pre-fix API source used for generation is retained
as inert text in [phase3_api.executed.py.txt](runs/phase3_api.executed.py.txt).
Its SHA-256 is
`f947fad6be0853e2636c61f083b9771ca133f92a0499d26a2d1120264f272bc9`.
Substituting this source hash into the existing code/input identity reproduces
the saved run identity
`b176ba8cedf6e00702eaf7e1aa9d4008a44bd892f6aa4a274663b1aeb812311f`.
This is historical evidence, not an instruction to restore or execute the old
bug. The active `phase3_api.py` retains the corrected behavior.
