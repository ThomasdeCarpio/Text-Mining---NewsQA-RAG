# Phase3 — school RAG abstention experiment

## Objective

Compare the selected Phase2 configuration (B0), a structured abstention prompt
(B1), and a development-calibrated reranker gate (B2). This is a new Phase3
dataset, not a reproduction of the old testset. Historical testset, reserve-file
and corpus checksums must not decide the quality of new questions.

Keep p2, BGE-M3 sparse retrieval, BGE-large reranking, depth 5, and Gemini 3.1
Flash-Lite (temperature 0, maximum output 512, minimal reasoning). No RAGAS or
LLM judge. Compare answers/refusals, control EM/token F1, citations and failures.
Automatic flags are diagnostics, not proof of hallucination. In particular, a
helpful correction of a counterfactual premise is not automatically a false fact.
The reported false-answer rate measures the negative-case abstention contract,
not a fact-checked hallucination rate. Citation validity measures valid indices,
not whether the cited sentence entails the answer.

## Data and evidence

Use the actual review bundle: 156 generated cases plus 44 authored proposals.
Preserve the existing article/base-question partitions. The target is:

| Type | Development | Final | Total |
|---|---:|---:|---:|
| answerable_control | 61 | 26 | 87 |
| natural_retrieval_miss | 2 | 1 | 3 |
| controlled_context_ablation | 16 | 6 | 22 |
| removed_article | 16 | 6 | 22 |
| counterfactual | 15 | 7 | 22 |
| external_unanswerable | 15 | 7 | 22 |
| partial_weak_evidence | 15 | 7 | 22 |
| Total | 140 | 60 | 200 |

The original target was 80 controls and 10 natural misses. Fresh local retrieval
answered eight of those proposed misses (six development, two final). They are
now controls; no questions were weakened or evidence deleted to force ten misses.
One former control also missed the answering passage: Barnett's requested
statement is in chunk 1, while only his name and general role occur in retrieved
chunk 0. This label correction was found during development, after 153 successful
responses and before any final generation. The three misses are descriptive
examples, not a reliable subtype-rate estimate: Barnett's statement (a topic/name
overlap boundary case), the short, underspecified ICE-lawsuit query and the
specific 2011 Wales–Ireland quarterfinal-score query. Other counts are unchanged.

These are observed, reviewed slots, not permission to accept bad labels. Every case needs a
question, an explicit scope, a source/evidence explanation and an AI assessment.
Never fake human approval. Source answers on negatives are construction metadata,
not expected model answers. Every external case must retain a supporting source
passage. Source support and absence from the active corpus are separate checks.

The revised external set retains 18 original proposals with recovered source
passages and uses four newly authored questions backed by complete articles
already in the uploaded corpus for four defective slots. Withhold those four
articles globally for this experiment (19,259 active chunks out of 19,263
uploaded); preserve the uploaded file. This is different from per-case article
removal. Keep all 22 external source articles distinct.

Controls require clear source answers and adequate generation evidence.
Ablations remove all answering passages, retaining nonempty contexts.
Weak evidence remains on topic but lacks the requested fact. Natural misses
must actually be produced by the selected retriever, not by deleting context.
Removed-article cases exclude all physical/canonical source chunks and must not
remain answerable from another article. External facts must be unsupported in
the chosen Phase3 corpus, not merely belong to a different article ID.

Counterfactual questions intentionally change a premise. Prefer one decisive
fact change; avoid answer-in-question, incidental descriptors and reasonable
zero/none answers. Clearly distinguish strict abstention-format compliance from
factual correction. Do not claim that every non-abstention is a hallucination.

Review each record, not only exact answer strings: aliases, paraphrases and
inference can make a negative answerable. Repairs retain their reasons and
original records. Changed questions/contexts invalidate old retrieval. Do not
rerun historical 14a to resample or overwrite reviewed cases.

## Workflow

1. Audit/repair the supplied bundle itself. No original-testset comparison.
2. Bind cases to the actual Phase3 corpus and embedded context texts. Use one
   frozen corpus for B0/B1/B2; never silently substitute texts under the same ID.
3. Collect the selected BGE retrieval before generation. Check controls and
   negative labels against resulting contexts; unresolved cases stay held.
   Natural misses keep their original provided-context measurement scope (B2
   does not gate them), but their contexts must be newly observed when the
   question or retrieval corpus changes, not copied or manually ablated.
4. Freeze the reviewed dataset and contexts. Run one development smoke case per
   type, then development 140, then final 60 without tuning on final.
5. B0/B1 receive identical contexts. B2 is derived without extra API calls and
   applies only to full-corpus cases with actual reranker scores.
6. Choose the threshold on development with false abstention <=10%; prefer lower
   false-answer rate, then higher abstention F1, then the lower threshold.
   If no threshold qualifies, report B2 as infeasible; do not relax criteria.
7. Select the development policy with generation success >=98%, false abstention
   <=10%, control F1 drop <=0.02 and citation validity drop <=0.01 relative to B0.
   Prefer lower false-answer rate; within 0.02 prefer B0, then B1, then B2.
   Final reports all available comparisons without reselecting the policy.
8. Keep actual per-case outputs and simple regression flags. Small subtype
   samples support descriptive conclusions, not production or population claims.

## Post-review ambiguity amendment

The completed original experiment is retained as the reference comparison.
Amend the linked wildfire control, ablation and weak-evidence cases (original
queue rows 13, 63 and 87) to ask: "What was found in a canyon east of San Diego
in the path of the Harris Fire?" This identifies the intended finding without
giving its answer. The existing counterfactual uses the same wording with San
Francisco substituted for San Diego and needs no further change.

Keep all 200 case IDs, labels, source answers, partitions, corpus exclusions,
models, prompts and scoring rules. Recheck the unchanged provided passages for
both negatives; obtain fresh selected retrieval and an evidence review for the
changed control. Store revised cases and outputs separately from the original.

Verify all old/new request payloads before reusing the 394 unchanged B0/B1
outcomes. This includes the four genuine generation failures: neither a new key
nor a desire for complete answers justifies replacing valid blocked cases.
Generate the six changed requests, recalibrate B2 and select the policy using
revised development only, then apply the locked decision to the unchanged final
set. Respect the shared request ledger and existing quota limits.

Report original and revised measurements separately. This is a post-review
amendment with reused responses, not a fresh 400-response run or an untouched
held-out test. Keep lexical metrics and qualitative error explanations distinct;
do not add RAGAS, an automated judge or tune rules to obtain better final scores.

## Reporting scope and completion

This revised school plan retains the original three B0/B1/B2 research questions
and development-only selection. The original final split was held out during
policy selection; the post-review amendment is a descriptive comparison, not
new held-out evidence. Completion requires
all 200 cases to have recorded outcomes, including failures, and an honest
account of generation behavior; it does not require B1/B2 to outperform B0.

The original plan additionally specified human/secondary review, a fuller metric
suite, citation F1, grouped bootstrap intervals, separate scope aggregates and
latency/cost reporting. This experiment uses AI-only review and descriptive
metrics, without those additional analyses. Split JSONL packaging is also
simplified. The explicit original-versus-revised reporting table and original
reference links are in `PHASE3_RESULTS.md`; these omissions must not be presented
as completed analyses or statistically significant findings. No final results
are retuned to close the experiment.

The experiment is complete when the following conditions are met:

- All 200 cases have an evidence review, documented repairs where needed and
  frozen generation contexts; the review method is reported accurately.
- All 400 B0/B1 outcomes, including failures, are retained, and B2 is derived
  without extra generation.
- Threshold and policy selection use development only; final measurements are
  reported without retuning.
- The report answers all three research questions and presents the retained
  core/subtype metrics with definitions and correct denominators.
- Errors, limitations and original-plan simplifications are disclosed, with
  sufficient retained evidence to reproduce the reported measurements.

These conditions are met by the completed experiment described in
[the results report](PHASE3_RESULTS.md). The project owner's authorization
permits use of the AI-reviewed experiment; it does not establish independent
human per-case or secondary review. Requirements explicitly simplified above
remain disclosed omissions.

## Local/API operation

Use p2 from the supplied `phase2_generation_prompts.yaml` as B0; p0, p1 and p3
are not the selected baseline. Preserve the original numbered Context/Question
message layout for the comparison.

Use the current Gemini credential through a hidden prompt or environment variable.
Replacing an expired credential does not change the experimental configuration
or reset quota. Never save the key in code/notebook outputs. User ceilings: 15 RPM and
500/day; count retries and preserve usage/completed responses across restarts.
Stop on quota/authentication failures; never erase the ledger to reset usage.
Record explicit provider content-filter blocks as terminal failures with their
reason and usage; do not retry them or score them as correct abstentions. This
diagnostic/retry fix was made after generation and leaves historical results intact.
Setup uses an isolated Python 3.12 venv, CUDA-12.8 PyTorch and the selected BGE
weights, installed outside the notebooks. CUDA matmul precision is explicitly `high` and bound
to the index/run, not silently changed between policies. This is a fresh local
comparison, not a bit-identical reproduction of historical Phase2 scores.
