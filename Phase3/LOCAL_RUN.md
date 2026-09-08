# Local Phase3 — school experiment

**Completed original run and ambiguity revision:** see [PHASE3_RESULTS.md](PHASE3_RESULTS.md).
Both versions contain all 200 cases. The notebooks now default to
`revision_config.json` and `runs/revision/`; default Run All in 14b displays the
revised results without API calls. Leave `EXECUTE=False`. Set the notebook's
`CONFIG` to `local_config.json` to inspect the preserved original experiment.
The four historical failures remain explicit failures in both versions.

The revision clarifies the wildfire control and two negative questions. It
reuses 394 exact-request outcomes and contains six fresh successful responses.
No generation rerun is needed. The post-run provider-block fix still protects
the original run from execution under a changed code identity.

Open the existing **14a** and **14b** notebooks. They now contain local Python
code; their `_colab` filenames are retained so existing links still work.
No cell installs packages, downloads data/models, requests human approval, or
contains an API key. Run All defaults to preparation/read-only checks, not API
generation.

## The short workflow

Preparation and generation are already complete. To inspect them now, use
14b's default Run All and the report. The revision workflow is:

1. **14a:** apply the three recorded question edits in a separate draft and
   collect fresh selected retrieval for the changed control. The original
   corpus/index and unchanged traces are reused; nothing is resampled.
2. **Evidence review and freeze:** check the control's new passages and the two
   negatives' retained passages before generation. Store actual findings in
   the selected work directory. This review is already complete and remains
   AI-only, not human sign-off or an LLM-judge scoring stage.
3. **Exact-request reuse:** `phase3_revision.py seed` verifies all 400 old/new
   payloads and carries forward 394 unchanged outcomes, including failures.
4. **14b:** explicit execution runs smoke, development and final using the
   selected configuration. Supply the current key privately through its hidden
   prompt or `GEMINI_API_KEY`. B2 reuses B1 and adds no API calls.

The revised results are in `runs/revision/<stage>/results.json` and
`comparison.csv`. Original results remain in `runs/<stage>/`.
They show answers, refusals, generation failures, control EM/token F1,
citation-index validity and changes relative to B0. No RAGAS. A counterfactual
correction is flagged for interpretation, not automatically called a false fact.

## Inputs and changes

The seven original review-bundle files and uploaded `chunks.jsonl` remain
unchanged. The unused historical `testset_resolved.jsonl` was removed from
Phase3 and is recoverable in the system Trash; it is not a workflow input.
The original reviewed draft remains at:

`phase3_full_review_bundle/results/phase3/reviewed/`

The amended draft and its 200 case reviews are in `runs/revision/reviewed/`.
`phase3_revision.py` applies only the three documented question corrections;
`runs/revision/reuse_audit.json` records which outcomes were reused and why.

`CASE_REVIEW.md` covers every one of the 200 slots. `cases.jsonl` includes exact
source/provided passages, original row IDs, repairs and reasons.
`review_decisions.json` is the small, editable source of those repairs.
The draft's `pending_retrieval`/`ready_for_generation=false` fields describe its
construction stage, not the current run. Actual generation readiness comes from
`phase3_run.py smoke --config Phase3/revision_config.json` (without `--execute`,
from the repository root), which checks the selected work directory's accepted
`retrieval_review.jsonl` and `frozen.json`. Notes labeled "Original
proposal assessment" preserve the earlier findings, not unresolved current holds.
For recovered external proposals, `construction.source_evidence_spans` preserves
the original dataset's answer offsets. After a question repair, those spans may
not be the current answer anchor: use `source_expected_answer`, the repair's
evidence quote and the full retained `source_evidence` passage together.

Four replacement external questions use complete articles already supplied in
`chunks.jsonl`. Their four articles are withheld **globally** from the active
Phase3 corpus, leaving 19,259 chunks. Nothing is deleted from the uploaded file.
By contrast, removed-article cases exclude their source only for that case.
All policies receive the same resulting contexts. The 200 cases, 140/60 split
and 22 distinct external source articles are preserved. Actual retrieval required
count adjustments: 87 controls and three natural misses instead of 80/10.
Forcing the old quota would have retained eight incorrect negative labels;
the Barnett correction prevents a missing statement from being scored as a
false control abstention. See the plan for the mid-development correction and
the evidence-checked reuse of already completed requests.

Original cloud code and the earlier auxiliary pre-work are recoverable at:

`/home/rachal/projects/text_mining/phase3-prework-backup.onHAdT/`

The auxiliary scripts/snapshots are no longer part of the active workflow.

Redundant historical ledger/retrieval snapshots, superseded smoke exports,
completed progress files and disposable cache files are recoverable in the
system Trash. All historical requests and predictions remain in the main
`api_usage.sqlite`; no quota or generation evidence was reset.

Keep the remaining `runs/label_audit/` files as provenance: the earlier cases,
review, freeze identity and exact-request reuse audit. Its historical retrieval
snapshot is reconstructible from the identical `runs/retrievals.json` traces
and the identity in `label_audit/frozen.before.json`, so a second large copy is
unnecessary. The one-off `reuse_label_only.py.txt` and pre-fix
`runs/phase3_api.executed.py.txt` are inert historical source, not runnable
workflow steps. Keep the local venv, selected index, model caches, source
evidence, original inputs and both completed experiment versions.

## Local settings

`local_config.json` uses this machine's RTX 5060 Ti, `device="cuda"`, and small
GPU batches. The isolated `Phase3/.venv` and **NewsQA Phase3 (local CUDA)** kernel
are installed. Choose that kernel in both notebooks. B0 reads **p2**
from the supplied `phase2_generation_prompts.yaml`. The selected models remain
BGE-M3, BGE-reranker-large and `gemini-3.1-flash-lite`; no automatic substitutes.

Retrieval requires Python 3.11+ and locally available project dependencies and
weights. A matching cached index is built locally from the active corpus, not
downloaded. Index building may be slow on CPU and restarts from the beginning
if interrupted; completed case retrievals and Gemini requests resume.

The local CUDA setting uses PyTorch's `matmul_precision="high"` (TensorFloat32
internal multiplication, float32 outputs). It is bound into the index/run
identity and used for every policy. It is not bit-identical to strict float32;
an eight-passage check found identical sparse token sets and maximum weight
difference 0.0066, with about 3.6× lower encoding time. Set `highest` and use a
fresh work directory if strict float32 is required. Models and retrieval logic
are unchanged. Local bulk batching also avoids repeated sizing passes in
FlagEmbedding. See [PyTorch's precision documentation](https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html).

Setup is already done here. To recreate it after an environment move (downloads
are explicitly authorized), from the repository root:

```bash
uv venv --python 3.12 Phase3/.venv
uv pip install --python Phase3/.venv/bin/python 'torch==2.7.1' --index https://download.pytorch.org/whl/cu128
uv pip install --python Phase3/.venv/bin/python -r Phase3/requirements-local.txt
Phase3/.venv/bin/python -m ipykernel install --user --name newsqa-phase3 --display-name 'NewsQA Phase3 (local CUDA)'
Phase3/.venv/bin/hf download BAAI/bge-m3 --include 'config.json' 'pytorch_model.bin' 'tokenizer*' 'special_tokens_map.json' 'sentencepiece.bpe.model' 'sparse_linear.pt' 'colbert_linear.pt'
Phase3/.venv/bin/hf download BAAI/bge-reranker-large --include 'config.json' 'model.safetensors' 'tokenizer*' 'special_tokens_map.json' 'sentencepiece.bpe.model'
```

These downloads also populate the `main` cache reference used by the shared
offline model loaders. The tested snapshots are BGE-M3
`5617a9f61b028005a4858fdac845db406aefb181` and BGE-large
`55611d7bca2a7133960a6d3b71e083071bbfc312`. If recreating the setup later with
changed weights, collect and review fresh contexts in a new work directory.

The retained external provenance file is already supplied. To recover it again,
`Phase3/.venv/bin/python Phase3/recover_external_sources.py` fetches only the
1.6 MB validation file at the original proposals' revision and extracts the 18
matching passages. It does not resample Phase2 or change retrieval data.

Command-line equivalents from the repository root:

```bash
source Phase3/.venv/bin/activate
python Phase3/review_phase3.py
python Phase3/phase3_revision.py prepare
python Phase3/phase3_run.py retrieve --config Phase3/revision_config.json
python Phase3/phase3_run.py retrieve --config Phase3/revision_config.json --execute
# After actual AI inspection of the resulting contexts:
python Phase3/phase3_run.py freeze --config Phase3/revision_config.json --execute
python Phase3/phase3_revision.py seed
python Phase3/phase3_run.py smoke --config Phase3/revision_config.json
# Only if generation is incomplete; asks for the current key privately:
python Phase3/phase3_revision.py run
```

Without `--execute`, the runner only checks readiness. Neither a successful
structural review nor a completed smoke test means all 200 labels are correct.
If a natural-miss candidate retrieves an answer, document the evidence and update
its label and the plan counts before generation. Do not fabricate a deletion or
deliberately weaken the question to force a target quota.

## API quota and restart

The [Gemini OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai)
is called directly, with no SDK auto-retries. The user's ceilings are 15 RPM
and 500/day; the default reserve leaves 475 local attempts. A clean run uses
400 calls because smoke responses are reused during development. Retries count
and are limited to three per case/policy across restarts. Explicit content-filter
blocks are terminal failures: the reason and usage are saved after one response,
and restarting does not retry them. They are never counted as correct abstentions.

Usage is 409 original benchmark attempts, two diagnostic requests and six
successful amendment requests: 417 ledger attempts in total. The six amendment
requests used the newly supplied credential without saving it. There is no
hardcoded old key to replace; update `GEMINI_API_KEY` yourself if an older key
is still set in your own terminal, or use the hidden prompt. The 15 RPM /
500-per-day ceilings and 25-request reserve remain unchanged. A new key does
not reset project quota or justify deleting the shared ledger.

Keep `Phase3/api_usage.sqlite`. It records attempts and responses, **not the
credential**, across all stages and run folders. Do not delete it to reset
quota. HTTP 429/auth failures pause with progress saved. Other applications and
provider token limits may still exhaust quota; local ceilings are not a claim
about the account's actual entitlement. See Google's
[rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

## Offline checks

```bash
python -m unittest discover -s Phase3 -p 'test_phase3*.py' -v
```

The 50 offline tests use fake HTTP and temporary synthetic results, including
five focused amendment/reuse tests.
Both notebooks' default cells also pass in the registered local kernel with
HTTP and key prompts blocked. All 200 frozen context sets, the 400 B0/B1 saved
outcomes, B2 derivations, metrics, comparison CSVs and quota accounting reconcile.
These checks do not certify semantic correctness or create experimental
findings; the report separately describes the AI evidence review and its limits.
