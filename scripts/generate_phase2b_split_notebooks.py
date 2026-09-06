#!/usr/bin/env python3
"""Generate the distributed Phase 2B Kaggle/Colab execution notebooks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "notebooks/Tests/13_phase_2_generation_tuning_kaggle.ipynb"
OUTPUT = SOURCE.parent

REPO_COMMIT = "973e1a7758de1defeda7f88d0cb60e34f57eb51f"

VARIANTS = [
    ("13a_phase_2b_0_preparation_kaggle.ipynb", "prepare", "kaggle", {}),
    ("13b_phase_2b_1_prompt_p1_colab.ipynb", "prompt", "colab", {"prompt_id": "p1"}),
    ("13c_phase_2b_1_prompt_p2_colab.ipynb", "prompt", "colab", {"prompt_id": "p2"}),
    ("13d_phase_2b_1_prompt_p3_colab.ipynb", "prompt", "colab", {"prompt_id": "p3"}),
    ("13e_phase_2b_2_depth_1_colab.ipynb", "context", "colab", {"depth": 1}),
    ("13f_phase_2b_2_depth_3_colab.ipynb", "context", "colab", {"depth": 3}),
    ("13g_phase_2b_3_finalist_1_kaggle.ipynb", "finalist", "kaggle", {"slot": 1}),
    ("13h_phase_2b_3_finalist_2_colab.ipynb", "finalist", "colab", {"slot": 2}),
    ("13i_phase_2b_4_heldout_final_kaggle.ipynb", "heldout", "kaggle", {}),
]


def _cell(notebook: dict, cell_id: str) -> dict:
    return copy.deepcopy(next(cell for cell in notebook["cells"] if cell.get("id") == cell_id))


def _markdown(cell_id: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(True)}


def _code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def _configuration(stage: str, platform: str, options: dict) -> str:
    if stage == "prepare":
        run_id = "phase2b_0_preparation"
    elif stage == "prompt":
        run_id = f"phase2b_1_{options['prompt_id']}"
    elif stage == "context":
        run_id = f"phase2b_2_depth_{options['depth']}"
    elif stage == "finalist":
        run_id = f"phase2b_3_finalist_{options['slot']}"
    else:
        run_id = "phase2b_4_heldout_final"
    baseline_source = ""
    baseline_results_comment = "Phase 2B.0 only: attached Phase 2A full-results ZIP/directory."
    if stage == "prepare":
        baseline_results_comment = "Optional local ZIP/directory override; otherwise download from private HF."
        baseline_source = """HF_BASELINE_REPO_ID='ThomasAnderson2009/newsqa-rag-phase2-experiments'
HF_BASELINE_REVISION='bd83647c784d7f7d466bbd954fe7d35b5252a2f5'  # phase2a-baseline-deduplicated-v2
HF_BASELINE_FILENAME='phase2a/baseline-deduplicated-v2/phase2_e2e_baseline_full_results.zip'
HF_BASELINE_SHA256='044c913d20942c2dc8c1dcfa39544a7baf082f386b6d5f934e01c9c7feaf6017'
"""
    specific = ""
    if stage == "prompt":
        specific = f"PROMPT_ID='{options['prompt_id']}'\n"
    elif stage == "context":
        specific = (
            "PROMPT_FINALISTS=[]  # Set exactly two IDs selected from the three Phase 2B.1 results.\n"
            f"CONTEXT_DEPTH={options['depth']}\n"
        )
    elif stage == "finalist":
        specific = (
            "FINALIST_CONFIG=None  # Set {'prompt_id':'p1','context_depth':1} from Phase 2B.2.\n"
            f"FINALIST_SLOT={options['slot']}\n"
        )
    elif stage == "heldout":
        specific = (
            "LOCKED_WINNER=None  # Set the approved winner from the two Phase 2B.3 reports.\n"
            "HELDOUT_APPROVED=False  # Change only after recording the winner decision.\n"
        )
    return f"""from pathlib import Path
REPO_URL='https://github.com/ThomasdeCarpio/Text-Mining---NewsQA-RAG.git'
REPO_COMMIT='{REPO_COMMIT}'
PLATFORM='{platform}'
RUN_ID='{run_id}'
PREPARATION_BUNDLE_PATH=''  # Required except in Phase 2B.0; Drive path on Colab or attached Kaggle input.
BASELINE_RESULTS_PATH=''  # {baseline_results_comment}
{baseline_source}RESTORE_CHECKPOINT_PATH=''  # Optional checkpoint from the same notebook only.
EXECUTE_API_CALLS=False  # Set True only for an approved execution notebook.
GEMINI_SECRET_NAME='GEMINI_API_KEY_1'  # Give each Colab prompt/depth notebook its own key.
UPSTREAM_DECISION_PATH=''  # Optional JSON decision record; retained in provenance.
RUN_STAGE=RUN_ID
PROMPT_ID=None
PROMPT_FINALISTS=[]
FINALIST_CONFIG=None
FINALIST_CONFIGS=[]
LOCKED_WINNER=None
HELDOUT_APPROVED=False
CONTEXT_DEPTH=None
{specific}SEED=42
SCREENING_QUESTIONS=80
JUDGE_CALIBRATION_QUESTIONS=20
FINAL_HELDOUT_ARTICLES=50
GENERATOR_MODEL='gemini-3.1-flash-lite'
GENERATOR_REASONING_EFFORT='minimal'
GENERATOR_MAX_TOKENS=512
GENERATOR_MIN_INTERVAL_SECONDS=4.2
JUDGE_MODEL='accounts/fireworks/models/glm-5p3-flash'
JUDGE_REASONING_EFFORT='low'
JUDGE_MAX_TOKENS=2048
GENERATOR_INPUT_PER_MILLION_USD=0.25
GENERATOR_OUTPUT_PER_MILLION_USD=1.50
JUDGE_INPUT_PER_MILLION_USD=0.15
JUDGE_OUTPUT_PER_MILLION_USD=0.50
TOP_K=20
RERANK_TOP_N=5
"""


def _setup(stage: str, platform: str) -> str:
    platform_setup = """KAGGLE_INPUT=Path('/kaggle/input'); RUNTIME_ROOT=Path('/kaggle/working')
from kaggle_secrets import UserSecretsClient
secrets=UserSecretsClient()
def optional_secret(name):
    try: return secrets.get_secret(name) or ''
    except Exception: return ''
""" if platform == "kaggle" else """from google.colab import drive, userdata
drive.mount('/content/drive')
KAGGLE_INPUT=Path('/content/drive/MyDrive'); RUNTIME_ROOT=Path('/content')
def optional_secret(name):
    try: return userdata.get(name) or ''
    except Exception: return ''
"""
    restore = "" if stage == "prepare" else """prep_input=Path(PREPARATION_BUNDLE_PATH) if PREPARATION_BUNDLE_PATH else None
if prep_input is None and PLATFORM=='kaggle':
    candidates=sorted(KAGGLE_INPUT.rglob('phase2b_preparation_bundle.zip'))
    prep_input=candidates[-1] if candidates else None
assert prep_input and prep_input.exists(), 'Attach/set PREPARATION_BUNDLE_PATH from Phase 2B.0'
WORK_ROOT.mkdir(parents=True,exist_ok=True); shutil.unpack_archive(prep_input,WORK_ROOT)
"""
    return f"""import hashlib, json, os, shutil, string, subprocess, sys, time, zipfile
{platform_setup}PROJECT_ROOT=RUNTIME_ROOT/'Text-Mining---NewsQA-RAG'
WORK_ROOT=RUNTIME_ROOT/RUN_ID
DATA_ROOT=WORK_ROOT/'data'; INDEX_ROOT=WORK_ROOT/'index'; BASELINE_ROOT=WORK_ROOT/'baseline'
RUNS_ROOT=WORK_ROOT/'runs'; IDS_ROOT=WORK_ROOT/'question_ids'; PROMPT_ROOT=WORK_ROOT/'prompts'; RESULTS=WORK_ROOT/'results'; LOGS=WORK_ROOT/'logs'
{restore}checkpoint_input=Path(RESTORE_CHECKPOINT_PATH) if RESTORE_CHECKPOINT_PATH else None
if checkpoint_input and checkpoint_input.exists():
    WORK_ROOT.mkdir(parents=True,exist_ok=True); shutil.unpack_archive(checkpoint_input,WORK_ROOT); print('Restored checkpoint:',checkpoint_input)
for path in [DATA_ROOT,INDEX_ROOT,BASELINE_ROOT,RUNS_ROOT,IDS_ROOT,PROMPT_ROOT,RESULTS,LOGS]: path.mkdir(parents=True,exist_ok=True)
assert not REPO_COMMIT.startswith('SET_TO_'), 'Pin REPO_COMMIT after committing the split notebooks'
if not PROJECT_ROOT.exists(): subprocess.run(['git','clone','--filter=blob:none',REPO_URL,str(PROJECT_ROOT)],check=True)
subprocess.run(['git','fetch','--depth=1','origin',REPO_COMMIT],cwd=PROJECT_ROOT,check=True,timeout=180)
subprocess.run(['git','checkout','--detach',REPO_COMMIT],cwd=PROJECT_ROOT,check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-kaggle.txt'],cwd=PROJECT_ROOT,check=True)
GENERATOR_API_KEY=optional_secret(GEMINI_SECRET_NAME); JUDGE_API_KEY=optional_secret('FIREWORKS_API_KEY'); HF_TOKEN=optional_secret('HF_TOKEN')
os.environ.update({{'HF_HOME':str(RUNTIME_ROOT/'hf_cache'),'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','LANGCHAIN_TRACING_V2':'false','LANGSMITH_TRACING':'false'}})
if EXECUTE_API_CALLS:
    assert GENERATOR_API_KEY, f'Configure the secret named {{GEMINI_SECRET_NAME}}'
    assert JUDGE_API_KEY, 'Configure FIREWORKS_API_KEY'
print('Run:',RUN_ID,'| platform:',PLATFORM,'| API execution:',EXECUTE_API_CALLS)
"""


def _helpers(source: dict) -> dict:
    cell = _cell(source, "helpers")
    text = "".join(cell["source"])
    text = text.replace("checkpoint=KAGGLE_WORKING/'newsqa_phase2b_checkpoint.zip'", "checkpoint=RUNTIME_ROOT/f'{RUN_ID}_checkpoint.zip'")
    text = text.replace("for name in ['runs','question_ids','prompts','results','logs','heldout_trace']:", "for name in ['baseline','index','runs','question_ids','prompts','results','logs','heldout_trace']:")
    cell["source"] = text.splitlines(True)
    return cell


def _load_artifacts(source: dict, stage: str) -> dict:
    cell = _cell(source, "load-artifacts")
    text = "".join(cell["source"])
    start = text.index("baseline_input=Path(BASELINE_RESULTS_PATH)")
    end = text.index("assert baseline_manifest['inputs']['generator_model']")
    if stage == "prepare":
        block = """baseline_input=Path(BASELINE_RESULTS_PATH) if BASELINE_RESULTS_PATH else None
if baseline_input is None:
    assert HF_TOKEN, 'Configure the read-only HF_TOKEN Kaggle secret for the private Phase 2 results repo'
    baseline_input=Path(hf_hub_download(repo_id=HF_BASELINE_REPO_ID,repo_type='dataset',revision=HF_BASELINE_REVISION,filename=HF_BASELINE_FILENAME,token=HF_TOKEN))
    assert sha256_file(baseline_input)==HF_BASELINE_SHA256, 'Phase 2A baseline archive checksum mismatch'
if baseline_input.is_file(): shutil.unpack_archive(baseline_input,BASELINE_ROOT)
else: shutil.copytree(baseline_input,BASELINE_ROOT,dirs_exist_ok=True)
"""
    else:
        block = """assert BASELINE_ROOT.exists(), 'Preparation bundle does not contain baseline inputs'
"""
    common = """manifest_candidates=list(BASELINE_ROOT.rglob('run_manifest.json')); assert manifest_candidates, 'Baseline run_manifest.json is missing'
baseline_manifest_path=next(path for path in manifest_candidates if len(json.loads(path.read_text()).get('inputs',{}).get('question_ids',[]))==281)
baseline_dir=baseline_manifest_path.parent; baseline_manifest=json.loads(baseline_manifest_path.read_text())
baseline_retrievals=baseline_dir/'retrievals.jsonl'; baseline_predictions=baseline_dir/'predictions.jsonl'; baseline_judges=baseline_dir/'judge_results.jsonl'
for path in [baseline_retrievals,baseline_predictions,baseline_judges]: assert path.exists(), path
"""
    cell["source"] = (text[:start] + block + common + text[end:]).splitlines(True)
    return cell


def _partitions(source: dict, stage: str) -> dict:
    cell = _cell(source, "partitions")
    if stage == "prepare":
        text = "".join(cell["source"])
        text = text.replace(
            "subset_manifest={'schema_version':1,",
            "subset_manifest={'schema_version':1,'baseline_artifact':{'repo_id':HF_BASELINE_REPO_ID,'revision':HF_BASELINE_REVISION,'filename':HF_BASELINE_FILENAME,'sha256':HF_BASELINE_SHA256},",
        )
        cell["source"] = text.splitlines(True)
        return cell
    text = "".join(cell["source"])
    prefix = """prepared_subset_path=RESULTS/'subset_manifest.json'
assert prepared_subset_path.exists(), 'Preparation bundle is missing subset_manifest.json'
prepared_subset_manifest=json.loads(prepared_subset_path.read_text())
"""
    suffix = """assert subset_manifest['counts']==prepared_subset_manifest['counts'], 'Subset counts differ from Phase 2B.0'
assert subset_manifest['sha256']==prepared_subset_manifest['sha256'], 'Subset IDs differ from Phase 2B.0'
assert subset_manifest['heldout_selection']==prepared_subset_manifest['heldout_selection'], 'Held-out article sample differs from Phase 2B.0'
print('Preparation subset hashes verified')
"""
    cell["source"] = (prefix + text + suffix).splitlines(True)
    return cell


def _stage_cell(stage: str, options: dict) -> dict:
    if stage == "prepare":
        source = "print('Preparation validated. No generation or judge requests were made.')\n"
    elif stage == "prompt":
        source = """assert PROMPT_ID in {'p1','p2','p3'}
assert EXECUTE_API_CALLS, 'Set EXECUTE_API_CALLS=True after checking the prompt assignment'
stage_outputs=execute_matrix([{'prompt_id':PROMPT_ID,'context_depth':5}],screening_ids,f'prompt_screen_{PROMPT_ID}',judge_calibration_ids)
"""
    elif stage == "context":
        source = """assert len(PROMPT_FINALISTS)==2 and len(set(PROMPT_FINALISTS))==2
assert set(PROMPT_FINALISTS)<={'p0','p1','p2','p3'}
assert CONTEXT_DEPTH in {1,3}
assert EXECUTE_API_CALLS, 'Set EXECUTE_API_CALLS=True after recording both prompt finalists'
stage_outputs=execute_matrix([{'prompt_id':prompt_id,'context_depth':CONTEXT_DEPTH} for prompt_id in PROMPT_FINALISTS],screening_ids,f'context_depth_{CONTEXT_DEPTH}',judge_calibration_ids)
"""
    elif stage == "finalist":
        source = """assert isinstance(FINALIST_CONFIG,dict) and set(FINALIST_CONFIG)=={'prompt_id','context_depth'}
assert FINALIST_CONFIG['prompt_id'] in {'p0','p1','p2','p3'} and FINALIST_CONFIG['context_depth'] in {1,3,5}
assert EXECUTE_API_CALLS, 'Set EXECUTE_API_CALLS=True after assigning this finalist slot'
FINALIST_CONFIGS=[FINALIST_CONFIG]
stage_outputs=execute_matrix([FINALIST_CONFIG],development_ids,f'finalist_{FINALIST_SLOT}',development_ids)
"""
    else:
        source = """assert HELDOUT_APPROVED and isinstance(LOCKED_WINNER,dict) and set(LOCKED_WINNER)=={'prompt_id','context_depth'}
assert LOCKED_WINNER['prompt_id'] in {'p0','p1','p2','p3'} and LOCKED_WINNER['context_depth'] in {1,3,5}
assert EXECUTE_API_CALLS, 'Set EXECUTE_API_CALLS=True only after locking the winner'
import torch
assert torch.cuda.is_available(), 'Enable a Kaggle GPU for held-out retrieval/reranking'
heldout_trace_dir=WORK_ROOT/'heldout_trace'; heldout_retrievals=heldout_trace_dir/'retrievals.jsonl'
command=[sys.executable,'-u','scripts/collect_benchmark_predictions.py','--retriever','sparse','--reranker','cross-encoder','--reranker-model','BAAI/bge-reranker-large','--testset',testset,'--variant-manifest',profile_path,'--config',config_path,'--run-dir',heldout_trace_dir,'--question-ids-file',IDS_ROOT/'heldout.json','--top-k',TOP_K,'--rerank-top-n',RERANK_TOP_N,'--retrieval-only','--max-attempts',3,'--retry-failed','--progress']
run_command(command,'build_heldout_trace')
original_source=baseline_retrievals; baseline_retrievals=heldout_retrievals
stage_outputs=execute_matrix([LOCKED_WINNER],heldout_ids,'heldout_final',heldout_ids)
baseline_retrievals=original_source
write_json(RESULTS/'heldout_access.json',{'approved':True,'winner':LOCKED_WINNER,'article_count':len(heldout_articles),'question_count':len(heldout_ids),'reserve_articles':100,'reserve_questions':len(heldout_reserve_ids),'executed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
"""
    return _code("execute-stage", source)


def _report(source: dict) -> dict:
    cell = _cell(source, "report")
    text = "".join(cell["source"]).replace("KAGGLE_WORKING", "RUNTIME_ROOT")
    cell["source"] = text.splitlines(True)
    return cell


def _export(stage: str, platform: str) -> dict:
    if stage == "prepare":
        source = """bundle=RUNTIME_ROOT/'phase2b_preparation_bundle.zip'
temporary=bundle.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ['baseline','index','question_ids','prompts','results']:
        root=WORK_ROOT/name
        for path in root.rglob('*'):
            if path.is_file(): archive.write(path,path.relative_to(WORK_ROOT))
temporary.replace(bundle)
print('Preparation bundle:',bundle,round(bundle.stat().st_size/2**20,1),'MiB')
"""
    else:
        destination = "Path('/content/drive/MyDrive/newsqa_phase2b')" if platform == "colab" else "RUNTIME_ROOT"
        source = f"""upstream={{'path':UPSTREAM_DECISION_PATH or None,'sha256':sha256_file(UPSTREAM_DECISION_PATH) if UPSTREAM_DECISION_PATH and Path(UPSTREAM_DECISION_PATH).exists() else None}}
write_json(RESULTS/'upstream_decision.json',upstream)
checkpoint=write_checkpoint()
result_bundle=RUNTIME_ROOT/f'{{RUN_ID}}_results.zip'; temporary=result_bundle.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ['runs','question_ids','prompts','results','logs','heldout_trace']:
        root=WORK_ROOT/name
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file(): archive.write(path,path.relative_to(WORK_ROOT))
temporary.replace(result_bundle)
destination={destination}; destination.mkdir(parents=True,exist_ok=True)
for artifact in [Path(checkpoint),Path(result_bundle)]:
    target=destination/artifact.name
    if artifact.resolve()!=target.resolve(): shutil.copy2(artifact,target)
    print('Saved:',target,round(target.stat().st_size/2**20,1),'MiB')
"""
    return _code("export", source)


def build_notebook(source: dict, stage: str, platform: str, options: dict) -> dict:
    if stage == "prepare":
        title = "Phase 2B.0 - Preparation"
    elif stage == "prompt":
        title = f"Phase 2B.1 - Prompt {options['prompt_id'].upper()}"
    elif stage == "context":
        title = f"Phase 2B.2 - Context Depth {options['depth']}"
    elif stage == "finalist":
        title = f"Phase 2B.3 - Finalist {options['slot']}"
    else:
        title = "Phase 2B.4 - Held-Out Final"
    description = {
        "prepare": "Freeze all subsets and package the Phase 2A baseline inputs. No model API calls are made.",
        "prompt": "Run one prompt on the fixed 80-question screening set and judge the fixed 20-question subset.",
        "context": "Run both selected prompt finalists at one context depth on the fixed 80/20 subsets. Depth 5 is reused from Phase 2B.1.",
        "finalist": "Run one configured finalist on all 281 development questions. Execute the two finalist notebooks independently.",
        "heldout": "Run the locked winner once on 284 questions from 50 seeded unseen articles. Results must not change the winner.",
    }[stage]
    input_description = (
        "Phase 2B.0 downloads the private Phase 2A baseline by exact HF commit "
        "and SHA-256. It also downloads and verifies the separate locked corpus artifact."
        if stage == "prepare"
        else "The Phase 2B.0 bundle supplies the frozen baseline traces and subset manifest. "
        "The locked corpus artifact is downloaded independently and verified by SHA-256."
    )
    cells = [
        _markdown("title", f"# {title} ({platform.title()})\n\n{description}\n"),
        _code("configuration", _configuration(stage, platform, options)),
        _markdown("environment", "## 1. Environment and immutable inputs\n\nSet only the configuration values at the top. Every run validates the locked artifact, preparation bundle, subset hashes, and repository commit.\n"),
        _code("setup", _setup(stage, platform)),
        _helpers(source),
        _markdown("inputs", f"## 2. Load validated inputs\n\n{input_description}\n"),
        _load_artifacts(source, stage),
        _cell(source, "runtime-profile"),
        _partitions(source, stage),
    ]
    if stage != "prepare":
        cells.extend([_cell(source, "runner"), _markdown("execution", "## 3. Execute this assigned stage\n\nThe output is resumable. Do not change configuration after successful records exist.\n"), _stage_cell(stage, options), _report(source)])
    cells.extend([_markdown("export-heading", "## 4. Export\n\nDownload or retain both the result bundle and checkpoint. Later stages must be configured from reviewed result artifacts, never by changing subset IDs.\n"), _export(stage, platform)])
    notebook = {key: copy.deepcopy(value) for key, value in source.items() if key != "cells"}
    notebook["cells"] = cells
    return notebook


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    for filename, stage, platform, options in VARIANTS:
        target = OUTPUT / filename
        target.write_text(
            json.dumps(build_notebook(source, stage, platform, options), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(target.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
