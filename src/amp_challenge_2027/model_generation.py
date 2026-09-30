"""Regenerate the submitted library from released models with fixed schedules.

Default execution requires Linux, NVIDIA RTX 4090 and a CUDA-compatible driver.
Every invocation samples again, scores candidates, selects the portfolio and
checks both final hashes before installing any submission file. Saved ranked
candidate tables and saved library FASTAs are never pipeline inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    'library.fasta': '65183eef34ef86924f311c47de9b9183dabad0634da8586d155c01d81944afba',
    'top.fasta': '7181777317b8a4c7c29841471ae9941dfb0de83b1a0c53c5f0a6bf6d0991e480',
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(work: Path, project: str, script: str, *args: str, python: str = '3.12') -> None:
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    env.update({'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'PYTHONHASHSEED':'42', 'PYTHONUNBUFFERED':'1', 'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    environment_name = 'root' if project == '.' else project.replace('/','_')
    env['UV_PROJECT_ENVIRONMENT'] = str(ROOT/'.generation-envs'/environment_name)
    cmd = ['uv','run','--project',str(work/project),'--locked','--python',python,'python',str(work/script),*map(str,args)]
    print(f'Running {script}',flush=True)
    subprocess.run(cmd, cwd=work, env=env, check=True)


def copy_inputs(work: Path) -> None:
    # Explicit allowlist: no original library, ranked table, generated AR/evolution
    # pool, or historical master-scoring table is copied into this workspace.
    directories = ['src','generation','data-engineering/src','evolutionary-search/src','evolutionary-search/scripts','evolutionary-search/data','shared-evaluator/src','shared-evaluator/scripts','shared-evaluator/models','portfolio-selection/src','portfolio-selection/scripts','diffusion-models/apex']
    for relative in directories:
        shutil.copytree(ROOT/relative,work/relative,ignore=shutil.ignore_patterns('.venv','__pycache__','*.pyc'))
    files = ['pyproject.toml','uv.lock','README.md','LICENSE','cloud/fetch_progen_checkpoint.py','scripts/verify_submission.py','scripts/verify_evaluator_assets.py','docs/original_auxiliary_schedule.json','shared-evaluator/reports/training_data_summary.json','data-engineering/data/challenge/antibacterial.fasta']
    for project in ['shared-evaluator','portfolio-selection','evolutionary-search']:
        files.extend([f'{project}/pyproject.toml',f'{project}/uv.lock'])
    for relative in files:
        destination=work/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,destination)
    for relative in ['outputs','shared-evaluator/outputs','shared-evaluator/reports','evolutionary-search/reports','portfolio-selection/outputs']:
        (work/relative).mkdir(parents=True,exist_ok=True)


def generate(output: Path | None = None, *, check: bool = False) -> None:
    if sys.platform != 'linux' or shutil.which('nvidia-smi') is None:
        raise RuntimeError('Model generation requires Linux + NVIDIA RTX 4090 (CUDA 12.4 runtime). The CPU-only command is: uv run export_submission. No existing submission files have been changed.')
    # Driver check precedes downloads and workspace changes.
    run(ROOT,'generation','generation/inference.py','--check',python='3.10')
    if check:
        return
    started=time.perf_counter()
    run_dir=ROOT/'generation_runs'
    run_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='model_',dir=run_dir) as tmp:
        work=Path(tmp).resolve()
        copy_inputs(work)
        # An exact verified cache is allowed; the model runs anew every time.
        checkpoint=ROOT/'autoregressive-models/checkpoints/progen2_small_amp_best_val'
        if checkpoint.is_dir():
            shutil.copytree(checkpoint,work/'autoregressive-models/checkpoints/progen2_small_amp_best_val')
        run(work,'.','cloud/fetch_progen_checkpoint.py','--source','public')
        run(work,'.','scripts/verify_evaluator_assets.py')
        run(work,'generation','generation/inference.py',python='3.10')
        run(work,'portfolio-selection','evolutionary-search/scripts/run_production_search.py')
        run(work,'shared-evaluator','generation/audit_candidates.py')
        run(work,'shared-evaluator','shared-evaluator/scripts/run_master_evaluation.py')
        run(work,'portfolio-selection','portfolio-selection/scripts/run_submission_portfolio.py','--run-id','model_generation')
        challenger=work/'portfolio-selection/outputs/challenger_model_generation'
        aux=[work/f'autoregressive-models/outputs/progen_aux_production_20260928_{letter}_ppl100.csv' for letter in ['a','b']]
        run(work,'shared-evaluator','portfolio-selection/scripts/assemble_submission_library.py','--run-id','model_generation','--challenger-dir',challenger,'--aux-csv',aux[0],'--aux-csv',aux[1])
        from amp_challenge_2027.generate import main as export
        produced=challenger/'fasta'
        export(challenger/'top100.parquet',challenger/'full_50k_library.parquet',produced,work/'data-engineering/data/challenge/antibacterial.fasta')
        actual={name:sha(produced/name) for name in EXPECTED}
        if actual != EXPECTED:
            raise RuntimeError(f'Regenerated output differs from the submitted artifacts. Existing submission preserved. Expected {EXPECTED}; actual {actual}')
        run(work,'.','scripts/verify_submission.py',work,'--library-fasta',produced/'library.fasta','--top-fasta',produced/'top.fasta','--no-replay')
        targets=[output.resolve()] if output else [ROOT/'generate',ROOT/'generate_broad_spectrum']
        # All stages and hashes must pass before any output is replaced.
        for target in targets:
            target.mkdir(parents=True,exist_ok=True)
            for name in EXPECTED:
                staging=target/f'.{name}.model-generation.tmp'
                shutil.copyfile(produced/name,staging)
                os.replace(staging,target/name)
        evidence={'scope':'Fresh local model sampling, mutation, reference audit, forest/APEX inference, MAP-Elites/DPP and library assembly; no saved ranked/library tables read. Archived baseline comparison inputs are used only to preserve source exclusions.','seed':42,'elapsed_seconds':time.perf_counter()-started,'outputs_sha256':actual,'matches_submitted_artifacts':True,'model_generation':True}
        for label,rel in [('master','shared-evaluator/reports/master_evaluation_manifest.json'),('selection','portfolio-selection/outputs/challenger_model_generation/manifest.json'),('assembly','portfolio-selection/outputs/challenger_model_generation/assembly_manifest.json'),('evolution','evolutionary-search/reports/production_run_manifest.json')]:
            evidence[label]=json.loads((work/rel).read_text())
        (run_dir/'latest_generation.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print('Fresh model generation matches both submitted FASTAs byte-for-byte.',flush=True)


def cli() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=None,help='Write verified FASTAs here instead of the two canonical folders.')
    parser.add_argument('--check',action='store_true',help='Check the exact-generation GPU/runtime requirements only.')
    args=parser.parse_args()
    try:
        generate(args.output_dir,check=args.check)
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Generation failed: {error}\n")


if __name__ == '__main__':
    cli()
