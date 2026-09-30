"""Local ProGen2 inference using the exact submitted sampling schedules.

No Beam SDK, account, saved candidate table, or final library is used here.
The initial schedule follows cloud/sample_from_volume.py; auxiliary schedules
follow the archived production sampler. Numerical differences between those
samplers (CPU float32 vs CUDA BF16 cumulative sums) are intentionally retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

AA = "ACDEFGHIKLMNPQRSTVWY"
REVISION = "43237a0b733c6629226a079266d2985c9fdce9b7"
WEIGHTS_SHA = "124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83"
TOKENIZER_SHA = "cc489cd8bfeab3c70c6a2954d2963b9fb8e7ee4b13aaa0e84e97d7f17f35d43c"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_runtime() -> dict:
    import torch
    import transformers
    import tokenizers
    if torch.__version__.split('+')[0] != '2.5.1' or transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.3':
        raise RuntimeError('Use the locked generation environment (PyTorch 2.5.1, transformers 4.46.3, tokenizers 0.20.3).')
    if not torch.cuda.is_available():
        raise RuntimeError('Exact submitted generation requires an NVIDIA RTX 4090 with CUDA. Use export_submission for the CPU artifact export.')
    name = torch.cuda.get_device_name(0)
    if '4090' not in name or torch.version.cuda != '12.4' or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f'Exact replay requires RTX 4090, CUDA 12.4 and BF16; found {name}, CUDA {torch.version.cuda}.')
    return {'gpu': name, 'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__}


class ProGen:
    def __init__(self, checkpoint: Path):
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModelForCausalLM
        self.runtime = check_runtime()
        if sha(checkpoint / 'model.safetensors') != WEIGHTS_SHA or sha(checkpoint / 'tokenizer.json') != TOKENIZER_SHA:
            raise ValueError('Checkpoint/tokenizer hash differs from the submitted model.')
        torch.use_deterministic_algorithms(True, warn_only=True)
        warnings.filterwarnings('ignore', message='cumsum_cuda_kernel does not have a deterministic implementation')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(str(checkpoint), code_revision=REVISION, trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda').eval()
        self.tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
        self.tokenizer.no_padding()
        vocab = self.tokenizer.get_vocab()
        self.aa_ids = torch.tensor([vocab[aa] for aa in AA], device='cuda')
        self.start = self.tokenizer.encode('1').ids[0]
        self.end = self.tokenizer.encode('2').ids[0]

    def batch(self, count: int, seed: int, temperature: float, *, initial: bool) -> list[str]:
        t = self.torch
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        current = t.full((count, 1), self.start, dtype=t.long, device='cuda')
        finished = t.zeros(count, dtype=t.bool, device='cuda')
        finished_rows = [False] * count
        lengths = t.zeros(count, dtype=t.long, device='cuda')
        tokens = [[] for _ in range(count)]
        past = None
        with t.no_grad():
            for _ in range(50):
                result = self.model(current, past_key_values=past, use_cache=True)
                past = result.past_key_values
                logits = result.logits[:, -1, :] / temperature
                masked = t.full_like(logits, float('-inf'))
                masked[:, self.aa_ids] = logits[:, self.aa_ids]
                allow_end = lengths >= 8
                masked[allow_end, self.end] = logits[allow_end, self.end]
                probs = t.softmax(masked, dim=-1)
                sorted_probs, sorted_ids = t.sort(probs, descending=True, dim=-1)
                cumulative = (t.cumsum(sorted_probs.float().cpu(), dim=-1).to('cuda') if initial else t.cumsum(sorted_probs, dim=-1))
                cutoff = cumulative > 0.90
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_tokens = sorted_ids.gather(-1, t.multinomial(sorted_probs, num_samples=1))
                # Transfer the token vector once; per-residue .item() calls
                # otherwise synchronize CUDA thousands of times per batch.
                # Token sampling and all floating-point operations are unchanged.
                for row, token in enumerate(next_tokens[:, 0].tolist()):
                    if not finished_rows[row]:
                        if token == self.end:
                            finished_rows[row] = True
                        else:
                            tokens[row].append(token)
                active = ~finished
                ended = next_tokens[:, 0] == self.end
                lengths += (active & ~ended).to(dtype=t.long)
                finished |= active & ended
                if all(finished_rows):
                    break
                current = next_tokens
        return [self.tokenizer.decode(row).replace(' ', '').strip() for row in tokens]

    def perplexity(self, sequences: list[str]) -> list[float]:
        t = self.torch
        token_rows = [self.tokenizer.encode('1' + s + '2').ids for s in sequences]
        ids = t.full((len(sequences), max(map(len, token_rows))), self.end, dtype=t.long, device='cuda')
        mask = t.zeros_like(ids)
        for index, row in enumerate(token_rows):
            ids[index, :len(row)] = t.tensor(row, dtype=t.long, device='cuda')
            mask[index, :len(row)] = 1
        with t.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            target = ids[:, 1:]
            target_mask = mask[:, 1:].to(dtype=logits.dtype)
            losses = t.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction='none').reshape(target.shape)
            nll = (losses * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)
            return t.exp(nll.clamp(max=20)).detach().cpu().tolist()

    def initial(self, root: Path) -> None:
        import pandas as pd
        records = []
        for attempts, temperature in [(1225, 0.85), (1400, 1.00), (875, 1.15)]:
            tier_done = 0
            while tier_done < attempts:
                offset = len(records)
                count = min(64, attempts - tier_done)
                sequences = self.batch(count, 42 + offset, temperature, initial=True)
                ppls = self.perplexity(sequences)
                for j, (seq, ppl) in enumerate(zip(sequences, ppls)):
                    valid = 8 <= len(seq) <= 50 and set(seq) <= set(AA)
                    records.append({'run_id': f'ar_beam_{offset+j:05d}', 'sequence': seq, 'length': len(seq), 'temperature': temperature, 'top_p': 0.90, 'batch_seed': 42 + offset, 'sample_index_in_batch': j, 'perplexity': round(float(ppl), 4), 'passes_perplexity_filter': None, 'is_valid': valid, 'reason': 'ok' if valid else 'invalid_length_or_residue', 'model': 'progen2-small-epoch1-best'})
                tier_done += count
        ordered_sha = hashlib.sha256(('\n'.join(row['sequence'] for row in records)+'\n').encode('ascii')).hexdigest()
        if ordered_sha != '4654cbdce2e072d808e78b1feb09608c5bdba8e7756789554ea16ca24706e300':
            raise RuntimeError('Initial ordered model samples differ from the submitted schedule; canonical output is preserved.')
        out = root / 'outputs/ar_candidates.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out, index=False)
        print(f'Generated {len(records)} initial candidates from model weights', flush=True)

    def auxiliary(self, root: Path, *, run: str, attempts: int, retained: int, seed: int) -> None:
        folder = root / 'autoregressive-models/outputs'
        folder.mkdir(parents=True, exist_ok=True)
        run_id = f'production_20260928_{run}'
        raw = folder / f'progen_aux_{run_id}.csv'
        scored = folder / f'progen_aux_{run_id}_ppl100.csv'
        fields = ['attempt_index', 'sequence', 'length', 'temperature', 'top_p', 'batch_seed', 'batch_offset', 'is_valid']
        start = time.perf_counter()
        count_valid = 0
        unique = set()
        with raw.open('w', newline='', encoding='ascii') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for offset in range(0, attempts, 64):
                temperature = (0.85, 1.00, 1.15)[(offset // 64) % 3]
                sequences = self.batch(min(64, attempts-offset), seed + offset, temperature, initial=False)
                for j, sequence in enumerate(sequences):
                    if offset + j >= retained:
                        continue
                    valid = 8 <= len(sequence) <= 50 and set(sequence) <= set(AA)
                    count_valid += int(valid)
                    if valid:
                        unique.add(sequence)
                    writer.writerow({'attempt_index': offset+j+1, 'sequence': sequence, 'length': len(sequence), 'temperature': temperature, 'top_p': 0.90, 'batch_seed': seed+offset, 'batch_offset': j, 'is_valid': valid})
                if offset % 4096 == 0:
                    print(f'Auxiliary {run.upper()}: sampled {offset}/{attempts}', flush=True)
        elapsed = time.perf_counter() - start
        source_file = Path(__file__).relative_to(root).as_posix()
        raw_manifest = {'run_id': run_id, 'csv_sha256': sha(raw), 'checkpoint_sha256': WEIGHTS_SHA, 'model_code_revision': REVISION, 'tokenizer_sha256': TOKENIZER_SHA, 'n_attempts': retained, 'n_valid': count_valid, 'n_unique_valid': len(unique), 'requested_attempts': attempts, 'run_status': 'completed_schedule_with_historical_prefix' if retained < attempts else 'completed', 'runner_source_file': source_file, 'runner_source_sha256': sha(Path(__file__)), 'base_seed': seed, 'batch_size':64}
        raw_manifest_path = raw.with_suffix('.manifest.json')
        raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2)+'\n')
        start_ppl = time.perf_counter()
        max_ppl = 0.0
        passes = 0
        total = 0
        with raw.open(newline='', encoding='ascii') as f, scored.open('w', newline='', encoding='ascii') as g:
            reader = csv.DictReader(f)
            writer = csv.DictWriter(g, fieldnames=fields + ['perplexity','passes_perplexity_filter'])
            writer.writeheader()
            rows = []
            def flush():
                nonlocal max_ppl, passes, total
                scores = self.perplexity([r['sequence'] for r in rows])
                for row, ppl in zip(rows, scores):
                    passed = row['is_valid'] == 'True' and ppl <= 100
                    passes += int(passed)
                    total += 1
                    max_ppl = max(max_ppl, float(ppl))
                    writer.writerow({**row,'perplexity': f'{ppl:.6f}', 'passes_perplexity_filter': str(passed)})
                rows.clear()
            for row in reader:
                rows.append(row)
                if len(rows) == 64:
                    flush()
                    if total % 4096 == 0:
                        print(f'Auxiliary {run.upper()}: perplexity {total}/{retained}', flush=True)
            if rows:
                flush()
        ppl_elapsed = time.perf_counter() - start_ppl
        manifest = {**raw_manifest, 'csv_sha256': sha(scored), 'source_csv': raw.relative_to(root).as_posix(), 'raw_source_csv_sha256': sha(raw), 'raw_source_manifest_sha256': sha(raw_manifest_path), 'n_generation_attempts':retained, 'n_generation_valid':count_valid, 'generation_runner_source_sha256':sha(Path(__file__)), 'perplexity_runner_source_file':source_file, 'perplexity_runner_source_sha256':sha(Path(__file__)), 'max_perplexity':100.0, 'max_observed_perplexity':max_ppl, 'n_perplexity_pass':passes, 'perplexity_elapsed_sec':ppl_elapsed, 'perplexity_attempts_per_sec':retained/ppl_elapsed, 'gpu':self.runtime['gpu'], 'generation_gpu':self.runtime['gpu'], 'generation_attempts_per_sec':attempts/elapsed}
        scored.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        schedule = json.loads((root/'docs/original_auxiliary_schedule.json').read_text())['runs'][0 if run=='a' else 1]
        if sha(raw) != schedule['raw_sha256'] or sha(scored) != schedule['ppl_sha256']:
            raise RuntimeError(f'Auxiliary {run} differs from the historical schedule; canonical output will not be replaced.')
        print(f'Auxiliary {run.upper()}: exact raw/PPL hashes verified', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_runtime()))
        return
    root = args.root.resolve()
    model = ProGen(root/'autoregressive-models/checkpoints/progen2_small_amp_best_val')
    model.initial(root)
    model.auxiliary(root, run='a', attempts=60000, retained=51069, seed=200000)
    model.auxiliary(root, run='b', attempts=85000, retained=85000, seed=300000)


if __name__ == '__main__':
    main()
