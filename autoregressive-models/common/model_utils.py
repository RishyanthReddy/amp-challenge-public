"""
Loading and sampling helpers for the ProGen2-small baseline (hugohrban/progen2-small on
HuggingFace — an unofficial but weight-identical mirror of Salesforce's ProGen2).

ProGen2's tokenizer uses "1" as a start-of-sequence marker meaning "generate N-to-C" and
"2" as the corresponding end token. Confirmed against 02_tokenizer_test.py output.

Incorporates three suggestions from code review (Rishi, Week 1):
  1. KV caching (past_key_values / use_cache=True) — only the new token is fed to the
     model each step instead of the whole sequence so far, which is what makes
     autoregressive sampling fast at scale.
  2. Min-length guard — the end token's logit is masked out until the sequence has
     reached MIN_LENGTH residues, so the model can never emit an invalid <8-residue
     candidate.
  3. Standard-amino-acid-only masking — every non-standard token (B, X, Z, U, O, and any
     other special/non-standard-residue tokens in the vocab) is masked out during
     sampling, so generation can't produce an automatically-invalid sequence.

KV-cache compatibility was smoke-tested on 2026-09-27 with one generated sequence using the
pinned custom model-code revision and the best-validation checkpoint. This confirms the
`past_key_values` interface works for that checkpoint; it does not establish large-run
throughput or memory limits.
"""

import torch
import math
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

from .validity import STANDARD_AMINO_ACIDS, MIN_LENGTH

MODEL_NAME = "hugohrban/progen2-small"
# Pin model weights, tokenizer, and custom ProGen implementation to one immutable
# Hugging Face Hub revision. trust_remote_code executes this repository's Python files.
MODEL_REVISION = "43237a0b733c6629226a079266d2985c9fdce9b7"
START_TOKEN = "1"
END_TOKEN = "2"


def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    device: str | None = None,
    torch_dtype=None,
):
    """
    torch_dtype: pass torch.float16 to load the model in half precision, roughly halving
    memory used by weights, gradients, and optimizer state — useful for fine-tuning
    larger checkpoints (e.g. progen2-medium) on memory-limited GPUs like a free-tier
    Colab T4. Defaults to fp32 (None) to match prior behavior for the baseline model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=MODEL_REVISION,
        code_revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_pretrained(model_name, revision=MODEL_REVISION)
    tokenizer.no_padding()

    return model, tokenizer, device


def _standard_aa_token_ids(tokenizer) -> list[int]:
    """Token IDs for exactly the 20 standard amino acids, per this tokenizer's vocab."""
    vocab = tokenizer.get_vocab()
    ids = [vocab[aa] for aa in STANDARD_AMINO_ACIDS if aa in vocab]
    if len(ids) != len(STANDARD_AMINO_ACIDS):
        missing = STANDARD_AMINO_ACIDS - set(vocab.keys())
        print(f"WARNING: {len(missing)} standard amino acids not found as single tokens "
              f"in the vocab: {missing}. Check the tokenizer's vocab layout.")
    return ids


def generate_sequence(
    model,
    tokenizer,
    device: str,
    max_len: int = 50,
    min_len: int = MIN_LENGTH,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int | None = None,
    max_perplexity: float | None = None,
) -> str:
    """
    Autoregressive sampling, one amino acid at a time, with temperature + top-p
    (nucleus) sampling, KV caching, a minimum-length guard, and standard-amino-acid-only
    masking. Stops early once the end token is sampled (only possible after min_len
    residues). Strips start/end markers before returning.
    """
    if seed is not None:
        torch.manual_seed(seed)

    start_id = tokenizer.encode(START_TOKEN).ids
    end_id = tokenizer.encode(END_TOKEN).ids[0]
    standard_ids = _standard_aa_token_ids(tokenizer)

    input_ids = torch.tensor([start_id], device=device)
    generated = input_ids
    cur_input = input_ids
    past_key_values = None
    n_residues = 0

    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            logits = logits / max(temperature, 1e-5)

            allowed_ids = list(standard_ids)
            if n_residues >= min_len:
                allowed_ids.append(end_id)

            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits[:, allowed_ids] = logits[:, allowed_ids]

            probs = torch.softmax(masked_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_in_sorted)

            generated = torch.cat([generated, next_token], dim=1)
            cur_input = next_token  # KV cache means we only feed the new token from here

            if next_token.item() == end_id:
                break
            n_residues += 1

    token_ids = generated[0].tolist()
    decoded = tokenizer.decode(token_ids)
    decoded_clean = decoded.replace(START_TOKEN, "").replace(END_TOKEN, "").strip()
    if max_perplexity is not None and decoded_clean:
        ppl = compute_perplexity(model, tokenizer, decoded_clean, device)
        if ppl > max_perplexity:
            return ""
    return decoded_clean
def compute_perplexity(model, tokenizer, sequence: str, device: str) -> float:
    """Compute sequence-level perplexity under the causal language model."""
    clean_seq = START_TOKEN + sequence.strip().upper() + END_TOKEN
    token_ids = tokenizer.encode(clean_seq).ids
    if len(token_ids) < 2:
        return float("inf")

    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return float(math.exp(loss))

def screen_by_perplexity(model, tokenizer, sequences: list[str], max_perplexity: float, device: str, filter_only: bool = True):
    """Filter candidate sequences by maximum perplexity threshold."""
    passing = []
    records = []
    for seq in sequences:
        ppl = compute_perplexity(model, tokenizer, seq, device)
        passes = ppl <= max_perplexity
        if passes:
            passing.append(seq)
        records.append({
            "sequence": seq,
            "perplexity": round(ppl, 2),
            "passes_perplexity_filter": passes,
        })
    return passing if filter_only else records
