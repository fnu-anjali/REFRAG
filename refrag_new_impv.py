#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFRAG-style RAG (compress → sense/select → expand) — Updated reference implementation (paper-aligned RL)

This is a modified version of your single-file implementation with changes to better match
the paper's selective compression RL formulation:

Key changes (vs your original):
  1) Mixed-input CPT (before RL): expands exactly T' = pL chunks chosen UNIFORMLY AT RANDOM
     (instead of "longest chunks") so the decoder learns to handle token+chunk embeddings anywhere.

  2) Policy network g_theta: 2-layer Transformer over chunk embeddings producing per-chunk logits s_i.

  3) Action sampling: sequential selection of EXACTLY T' chunk indices without replacement
     (mask already-picked indices), reusing the same logits across steps.

  4) RL objective: GRPO-style grouped baseline + PPO-style clipped ratio objective.
     For each training example x: sample G action sequences l^(i), compute rewards r_i = -NLL(target),
     normalize within group to get advantages, optimize PPO clip objective.

Important notes:
  - This file is still a "reference implementation". Hyperparams and data plumbing are kept simple.
  - RL here uses CPT-like data: JSONL with fields {"tokens": "...", "split": {"s":..., "o":...}}.
    This matches the paper's definition of reward on xs+1:s+o.
"""
import os, sys, json, math, time, random, argparse, copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import time

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Transformers cache API (tuple past_key_values is deprecated in newer versions)
try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:
    Cache = None
    DynamicCache = None

try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_faiss():
    if faiss is None:
        raise RuntimeError("FAISS is not installed. Install with `pip install faiss-cpu` (or faiss-gpu).")


def safe_torch_load(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        allow_unsafe = os.environ.get("REFRAG_ALLOW_UNSAFE_TORCH_LOAD", "").strip().lower() in ("1","true","yes","y","on")
        if allow_unsafe:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def maybe_torch_compile(module: nn.Module, enabled: bool, mode: str = "reduce-overhead", fullgraph: bool = False):
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        print("[torch_compile] torch.compile not available; continuing without compile.")
        return module
    try:
        return torch.compile(module, mode=mode, fullgraph=fullgraph)
    except Exception as e:
        print(f"[torch_compile] compile failed; continuing without compile. error={repr(e)}")
        return module


# ----------------------------
# Retrieval (FAISS + encoder)
# ----------------------------

class PassageEncoder(nn.Module):
    """Passage encoder that returns a fixed vector per passage using CLS pooling."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device=None):
        super().__init__()
        self.device = device or now_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.out_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode_passages(self, texts: List[str], bs: int = 32) -> np.ndarray:
        self.encoder.eval()
        if not texts:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.encoder(**toks).last_hidden_state
            emb = out[:, 0, :]  # CLS
            emb = F.normalize(emb, dim=-1)
            vecs.append(emb.detach().cpu().float().numpy())
        return np.concatenate(vecs, axis=0)

    @torch.no_grad()
    def encode_query(self, text: str) -> np.ndarray:
        v = self.encode_passages([text], bs=1)
        return v[0] if len(v) else np.zeros((self.out_dim,), dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    ensure_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ≈ cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str):
    ensure_faiss()
    return faiss.read_index(index_path)


def search_index(index, query_vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    ensure_faiss()
    q = query_vec.astype(np.float32)[None, :]
    faiss.normalize_L2(q)
    D, I = index.search(q, topk)
    return D[0], I[0]


def parse_original(context_str):
    """Extract text after 'Original:' and before 'Entites:'"""
    if isinstance(context_str, dict):
        return context_str.get("Original", "")
    # It's a raw string like "Prefix: ...; Original: ... \nEntites: ..."
    if "Original:" in context_str:
        original = context_str.split("Original:")[1]
        if "\nEntites:" in original:
            original = original.split("\nEntites:")[0]
        return original.strip()
    return context_str.strip()
# ----------------------------
# REFRAG Core
# ----------------------------

@dataclass
class REFRAGConfig:
    encoder_name: str = "roberta-base"
    decoder_name: str = "meta-llama/Llama-3.2-3B"
    chunk_len_tokens: int = 64     # k
    max_q_tokens: int = 256
    max_ctx_tokens: int = 2048     # s
    max_out_tokens: int = 256      # o
    selective_p: float = 0.25      # p (uncompressed fraction)
    lr: float = 2e-5
    wd: float = 0.0
    grad_clip: float = 1.0
    fp16: bool = True
    seed: int = 1337
    torch_compile: bool = False

    # PPO/GRPO
    ppo_clip_eps: float = 0.2
    grpo_group_G: int = 4
    policy_lr: float = 1e-4
    policy_transformer_heads: int = 8
    policy_transformer_layers: int = 2
    policy_dropout: float = 0.0
    sync_old_every: int = 10


class ChunkEncoder(nn.Module):
    """Encoder that returns one vector per text chunk via CLS pooling."""
    def __init__(self, name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModel.from_pretrained(name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, texts: List[str], device=None) -> torch.Tensor:
        device = device or next(self.model.parameters()).device
        if len(texts) == 0:
            return torch.zeros((0, self.out_dim), device=device)
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = self.model(**toks).last_hidden_state[:, 0, :]
        h = F.normalize(h, dim=-1)
        return h


class TokenProjector(nn.Module):
    """Projection ϕ: encoder-dim → decoder token-embedding dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.proj(x)


class SelectPolicyTransformer(nn.Module):
    """2-layer transformer over chunk embeddings producing logits per chunk."""
    def __init__(self, d_model: int, nhead: int = 8, nlayers: int = 2, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = self.tr(c.unsqueeze(0))               # [1,L,D]
        s = self.head(h).squeeze(0).squeeze(-1)   # [L]
        return s

class SelectPolicy(nn.Module):
    """
    Tiny policy π(ci) that outputs expansion prob per chunk.
    Input: chunk embedding ci (encoder space) + scalar pos (normalized [0,1]).
    Output: logits ∈ R (Bernoulli).
    """
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, c: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c, pos01], dim=-1)
        return self.net(x).squeeze(-1)  # [L]

class REFRAG(nn.Module):
    def __init__(self, cfg: REFRAGConfig):
        super().__init__()
        self.cfg = cfg
        self.device = now_device()

        self.encoder = ChunkEncoder(cfg.encoder_name).to(self.device)
        self.decoder_tok = AutoTokenizer.from_pretrained(cfg.decoder_name, use_fast=True)
        self.decoder = AutoModelForCausalLM.from_pretrained(cfg.decoder_name).to(self.device)
        self.decoder = maybe_torch_compile(self.decoder, enabled=cfg.torch_compile)

        self.dec_embed_dim = self.decoder.get_input_embeddings().weight.shape[1]
        self.projector = TokenProjector(self.encoder.out_dim, self.dec_embed_dim).to(self.device)
        self.policy = SelectPolicyTransformer(
            self.encoder.out_dim,
            nhead=cfg.policy_transformer_heads,
            nlayers=cfg.policy_transformer_layers,
            dropout=cfg.policy_dropout,
        ).to(self.device)
        # self.policy = SelectPolicy(self.encoder.out_dim, hidden=cfg.policy_hidden).to(self.device)

        self.eos_id = self.decoder_tok.eos_token_id
        self.pad_id = self.decoder_tok.pad_token_id or self.decoder_tok.eos_token_id

    def _new_cache(self):
        if DynamicCache is None:
            return None
        return DynamicCache()

    def _ensure_cache(self, past_key_values):
        if Cache is None or DynamicCache is None:
            return past_key_values
        if past_key_values is None:
            return DynamicCache()
        if isinstance(past_key_values, Cache):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)

    def _tokenize(self, text: str, max_len: int) -> Dict[str, torch.Tensor]:
        return self.decoder_tok(text, truncation=True, max_length=max_len, padding=False, return_tensors="pt")

    @torch.no_grad()
    def _decoder_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings()(input_ids.to(self.device))

    def _chunk_text(self, text: str, k_tokens: int) -> Tuple[List[str], List[torch.Tensor]]:
        toks = self.decoder_tok(text, truncation=True, max_length=self.cfg.max_ctx_tokens, return_tensors="pt")
        ids = toks.input_ids[0]
        id_chunks = [ids[i:i+k_tokens] for i in range(0, ids.size(0), k_tokens)]
        str_chunks = [self.decoder_tok.decode(ch, skip_special_tokens=True) for ch in id_chunks]
        return str_chunks, id_chunks

    def _encode_chunks(self, chunk_strs: List[str]) -> torch.Tensor:
        return self.encoder(chunk_strs, device=self.device)

    def _project_chunks(self, c: torch.Tensor) -> torch.Tensor:
        return self.projector(c)

    def Tprime(self, L: int, p: float) -> int:
        if L <= 0 or p <= 0.0:
            return 0
        return max(1, int(round(p * L)))

    def picked_to_mask(self, L: int, picked: List[int]) -> torch.Tensor:
        m = torch.zeros(L, dtype=torch.bool, device=self.device)
        if picked:
            m[torch.tensor(picked, device=self.device)] = True
        return m

    def sample_action_sequence(self, logits: torch.Tensor, Tprime: int):
        L = int(logits.numel())
        picked: List[int] = []
        logps: List[torch.Tensor] = []

        # keep mask on same device as logits
        mask = torch.zeros(L, dtype=torch.bool, device=logits.device)

        for _ in range(Tprime):
            # non-inplace masking
            masked_logits = logits.masked_fill(mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)

            a = dist.sample()
            logps.append(dist.log_prob(a))

            ai = int(a.item())
            picked.append(ai)

            # non-inplace mask update
            mask = mask.clone()
            mask[ai] = True

        return picked, torch.stack(logps)

    @staticmethod
    def logprobs_for_picked(logits: torch.Tensor, picked: List[int]) -> torch.Tensor:
        L = int(logits.numel())
        mask = torch.zeros(L, dtype=torch.bool, device=logits.device)
        out: List[torch.Tensor] = []

        for ai in picked:
            masked_logits = logits.masked_fill(mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)

            a_t = torch.tensor(int(ai), device=logits.device)
            out.append(dist.log_prob(a_t))

            mask = mask.clone()
            mask[int(ai)] = True

        return torch.stack(out)

    # --- Heuristic unchanged (optional) ---
    def _heuristic_select(self, chunk_ids: List[torch.Tensor], q_text: str, p_max: float) -> torch.Tensor:
        L = len(chunk_ids)
        if L == 0 or p_max <= 0:
            return torch.zeros(L, dtype=torch.bool, device=self.device)
        scores = []
        with torch.no_grad():
            for ch in chunk_ids:
                inp = torch.cat([self._tokenize(q_text, self.cfg.max_q_tokens).input_ids[0].to(self.device), ch.to(self.device)], dim=0).unsqueeze(0)
                labels = inp.clone()
                out = self.decoder(input_ids=inp, labels=labels)
                ppl = torch.exp(out.loss).item()
                scores.append(ppl)
        scores = np.asarray(scores)
        k = max(1, int(round(p_max * L)))
        top_idx = scores.argsort()[::-1][:k].copy()
        mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        mask[top_idx] = True
        return mask

    def build_decoder_inputs_inst_prompt(self, question: str, passages: List[str], k: int, p: float,
                            use_policy: bool = True, sample_policy: bool = False) -> Tuple[torch.Tensor, Dict]:
        
        # Add instruction prompt wrapping the question
        instruction = (
            f"You are a helpful assistant. Answer the question using only the provided context.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n"
        )
        answer_prefix = "\n\nAnswer:"
        
        q_ids = self._tokenize(instruction, self.cfg.max_q_tokens).input_ids.to(self.device)
        q_emb = self._decoder_token_embeddings(q_ids)


        ctx_text = "".join(passages)
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        L = len(chunk_strs)

        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)
            ecnk = self._project_chunks(c)

        expand_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        picked: List[int] = []
        Tpr = self.Tprime(L, p)
        if use_policy and Tpr > 0 and L > 0:
            with torch.no_grad():
                logits = self.policy(c)
            if sample_policy:
                picked, _ = self.sample_action_sequence(logits, Tprime=Tpr)
            else:
                picked = torch.topk(logits, k=min(Tpr, L)).indices.detach().cpu().tolist()
            expand_mask = self.picked_to_mask(L, picked)
        elif (not use_policy) and L > 0 and p > 0:
            expand_mask = self._heuristic_select(chunk_ids, q_text=question, p_max=p)
            picked = torch.nonzero(expand_mask).squeeze(-1).detach().cpu().tolist()

        # After building seq_embs, append the answer prefix
        answer_ids = self._tokenize(answer_prefix, 16).input_ids.to(self.device)
        answer_emb = self._decoder_token_embeddings(answer_ids)
        
        seq_embs = [q_emb.squeeze(0)]
        seg_flags = []
        for i, ids in enumerate(chunk_ids):
            if expand_mask[i]:
                tok_emb = self._decoder_token_embeddings(ids.unsqueeze(0))
                seq_embs.append(tok_emb.squeeze(0))
                seg_flags.extend([1] * tok_emb.size(1))
            else:
                seq_embs.append(ecnk[i].unsqueeze(0))
                seg_flags.append(0)
        
        # Append "Answer:" suffix so model knows to generate the answer
        seq_embs.append(answer_emb.squeeze(0))
        seg_flags.extend([1] * answer_emb.size(1))

        final = torch.cat(seq_embs, dim=0).unsqueeze(0)
        extras = {"picked": picked, "expand_mask": expand_mask.detach().cpu().numpy().tolist(), "num_chunks": L, "Tprime": int(Tpr), "token_positions_flag": seg_flags}
        return final, extras

    def build_decoder_inputs(self, question: str, passages: List[str], k: int, p: float,
                            use_policy: bool = True, sample_policy: bool = False) -> Tuple[torch.Tensor, Dict]:
        q_ids = self._tokenize(question, self.cfg.max_q_tokens).input_ids.to(self.device)
        q_emb = self._decoder_token_embeddings(q_ids)

        ctx_text = "".join(passages)
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        L = len(chunk_strs)

        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)
            ecnk = self._project_chunks(c)

        expand_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        picked: List[int] = []
        Tpr = self.Tprime(L, p)
        if use_policy and Tpr > 0 and L > 0:
            with torch.no_grad():
                logits = self.policy(c)
            if sample_policy:
                picked, _ = self.sample_action_sequence(logits, Tprime=Tpr)
            else:
                picked = torch.topk(logits, k=min(Tpr, L)).indices.detach().cpu().tolist()
            expand_mask = self.picked_to_mask(L, picked)
        elif (not use_policy) and L > 0 and p > 0:
            expand_mask = self._heuristic_select(chunk_ids, q_text=question, p_max=p)
            picked = torch.nonzero(expand_mask).squeeze(-1).detach().cpu().tolist()

        seq_embs = [q_emb.squeeze(0)]
        seg_flags = []
        for i, ids in enumerate(chunk_ids):
            if expand_mask[i]:
                tok_emb = self._decoder_token_embeddings(ids.unsqueeze(0))
                seq_embs.append(tok_emb.squeeze(0))
                seg_flags.extend([1] * tok_emb.size(1))
            else:
                seq_embs.append(ecnk[i].unsqueeze(0))
                seg_flags.append(0)

        final = torch.cat(seq_embs, dim=0).unsqueeze(0)
        extras = {"picked": picked, "expand_mask": expand_mask.detach().cpu().numpy().tolist(), "num_chunks": L, "Tprime": int(Tpr), "token_positions_flag": seg_flags}
        return final, extras

    @torch.no_grad()
    def generate(self, question: str, passages: List[str], k: int, p: float,
                 max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0,
                 use_policy: bool = True, sample_policy: bool = False) -> Dict:
        self.decoder.eval()
        emb_in, extras = self.build_decoder_inputs_inst_prompt(question, passages, k=k, p=p, use_policy=use_policy, sample_policy=sample_policy)

        cache = self._new_cache()
        t0 = time.time()
        out = self.decoder(inputs_embeds=emb_in, use_cache=True, past_key_values=cache)
        past_key_values = self._ensure_cache(out.past_key_values)
        ttft = time.time() - t0

        generated = []
        ttit_list = []
        last = torch.tensor([[self.eos_id]], device=self.device)

        for _ in range(max_new_tokens):
            step_emb = self.decoder.get_input_embeddings()(last)
            t1 = time.time()
            out = self.decoder(inputs_embeds=step_emb, use_cache=True, past_key_values=past_key_values)
            ttit_list.append(time.time() - t1)

            logits = out.logits[:, -1, :]
            past_key_values = self._ensure_cache(out.past_key_values)

            if temperature > 0.0:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_id = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = sorted_indices.gather(-1, next_id)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            nid = next_id.item()
            if nid == self.eos_id:
                break
            generated.append(nid)
            last = next_id

        text = self.decoder_tok.decode(generated, skip_special_tokens=True)
        throughput = (len(generated) / max(sum(ttit_list), 1e-6)) if ttit_list else 0.0
        return {"answer": text.strip(), "TTFT_sec": ttft, "TTIT_avg_sec": float(np.mean(ttit_list)) if ttit_list else 0.0,
                "throughput_tok_per_sec": throughput, "meta": extras}

    # --- CPT losses ---
    def loss_reconstruction(self, ctx_text: str, current_step, k: int, num_chunks_cap: Optional[int] = None) -> torch.Tensor:
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        # print(f"chunk_strs: {chunk_strs}", len(chunk_strs))
        # print(f"chunk_ids: {chunk_ids}", len(chunk_ids))
        # print(f"num_chunks_cap: {num_chunks_cap}")
        if num_chunks_cap is not None:
            chunk_strs = chunk_strs[:num_chunks_cap]
            chunk_ids = chunk_ids[:num_chunks_cap]
        # print(f"chunk_strs: {chunk_strs}")
        # print(f"chunk_ids: {chunk_ids}")
        L = len(chunk_strs)
        if L == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)
            e = self._project_chunks(c)
        # Detach to prevent gradients
        c = c.detach()
        e = e.detach()

        # all chunk embeddings as prefix, reconstruct ALL tokens x1:L*k together
        prefix_emb = e.unsqueeze(0)  # (1, L, hidden)

        # all token ids concatenated: x1:L*k
        all_token_ids = torch.cat([ids.to(self.device) for ids in chunk_ids], dim=0)  # (L*k,)
        token_emb = self.decoder.get_input_embeddings()(all_token_ids.unsqueeze(0))   # (1, L*k, hidden)

        # full input: [e1, e2, ..., eL, token_1, token_2, ..., token_L*k]
        inp_emb = torch.cat([prefix_emb, token_emb], dim=1)  # (1, L+L*k, hidden)

        # labels: -100 for the L prefix positions, real ids for L*k token positions
        ignore = torch.full((1, L), -100, dtype=torch.long, device=self.device)
        labels = torch.cat([ignore, all_token_ids.unsqueeze(0)], dim=1)  # (1, L+L*k)

        out = self.decoder(inputs_embeds=inp_emb, labels=labels)

        if current_step % 100 == 0:
            predicted_ids = torch.argmax(out.logits, dim=-1)  # (1, L+L*k)
            # skip L prefix positions, drop last (shifted), decode middle
            predicted_text = self.decoder_tok.decode(predicted_ids[0][L:-1], skip_special_tokens=True)
            original_text = self.decoder_tok.decode(all_token_ids, skip_special_tokens=True)
            print(f"num_chunks_cap: {num_chunks_cap}")
            print(f"  loss     : {out.loss.item():.4f}")
            print(f"  original : {original_text}")
            print(f"  predicted: {predicted_text}")
            print(f"  prefix length  : {prefix_emb.shape[1]}")
            print(f"  tokens to recon: {token_emb.shape[1]}")
            print(f"  total inp_emb  : {inp_emb.shape[1]}")
            print(f"  logits shape   : {out.logits.shape}")

        return out.loss

    def loss_next_para(self, full_text: str, steps: int, s: int, o: int, k: int, expand_frac: float = 0.0) -> torch.Tensor:
        toks = self.decoder_tok(full_text, truncation=True, max_length=s + o, return_tensors="pt")
        ids = toks.input_ids[0].to(self.device)
        N = int(ids.numel())
        if N < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        min_tgt = max(1, int(min(o, 32)))
        ctx_len = min(int(s), N - min_tgt)
        ctx_len = max(1, int(ctx_len))
        tgt_len = min(int(o), N - ctx_len)
        if tgt_len <= 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        ctx_ids = ids[:ctx_len]
        out_ids = ids[ctx_len:ctx_len + tgt_len]

        ctx_str = self.decoder_tok.decode(ctx_ids, skip_special_tokens=True)
        chunk_strs, chunk_ids = self._chunk_text(ctx_str, k_tokens=k)

        # Use no_grad for encoding to save memory
        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)
            e = self._project_chunks(c)
        # Detach to prevent gradient tracking through encoder
        c = c.detach()
        e = e.detach()
        L = len(chunk_ids)

        # Paper-aligned: choose exactly T'=pL chunks uniformly at random to remain uncompressed
        expand_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        if L > 0 and expand_frac > 0.0:
            Tpr = self.Tprime(L, expand_frac)
            idx = torch.randperm(L, device=self.device)[:min(Tpr, L)]
            expand_mask[idx] = True

        seq = []
        for i, ids_i in enumerate(chunk_ids):
            if expand_mask[i]:
                tok_emb = self._decoder_token_embeddings(ids_i.unsqueeze(0)).squeeze(0)
                seq.append(tok_emb.detach())
            else:
                seq.append(e[i].unsqueeze(0))
        if len(seq) == 0:
            tok_emb = self._decoder_token_embeddings(ctx_ids.unsqueeze(0)).squeeze(0)
            seq.append(tok_emb.detach())

        ctx_emb = torch.cat(seq, dim=0).unsqueeze(0)
        tgt_ids = out_ids.unsqueeze(0)
        tgt_emb = self._decoder_token_embeddings(tgt_ids)

        inputs = torch.cat([ctx_emb, tgt_emb], dim=1)
        labels = torch.full((1, inputs.size(1)), -100, dtype=torch.long, device=self.device)
        labels[0, ctx_emb.size(1):] = out_ids

        out = self.decoder(inputs_embeds=inputs, labels=labels)

        if steps > 0 and (steps % 100 == 0):
            with torch.no_grad():
                # Ground truth target text
                gt_text = self.decoder_tok.decode(out_ids.detach().cpu(), skip_special_tokens=True)

                # Predicted token ids for ONLY the target positions
                T_ctx = int(ctx_emb.size(1))
                T_tgt = int(tgt_ids.size(1))

                # Use detach and move to CPU immediately to free GPU memory
                pred_ids = out.logits[0, T_ctx:T_ctx + T_tgt].detach().cpu().argmax(dim=-1)
                pred_text = self.decoder_tok.decode(pred_ids, skip_special_tokens=True)

                print("\n====== DEBUG loss_next_para ======")
                print("GT target:\n", gt_text)
                print("Pred target (argmax):\n", pred_text)
                print("=================================\n")

                # Explicitly delete to free memory
                del pred_ids, gt_text, pred_text

        return out.loss

    @torch.no_grad()
    def reward_neg_nll(self, full_text: str, s: int, o: int, k: int, picked: List[int]) -> torch.Tensor:
        toks = self.decoder_tok(full_text, truncation=True, max_length=s + o, return_tensors="pt")
        ids = toks.input_ids[0].to(self.device)
        N = int(ids.numel())
        if N < 2:
            return torch.tensor(0.0, device=self.device)

        s_eff = min(int(s), N - 1)
        o_eff = min(int(o), N - s_eff)
        if o_eff <= 0:
            return torch.tensor(0.0, device=self.device)

        ctx_ids = ids[:s_eff]
        tgt_ids = ids[s_eff:s_eff + o_eff]

        ctx_str = self.decoder_tok.decode(ctx_ids, skip_special_tokens=True)
        chunk_strs, chunk_ids = self._chunk_text(ctx_str, k_tokens=k)
        L = len(chunk_ids)

        c = self._encode_chunks(chunk_strs)
        e = self._project_chunks(c)
        expand_mask = self.picked_to_mask(L, picked)

        seq = []
        for i, ids_i in enumerate(chunk_ids):
            if expand_mask[i]:
                seq.append(self._decoder_token_embeddings(ids_i.unsqueeze(0)).squeeze(0))
            else:
                seq.append(e[i].unsqueeze(0))
        if len(seq) == 0:
            seq.append(self._decoder_token_embeddings(ctx_ids.unsqueeze(0)).squeeze(0))

        ctx_emb = torch.cat(seq, dim=0).unsqueeze(0)
        tgt = tgt_ids.unsqueeze(0)
        tgt_emb = self._decoder_token_embeddings(tgt)

        inputs = torch.cat([ctx_emb, tgt_emb], dim=1)
        labels = torch.full((1, inputs.size(1)), -100, dtype=torch.long, device=self.device)
        labels[0, ctx_emb.size(1):] = tgt_ids

        out = self.decoder(inputs_embeds=inputs, labels=labels)
        return -out.loss.detach()


# ----------------------------
# Optim / Training helpers
# ----------------------------

def setup_optim(params, lr, wd, total_steps):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)
    return opt, sch


# ----------------------------
# Index build / load helpers
# ----------------------------

def cmd_index(args):
    seed_everything()
    enc = PassageEncoder(args.embed_model)
    with open(args.corpus, "r", encoding="utf-8") as f:
        passages = [ln.strip() for ln in f if ln.strip()]
    embs = enc.encode_passages(passages, bs=64)
    os.makedirs(args.index_dir, exist_ok=True)
    np.save(os.path.join(args.index_dir, "texts.npy"), np.array(passages, dtype=object))
    build_faiss_index(embs, os.path.join(args.index_dir, "faiss.index"))
    print(f"[index] built with {len(passages)} passages → {args.index_dir}")


def load_index_bundle(index_dir: str):
    texts = np.load(os.path.join(index_dir, "texts.npy"), allow_pickle=True).tolist()
    index = load_faiss_index(os.path.join(index_dir, "faiss.index"))
    return texts, index


def curriculum_schedule(total_steps: int, max_chunks: int):
    return [1 + int((max_chunks - 1) * (t / max(1, total_steps - 1))) for t in range(total_steps)]


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def cmd_cpt_recon(args):
    seed_everything()
    cfg = REFRAGConfig(encoder_name=args.enc, decoder_name=args.dec, chunk_len_tokens=args.k, lr=args.lr, fp16=False, torch_compile=args.torch_compile)
    model = REFRAG(cfg).to(now_device())
    for p in model.decoder.parameters():
        p.requires_grad = False
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=args.steps)
    data = list(load_jsonl(args.train_json))
    if not data:
        print("[cpt_recon] no data.")
        return
    model.train()
    plan_cache = None

    for step in range(args.steps):
        ex = random.choice(data)
        text = ex["tokens"]
        chunk_strs, _ = model._chunk_text(text, k_tokens=cfg.chunk_len_tokens)
        max_chunks = max(1, len(chunk_strs))
        # print(f"max_chunks: {max_chunks}")
        plan_cache = plan_cache or curriculum_schedule(args.steps, max_chunks)
        # print(f"plan_cache: {plan_cache}")
        cap = plan_cache[step]
        # print(f"cap : {cap}")
        loss = model.loss_reconstruction(text,current_step=step, k=cfg.chunk_len_tokens, num_chunks_cap=cap)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sch.step()
        if step % max(1, args.log_every) == 0:
            print(f"[cpt_recon] step {step}/{args.steps} loss={loss.item():.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), os.path.join(args.out_dir, "encoder.pt"))
    torch.save(model.projector.state_dict(), os.path.join(args.out_dir, "projector.pt"))
    print(f"[cpt_recon] saved to {args.out_dir}")


def cmd_cpt_next(args):
    seed_everything()
    cfg = REFRAGConfig(encoder_name=args.enc, decoder_name=args.dec, chunk_len_tokens=args.k, lr=args.lr, fp16=True, torch_compile=args.torch_compile)
    model = REFRAG(cfg).to(now_device())
    if args.load_dir:
        enc_p = os.path.join(args.load_dir, "encoder.pt")
        proj_p = os.path.join(args.load_dir, "projector.pt")
        if os.path.exists(enc_p):
            model.encoder.load_state_dict(safe_torch_load(enc_p, map_location=now_device()))
        if os.path.exists(proj_p):
            model.projector.load_state_dict(safe_torch_load(proj_p, map_location=now_device()))
        print("[cpt_next] loaded encoder/projector init.")
    params = list(model.parameters())
    opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=args.steps)
    data = list(load_jsonl(args.train_json))
    if not data:
        print("[cpt_next] no data.")
        return
    model.train()

    # CHECKPOINT_STEPS = {500}
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    accumulated_loss = 0.0

    for step in range(args.steps):
        # Clear cache at start of accumulation cycle
        if step % gradient_accumulation_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ex = random.choice(data)
        text = ex["tokens"]
        s = ex.get("split", {}).get("s", getattr(args, 'max_ctx_len', 2048))
        o = ex.get("split", {}).get("o", 256)
        loss = model.loss_next_para(text, steps=step, s=s, o=o, k=cfg.chunk_len_tokens, expand_frac=args.expand_frac)

        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps

        # Store loss value before backward to avoid holding graph
        loss_val = loss.item()
        accumulated_loss += loss_val

        # Backward without zeroing gradients
        loss.backward()

        # Explicitly delete loss to free computation graph
        del loss

        # Only step optimizer after accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sch.step()
            opt.zero_grad()

            if step % max(1, args.log_every) == 0:
                print(f"[cpt_next] step {step}/{args.steps} loss={accumulated_loss:.4f}, time={time.strftime('%Y-%m-%d %H:%M:%S')}")

            accumulated_loss = 0.0

        # pbar.set_postfix(loss=f"{loss.item():.4f}")

        # # Optional: still print occasionally (won't break tqdm)
        # if step % max(1, args.log_every) == 0:
        #     tqdm.write(f"[cpt_next] step {step}/{args.steps} loss={loss.item():.4f}")

        # # Save intermediate checkpoints
        # if step in CHECKPOINT_STEPS:
        #     # Clear cache before checkpoint to reduce memory pressure
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     ckpt_dir = os.path.join(args.out_dir, f"checkpoint_step{step}")
        #     os.makedirs(ckpt_dir, exist_ok=True)
        #     torch.save(model.state_dict(), os.path.join(ckpt_dir, "refrag_full.pt"))
        #     print(f"[cpt_next] checkpoint saved at step {step} → {ckpt_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "refrag_full.pt"))
    print(f"[cpt_next] saved full model to {args.out_dir}")


def cmd_train_policy(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        fp16=False,
        torch_compile=args.torch_compile,
        policy_lr=args.policy_lr,
        ppo_clip_eps=args.ppo_clip_eps,
        grpo_group_G=args.group_G,
        sync_old_every=args.sync_old_every,
    )
    model = REFRAG(cfg).to(now_device())

    if args.load_dir:
        full_p = os.path.join(args.load_dir, "refrag_full.pt")
        pol_p = os.path.join(args.load_dir, "policy.pt")
        if os.path.exists(full_p):
            model.load_state_dict(safe_torch_load(full_p, map_location=now_device()), strict=False)
            print("[train_policy] loaded refrag_full.pt (strict=False).")
        if os.path.exists(pol_p):
            model.policy.load_state_dict(safe_torch_load(pol_p, map_location=now_device()))
            print("[train_policy] loaded policy.pt.")

    for p in model.decoder.parameters(): p.requires_grad = False
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.projector.parameters(): p.requires_grad = False
    for p in model.policy.parameters(): p.requires_grad = True

    policy_old = copy.deepcopy(model.policy).eval().to(model.device)
    for p in policy_old.parameters(): p.requires_grad = False

    opt = torch.optim.AdamW(model.policy.parameters(), lr=cfg.policy_lr)

    data = list(load_jsonl(args.cpt_json))
    if not data:
        print("[train_policy] no data.")
        return

    eps = cfg.ppo_clip_eps
    G = cfg.grpo_group_G

    model.train()
    torch.autograd.set_detect_anomaly(True)
    for step in range(args.steps):
        ex = random.choice(data)
        text = ex["tokens"]
        s = ex.get("split", {}).get("s", 2048)
        o = ex.get("split", {}).get("o", 256)

        toks = model.decoder_tok(text, truncation=True, max_length=int(s), return_tensors="pt")
        ctx_ids = toks.input_ids[0].to(model.device)
        if ctx_ids.numel() < 2:
            continue
        ctx_str = model.decoder_tok.decode(ctx_ids, skip_special_tokens=True)
        chunk_strs, chunk_ids = model._chunk_text(ctx_str, k_tokens=cfg.chunk_len_tokens)
        if len(chunk_ids) == 0:
            continue

        with torch.no_grad():
            c = model._encode_chunks(chunk_strs)
        L = int(c.size(0))
        Tpr = model.Tprime(L, args.p)
        if Tpr <= 0:
            continue

        logits = model.policy(c)
        with torch.no_grad():
            logits_old = policy_old(c)

        logps_list, logps_old_list, rewards_list = [], [], []
        for _ in range(G):
            picked, logps = model.sample_action_sequence(logits, Tprime=Tpr)
            with torch.no_grad():
                logps_old = REFRAG.logprobs_for_picked(logits_old, picked)
                r = model.reward_neg_nll(text, s=s, o=o, k=cfg.chunk_len_tokens, picked=picked)
            logps_list.append(logps)
            logps_old_list.append(logps_old)
            rewards_list.append(r)

        rewards = torch.stack(rewards_list)  # [G]
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        loss = 0.0
        for i in range(G):
            ratio = torch.exp(logps_list[i] - logps_old_list[i])
            unclipped = ratio * adv[i]
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv[i]
            loss = loss + (-torch.mean(torch.minimum(unclipped, clipped)))
        loss = loss / G

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.policy.parameters(), cfg.grad_clip)
        opt.step()

        if (step + 1) % max(1, cfg.sync_old_every) == 0:
            policy_old.load_state_dict(model.policy.state_dict())

        if step % max(1, args.log_every) == 0:
            print(f"[train_policy] step {step}/{args.steps} loss={loss.item():.4f} reward_mean={rewards.mean().item():.4f} T'={Tpr} L={L}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.policy.state_dict(), os.path.join(args.out_dir, "policy.pt"))
    print(f"[train_policy] saved policy to {args.out_dir}")


def cmd_generate(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        max_ctx_tokens=args.ctx_max,
        selective_p=args.p,
        fp16=False,
        torch_compile=args.torch_compile,
    )
    model = REFRAG(cfg)
    if args.load_dir:
        full_p = os.path.join(args.load_dir, "refrag_full.pt")
        enc_p = os.path.join(args.load_dir, "encoder.pt")
        proj_p = os.path.join(args.load_dir, "projector.pt")
        pol_p = os.path.join(args.load_dir, "policy.pt")
        if os.path.exists(full_p):
            model.load_state_dict(safe_torch_load(full_p, map_location=now_device()), strict=False)
            print("[generate] loaded refrag_full.pt (strict=False).")
        else:
            if os.path.exists(enc_p):
                model.encoder.load_state_dict(safe_torch_load(enc_p, map_location=now_device()))
            if os.path.exists(proj_p):
                model.projector.load_state_dict(safe_torch_load(proj_p, map_location=now_device()))
            if os.path.exists(pol_p):
                model.policy.load_state_dict(safe_torch_load(pol_p, map_location=now_device()))
            print("[generate] loaded available component weights.")

    # texts, index = load_index_bundle(args.index_dir)
    # qenc = PassageEncoder(args.embed_model)
    # qv = qenc.encode_query(args.question)
    # _, I = search_index(index, qv, args.topk)
    # passages = get_passages(args.question)
    # passages = [texts[i] for i in I]

    with open(args.input_file, "r") as f:
        raw = json.load(f)

    data = raw["data"]
    results = []
    total = len(data) * args.num_runs


    for item in data:
        question = item["question"]
        passages = [parse_original(entry) for entry in item["context"]]

        for run_idx in range(args.num_runs):

            out = model.generate(
                question=question,
                passages=passages,
                k=args.k,
                p=args.p,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                use_policy=(not args.heuristic),
                sample_policy=args.sample_policy,
            )


            results.append({
                "question": question,
                "run": run_idx + 1,
                "passages": passages,
                "Context": item["context"],
                "ground_truth": item["ground_truth"],
                **out,
            })


    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"[generate] saved {len(results)} results to {args.output_file}")

def build_argparser():
    p = argparse.ArgumentParser(description="REFRAG-style RAG (paper-aligned selective compression RL)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("index", help="Build FAISS index from corpus")
    sp.add_argument("--corpus", type=str, required=True)
    sp.add_argument("--index_dir", type=str, required=True)
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("cpt_recon", help="CPT-A: reconstruction curriculum")
    sp.add_argument("--train_json", type=str, required=True)
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--out_dir", type=str, default="runs/cpt_recon")
    sp.add_argument("--torch_compile", action="store_true")
    sp.set_defaults(func=cmd_cpt_recon)

    sp = sub.add_parser("cpt_next", help="CPT-B: next-paragraph prediction (random mixed inputs)")
    sp.add_argument("--train_json", type=str, required=True)
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--expand_frac", type=float, default=0.25, help="Uncompressed fraction p during CPT-B")
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--load_dir", type=str, default="")
    sp.add_argument("--out_dir", type=str, default="runs/cpt_next")
    sp.add_argument("--torch_compile", action="store_true")
    sp.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    sp.add_argument("--max_ctx_len", type=int, default=2048, help="Max context length (reduce to save memory)")
    sp.set_defaults(func=cmd_cpt_next)

    sp = sub.add_parser("train_policy", help="Train policy with GRPO+PPO (paper-aligned)")
    sp.add_argument("--cpt_json", type=str, required=True)
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=2000)
    sp.add_argument("--policy_lr", type=float, default=1e-4)
    sp.add_argument("--p", type=float, default=0.25)
    sp.add_argument("--group_G", type=int, default=4)
    sp.add_argument("--ppo_clip_eps", type=float, default=0.2)
    sp.add_argument("--sync_old_every", type=int, default=10)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--load_dir", type=str, default="")
    sp.add_argument("--out_dir", type=str, default="runs/policy")
    sp.add_argument("--torch_compile", action="store_true")
    sp.set_defaults(func=cmd_train_policy)

    sp = sub.add_parser("generate", help="RAG generate with compression/expansion")
    # sp.add_argument("--index_dir", type=str, required=True)
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    # sp.add_argument("--question", type=str, required=True)
    sp.add_argument("--topk", type=int, default=8)
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--p", type=float, default=0.25)
    sp.add_argument("--ctx_max", type=int, default=2048)
    sp.add_argument("--max_new", type=int, default=256)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--top_p", type=float, default=1.0)
    sp.add_argument("--heuristic", action="store_true")
    sp.add_argument("--sample_policy", action="store_true")
    sp.add_argument("--load_dir", type=str, default="")
    sp.add_argument("--torch_compile", action="store_true")
    sp.add_argument("--input_file", type=str, required=True)
    sp.add_argument("--output_file", type=str, required=True)
    sp.add_argument("--num_runs", type=int, default=1)
    sp.set_defaults(func=cmd_generate)

    return p


def main():
    p = build_argparser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
