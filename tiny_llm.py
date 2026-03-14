#!/usr/bin/env python3
"""A tiny, working character-level language model (GPT-style) in pure PyTorch.

Usage examples:
  python tiny_llm.py train --steps 600
  python tiny_llm.py generate --prompt "Once" --tokens 200
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT = """
In the beginning there was only silence.
Then came a spark, a sentence, and then another.
Words became stories, and stories became worlds.
A tiny model can still learn rhythm, memory, and surprise.
""".strip()


@dataclass
class Config:
    block_size: int = 64
    batch_size: int = 32
    n_embed: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    steps: int = 600
    eval_interval: int = 100
    eval_batches: int = 20


class Head(nn.Module):
    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            [Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embed, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(vocab_size, cfg.n_embed)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            *[Block(cfg.n_embed, cfg.n_heads, cfg.block_size, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        tok = self.token_embed(idx)
        pos = self.pos_embed(torch.arange(t, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class Trainer:
    def __init__(self, text: str, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.tokenizer = CharTokenizer(text)
        data = self.tokenizer.encode(text)
        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]

        self.model = TinyGPT(len(self.tokenizer.chars), cfg).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.cfg.block_size - 1, (self.cfg.batch_size,))
        x = torch.stack([data[i : i + self.cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.cfg.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        out: dict[str, float] = {}
        self.model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(self.cfg.eval_batches)
            for k in range(self.cfg.eval_batches):
                xb, yb = self.get_batch(split)
                _, loss = self.model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train(self) -> None:
        for step in range(self.cfg.steps):
            if step % self.cfg.eval_interval == 0 or step == self.cfg.steps - 1:
                losses = self.estimate_loss()
                print(
                    f"step {step:4d}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

            xb, yb = self.get_batch("train")
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def sample(self, prompt: str, tokens: int) -> str:
        if not prompt:
            prompt = " "
        for ch in prompt:
            if ch not in self.tokenizer.stoi:
                raise ValueError(
                    f"Prompt contains unseen character {ch!r}. "
                    f"Allowed chars: {''.join(self.tokenizer.chars)!r}"
                )
        idx = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)
        out = self.model.generate(idx, max_new_tokens=tokens)[0]
        return self.tokenizer.decode(out.cpu())


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT * 40
    content = Path(path).read_text(encoding="utf-8")
    if len(content) < 200:
        repeats = math.ceil(200 / max(1, len(content)))
        content = (content + "\n") * repeats
    return content


def save_checkpoint(path: Path, trainer: Trainer) -> None:
    payload = {
        "config": trainer.cfg.__dict__,
        "chars": trainer.tokenizer.chars,
        "state_dict": trainer.model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyGPT, CharTokenizer, Config]:
    payload = torch.load(path, map_location=device)
    cfg = Config(**payload["config"])
    tokenizer = CharTokenizer("".join(payload["chars"]))
    model = TinyGPT(len(tokenizer.chars), cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, tokenizer, cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and sample from a tiny GPT-style LLM.")
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Train a tiny model and optionally save checkpoint.")
    train.add_argument("--text", type=str, default=None, help="Path to UTF-8 training text.")
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--save", type=str, default="tiny_llm.pt")
    train.add_argument("--prompt", type=str, default="In ")
    train.add_argument("--tokens", type=int, default=200)

    generate = sub.add_parser("generate", help="Generate text from a saved checkpoint.")
    generate.add_argument("--checkpoint", type=str, default="tiny_llm.pt")
    generate.add_argument("--prompt", type=str, default="In ")
    generate.add_argument("--tokens", type=int, default=200)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        cfg = Config(steps=args.steps)
        text = load_text(args.text)
        trainer = Trainer(text, cfg, device)
        trainer.train()
        sample = trainer.sample(args.prompt, args.tokens)
        print("\n--- SAMPLE ---")
        print(sample)

        if args.save:
            save_path = Path(args.save)
            save_checkpoint(save_path, trainer)
            print(f"\nSaved checkpoint to {save_path}")

    elif args.mode == "generate":
        model, tokenizer, cfg = load_checkpoint(Path(args.checkpoint), device)
        for ch in args.prompt:
            if ch not in tokenizer.stoi:
                raise ValueError(f"Prompt has unseen char {ch!r}; charset is fixed by checkpoint.")

        idx = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.generate(idx, args.tokens)[0].cpu()
        print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
