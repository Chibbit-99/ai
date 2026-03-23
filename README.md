# Tiny Python LLM

This repository now contains a **working, minimal LLM** implementation in Python using PyTorch.

## File

- `tiny_llm.py`: character-level GPT-style transformer with:
  - causal self-attention
  - multi-head attention
  - transformer blocks
  - training loop
  - checkpoint save/load
  - text generation

## Quick start

```bash
python tiny_llm.py train --steps 300 --save tiny_llm.pt --prompt "In " --tokens 120
python tiny_llm.py generate --checkpoint tiny_llm.pt --prompt "Once " --tokens 120
```

You can train on your own UTF-8 text file:

```bash
python tiny_llm.py train --text your_corpus.txt --steps 1000 --save tiny_llm.pt
```

> Note: This is a tiny educational model, not a production-grade frontier model.
