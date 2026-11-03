"""Lazy PyTorch-only loading helpers for optional Hugging Face models."""

from __future__ import annotations

import os


os.environ.setdefault("USE_TF", "0")


def load_transformer_pipeline(task: str, model_name: str, **kwargs):
    """Load a Hugging Face pipeline only when an analysis explicitly requests it."""
    from transformers import pipeline

    return pipeline(task, model=model_name, **kwargs)


def load_transformer_encoder(model_name: str):
    """Load a tokenizer and encoder on the best locally available device."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()
    return tokenizer, encoder, device
