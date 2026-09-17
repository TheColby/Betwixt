"""Compute device selection.

Renders may be long; that is acceptable. What is not acceptable is silently
falling back to CPU on a machine that has a GPU, so selection is reported.
"""

from __future__ import annotations


def select_device(requested: str = "auto") -> str:
    """Resolve a device string, preferring CUDA, then MPS, then CPU."""
    try:
        import torch
    except ImportError:
        if requested not in ("auto", "cpu"):
            raise RuntimeError(
                f"device '{requested}' requested but PyTorch is not installed")
        return "cpu"

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe(device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu (PyTorch not installed)"
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        name = torch.cuda.get_device_name(idx)
        gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"{device} ({name}, {gb:.1f} GB)"
    if device == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
