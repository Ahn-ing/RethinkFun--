from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_model(model: nn.Module, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def _remap_state_dict_keys(
    state_dict: dict[str, Any],
    replacements: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """按字符串替换规则重写 key；value 原样保留。"""
    new_sd: dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k
        for old, new in replacements:
            nk = nk.replace(old, new)
        new_sd[nk] = v
    return new_sd


def load_model(
    model: nn.Module,
    load_path: str | Path,
    device: torch.device | str,
    *,
    strict: bool = True,
    key_replacements: Iterable[tuple[str, str]] | None = None,
) -> nn.Module:
    """
    加载模型权重；支持对 state_dict 的 key 做重命名以兼容重构。
    兼容两种保存格式：
    - 直接 torch.save(model.state_dict(), path)
    - torch.save({"state_dict": model.state_dict(), ...}, path)
    """
    ckpt = torch.load(str(load_path), map_location=device)
    state_dict: dict[str, Any] = ckpt.get("state_dict", ckpt)

    # 自动兼容：rpe -> ape（按你当前的报错定制）
    auto_replacements: list[tuple[str, str]] = []
    if any(k.startswith("encoder.rpe.") for k in state_dict) and not any(
        k.startswith("encoder.ape.") for k in state_dict
    ):
        auto_replacements.append(("encoder.rpe.", "encoder.ape."))
    if any(k.startswith("decoder.rpe.") for k in state_dict) and not any(
        k.startswith("decoder.ape.") for k in state_dict
    ):
        auto_replacements.append(("decoder.rpe.", "decoder.ape."))

    replacements: list[tuple[str, str]] = []
    replacements.extend(auto_replacements)
    if key_replacements:
        replacements.extend(list(key_replacements))

    if replacements:
        state_dict = _remap_state_dict_keys(state_dict, replacements)

    incompatible = model.load_state_dict(state_dict, strict=strict)

    # strict=False 时打印差异，便于排查
    if not strict:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            print("[load_model] missing_keys:", missing)
            print("[load_model] unexpected_keys:", unexpected)

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    best_loss: float,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "epoch": epoch,
        "best_loss": best_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
    }
    torch.save(obj, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    load_path: str | Path,
    device: torch.device,
) -> tuple[int, float | None]:
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {load_path}")

    obj = torch.load(load_path, map_location=device)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])

    start_epoch = obj.get("epoch", -1) + 1
    best_loss = obj.get("best_loss", None)
    return start_epoch, best_loss