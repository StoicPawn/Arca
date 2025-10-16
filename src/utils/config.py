"""Utility helpers for loading and handling experiment configuration."""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    """Dataclass wrapper around nested configuration dictionaries."""

    data: Dict[str, Any]
    model: Dict[str, Any]
    eval: Dict[str, Any]
    logging: Dict[str, Any]
    seed: int = 42
    modalities: Optional[Dict[str, bool]] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "Config":
        return cls(**cfg)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config.from_dict(cfg)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arca configuration parser")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional KEY=VALUE pairs that override configuration values. "
            "Supports dotted notation, e.g. data.batch_size.vision=128"
        ),
    )
    return parser


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list[str]]) -> Dict[str, Any]:
    if not overrides:
        return cfg

    def _set_nested(target: Dict[str, Any], key: str, value: Any) -> None:
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", maxsplit=1)
        try:
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse override value '{value}'") from exc
        _set_nested(cfg, key, parsed_value)

    return cfg


def load_from_args(args: Optional[list[str]] = None) -> Config:
    parser = create_argparser()
    namespace = parser.parse_args(args=args)
    cfg = load_config(namespace.config)
    merged = apply_overrides(cfg.to_dict(), namespace.override)
    return Config.from_dict(merged)


__all__ = ["Config", "load_config", "load_from_args", "apply_overrides"]
