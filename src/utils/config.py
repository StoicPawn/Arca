from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Any, Dict

import yaml


@dataclasses.dataclass
class Config:
    """Container for the project configuration."""

    raw: Dict[str, Any]

    @property
    def data_root(self) -> pathlib.Path:
        root = self.raw.get("data", {}).get("root", "./data")
        return pathlib.Path(root).expanduser().resolve()

    @property
    def logging_dir(self) -> pathlib.Path:
        out = self.raw.get("logging", {}).get("output_dir", "./artifacts")
        return pathlib.Path(out).expanduser().resolve()


def load_config(path: str | pathlib.Path) -> Config:
    path = pathlib.Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(raw=data)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arca configuration loader")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser


__all__ = ["Config", "load_config", "build_arg_parser"]
