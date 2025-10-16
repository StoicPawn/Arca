"""Dataset download, preprocessing, and DataLoader utilities."""
from __future__ import annotations

import hashlib
import json
import numbers
import shutil
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchvision import datasets as tv_datasets
import torchvision.transforms as T

import torchaudio
from torchaudio import transforms as ta_transforms

from datasets import ClassLabel, load_dataset, load_from_disk

from src.utils.config import Config, load_from_args
from src.utils.seed import set_seed

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetPaths:
    root: Path
    raw: Path
    processed: Path

    @classmethod
    def from_root(cls, root: Path) -> "DatasetPaths":
        raw = root / "raw"
        processed = root / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        return cls(root=root, raw=raw, processed=processed)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _compute_digest(path: Path, algorithm: str) -> str:
    """Compute the checksum for ``path`` using ``algorithm``."""

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported checksum algorithm '{algorithm}'.") from exc

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_checksum_spec(
    spec: Union[str, Mapping[str, str], None]
) -> Tuple[str, str]:
    """Normalize checksum configuration into ``(digest, algorithm)`` tuple.

    ``spec`` can either be a raw digest string (md5/sha256) or a mapping with
    ``hash``/``value`` and ``algorithm`` fields. Empty or missing specs yield an
    empty digest and default algorithm (sha256).
    """

    if spec is None:
        return "", "sha256"

    if isinstance(spec, str):
        digest = spec.strip()
        if not digest:
            return "", "sha256"
        length = len(digest)
        if length == 32:
            return digest, "md5"
        if length == 64:
            return digest, "sha256"
        raise ValueError(
            "Checksum string must be a 32-character MD5 or 64-character SHA256 digest."
        )

    algorithm = spec.get("algorithm", "sha256").lower()
    digest = spec.get("value") or spec.get("hash") or ""
    digest = digest.strip()
    if not digest:
        return "", algorithm
    return digest, algorithm


def verify_checksum_file(path: Path, expected: str, algorithm: str = "sha256") -> bool:
    """Verify that ``path`` matches the expected checksum."""

    return path.is_file() and _compute_digest(path, algorithm) == expected


def _find_archive(base_path: Path, patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        match = next(base_path.rglob(pattern), None)
        if match is not None:
            return match
    return None


def _verify_downloaded_dataset(
    dataset_name: str,
    dataset_obj,
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    archive_patterns: List[str],
) -> None:
    """Verify dataset checksum with graceful fallback to internal integrity checks.

    Some torchvision/torchaudio datasets remove the original archive after
    extraction, which means we cannot always recompute the user-specified
    checksum. When this happens we fall back to the dataset's internal
    ``_check_integrity`` method (when available). If that check succeeds we keep
    the dataset while emitting a warning to make users aware of the mismatch.
    """

    digest, algorithm = _normalize_checksum_spec(checksum_spec)
    if not digest:
        return

    archive = _find_archive(paths.raw, archive_patterns)
    if archive is not None and verify_checksum_file(archive, digest, algorithm):
        return

    integrity_fn = getattr(dataset_obj, "_check_integrity", None)
    if callable(integrity_fn):
        try:
            if integrity_fn():
                warnings.warn(
                    (
                        f"Checksum verification failed for {dataset_name} archive, "
                        "but the dataset passed its internal integrity check. "
                        "Please update the configured checksum if the archive "
                        "was legitimately refreshed."
                    ),
                    RuntimeWarning,
                )
                return
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"Failed to run internal integrity check for {dataset_name}: {exc}",
                RuntimeWarning,
            )

    raise RuntimeError(f"Checksum verification failed for {dataset_name} dataset.")


def _load_torch_dataset(
    dataset_cls,
    dataset_name: str,
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    archive_patterns: List[str],
    **kwargs: Any,
):
    """Load a torch dataset (vision or audio) while reusing cached files."""

    needs_download = False

    try:
        dataset = dataset_cls(root=str(paths.raw), download=False, **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "download=true" not in message:
            raise
        needs_download = True
    else:
        try:
            _verify_downloaded_dataset(
                dataset_name, dataset, paths, checksum_spec, archive_patterns
            )
            return dataset
        except RuntimeError as verify_error:
            warnings.warn(
                (
                    f"Existing {dataset_name} files failed verification and will be "
                    f"downloaded again: {verify_error}"
                ),
                RuntimeWarning,
            )
            base_folder = getattr(dataset, "base_folder", None)
            root_path = Path(getattr(dataset, "root", paths.raw))
            target_path = root_path / base_folder if base_folder else root_path
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
            needs_download = True

    if not needs_download:
        raise RuntimeError(
            f"Failed to load torchvision dataset {dataset_name} without downloading."
        )

    dataset = dataset_cls(root=str(paths.raw), download=True, **kwargs)
    _verify_downloaded_dataset(dataset_name, dataset, paths, checksum_spec, archive_patterns)
    return dataset


# ---------------------------------------------------------------------------
# Vision preprocessing
# ---------------------------------------------------------------------------


def _prepare_stl10(
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    del dataset_cfg  # Unused: retained for API parity
    dataset = _load_torch_dataset(
        tv_datasets.STL10,
        "STL10",
        paths,
        checksum_spec,
        ["stl10*tar.gz", "stl10*zip"],
        split="unlabeled",
    )

    # STL10 ships with 96x96x3 numpy arrays
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0

    metadata: Dict[str, Any] = {
        "split": "unlabeled",
        "labels_available": False,
        "num_samples": images.shape[0],
    }
    if hasattr(dataset, "classes"):
        metadata["label_names"] = list(dataset.classes)

    return {"features": images}, metadata


def _prepare_cifar10(
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    cfg = dict(dataset_cfg or {})
    train_split = cfg.get("train", True)
    dataset = _load_torch_dataset(
        tv_datasets.CIFAR10,
        "CIFAR10",
        paths,
        checksum_spec,
        ["cifar-10-python.tar.gz", "cifar10*zip"],
        train=train_split,
    )

    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.tensor(dataset.targets, dtype=torch.long)

    metadata: Dict[str, Any] = {
        "split": "train" if train_split else "test",
        "labels_available": True,
        "num_samples": images.shape[0],
        "label_names": list(getattr(dataset, "classes", [])),
    }

    return {"features": images, "labels": labels}, metadata


def _prepare_dtd(
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    del dataset_cfg  # Unused: retained for API parity
    dataset = _load_torch_dataset(
        tv_datasets.DTD,
        "DTD",
        paths,
        checksum_spec,
        ["dtd*tar.gz", "dtd*zip"],
        split="train",
    )

    resize = T.Resize((96, 96))
    to_tensor = T.ToTensor()

    tensors = []
    labels: List[int] = []
    for img, target in tqdm(dataset, desc="Processing DTD", leave=False):
        tensors.append(resize(img))
        labels.append(int(target))
    data = torch.stack([to_tensor(x) for x in tensors])
    payload: Dict[str, torch.Tensor] = {"features": data, "labels": torch.tensor(labels, dtype=torch.long)}
    metadata: Dict[str, Any] = {
        "split": "train",
        "labels_available": True,
        "num_samples": data.shape[0],
        "label_names": list(getattr(dataset, "classes", [])),
    }
    return payload, metadata


VISION_DATASETS = {
    "CIFAR10": _prepare_cifar10,
    "STL10": _prepare_stl10,
    "DTD": _prepare_dtd,
}


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def _ensure_torchaudio_backend() -> None:
    """Select a torchaudio backend that does not rely on TorchCodec."""

    preferred_backends = ("soundfile", "sox_io")

    get_backend = getattr(torchaudio, "get_audio_backend", None)
    set_backend = getattr(torchaudio, "set_audio_backend", None)

    if set_backend is None:
        # Modern torchaudio versions might not expose backend selection.
        return

    current_backend: Optional[str] = None
    if callable(get_backend):
        try:
            current_backend = get_backend()
        except RuntimeError:
            current_backend = None

    if current_backend in preferred_backends:
        return

    for backend in preferred_backends:
        try:
            set_backend(backend)
            return
        except RuntimeError:
            continue

    warnings.warn(
        (
            "Falling back to torchaudio's default backend. "
            "Install 'torchcodec' or ensure one of the optional backends "
            "('soundfile' or 'sox_io') is available to avoid audio loading errors."
        ),
        RuntimeWarning,
    )


def _load_waveform_without_torchaudio(file_audio: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """Load a waveform using the standard library as a TorchCodec-free fallback."""

    with wave.open(str(file_audio), "rb") as handle:
        num_channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        num_frames = handle.getnframes()
        audio_bytes = handle.readframes(num_frames)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise RuntimeError(
            f"Unsupported sample width {sample_width} bytes in '{file_audio}'."
        )

    dtype = dtype_map[sample_width]
    waveform = np.frombuffer(audio_bytes, dtype=dtype)
    if num_channels > 1:
        waveform = waveform.reshape(-1, num_channels).T
    else:
        waveform = waveform.reshape(1, -1)

    if sample_width == 1:
        waveform = (waveform.astype(np.float32) - 128.0) / 128.0
    else:
        scale = float(2 ** (8 * sample_width - 1))
        waveform = waveform.astype(np.float32) / scale

    return torch.from_numpy(waveform), sample_rate


def _load_yesno_samples(dataset: torchaudio.datasets.YESNO) -> List[Tuple[torch.Tensor, int, List[int]]]:
    """Load YESNO samples, falling back to pure Python decoding if necessary."""

    samples: List[Tuple[torch.Tensor, int, List[int]]] = []

    try:
        for idx in range(len(dataset)):
            samples.append(dataset[idx])
        return samples
    except RuntimeError as exc:
        if "libtorchcodec" not in str(exc):
            raise

    warnings.warn(
        (
            "TorchCodec could not be loaded; falling back to Python's 'wave' module "
            "for decoding YESNO audio files. Audio loading will be slower but should "
            "remain functional."
        ),
        RuntimeWarning,
    )

    base_path = Path(dataset._path)  # noqa: SLF001 - accessing private attribute for fallback
    for fileid in dataset._walker:  # noqa: SLF001
        file_audio = base_path / f"{fileid}.wav"
        waveform, sample_rate = _load_waveform_without_torchaudio(file_audio)
        labels = [int(char) for char in fileid.split("_")]
        samples.append((waveform, sample_rate, labels))

    return samples


def _compute_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    target_rate: int = 16_000,
    n_mels: int = 96,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
) -> torch.Tensor:
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    win_length = int(target_rate * win_length_ms / 1000)
    hop_length = int(target_rate * hop_length_ms / 1000)
    mel = ta_transforms.MelSpectrogram(
        sample_rate=target_rate,
        n_fft=2048,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(waveform)
    log_mel = torch.log(mel + 1e-6)
    return log_mel


def _load_or_download_hf_dataset(
    paths: DatasetPaths,
    dataset_name: str,
    *,
    repo_id: str,
    split: str,
    config_name: Optional[str],
    revision: Optional[str],
    cache_subdir: Optional[str],
):
    hf_root = paths.raw / "huggingface"
    dataset_root = hf_root / (cache_subdir or repo_id.replace("/", "_"))
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_disk_path = dataset_root / "dataset"
    cache_dir = dataset_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_disk_path.exists():
        dataset = load_from_disk(str(dataset_disk_path))
    else:
        dataset = load_dataset(
            repo_id,
            name=config_name,
            split=split,
            cache_dir=str(cache_dir),
            revision=revision,
        )
        dataset.save_to_disk(str(dataset_disk_path))

    return dataset, dataset_root


def _prepare_hf_audio_dataset(
    paths: DatasetPaths,
    dataset_name: str,
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    dataset_cfg = dict(dataset_cfg or {})

    repo_id = dataset_cfg.get("repo_id")
    if not repo_id:
        raise ValueError(
            f"Hugging Face dataset configuration for {dataset_name} must include a 'repo_id'."
        )

    config_name = dataset_cfg.get("config") or dataset_cfg.get("config_name")
    split = dataset_cfg.get("split", "train")
    revision = dataset_cfg.get("revision")
    cache_subdir = dataset_cfg.get("cache_subdir")
    audio_column = dataset_cfg.get("audio_column", "audio")
    label_column = dataset_cfg.get("label_column")
    label_name_column = dataset_cfg.get("label_name_column")
    expected_fingerprint = dataset_cfg.get("fingerprint") or dataset_cfg.get(
        "expected_fingerprint"
    )

    dataset, dataset_root = _load_or_download_hf_dataset(
        paths,
        dataset_name,
        repo_id=repo_id,
        split=split,
        config_name=config_name,
        revision=revision,
        cache_subdir=cache_subdir,
    )

    fingerprint = dataset._fingerprint
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise RuntimeError(
            (
                f"Loaded Hugging Face dataset for {dataset_name} with fingerprint {fingerprint}, "
                f"which does not match the expected fingerprint {expected_fingerprint}."
            )
        )

    if revision is None and expected_fingerprint is None:
        warnings.warn(
            (
                f"No explicit revision or fingerprint configured for {dataset_name}. "
                "The dataset fingerprint has been recorded in the metadata file to aid "
                "future reproducibility."
            ),
            RuntimeWarning,
        )

    dataset_length = len(dataset)
    lengths_tensor = torch.empty(dataset_length, dtype=torch.long)
    raw_labels: List[Any] = []

    max_length = 0
    n_mels: Optional[int] = None

    for idx in tqdm(
        range(dataset_length), desc=f"Scanning {dataset_name}", leave=False
    ):
        sample = dataset[idx]
        if audio_column not in sample:
            raise KeyError(
                f"Column '{audio_column}' not found in Hugging Face dataset for {dataset_name}."
            )

        audio_sample = sample[audio_column]
        waveform = torch.tensor(audio_sample["array"], dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)

        sample_rate = int(audio_sample["sampling_rate"])
        log_mel = _compute_log_mel(waveform, sample_rate).squeeze(0)

        time_steps = log_mel.size(-1)
        lengths_tensor[idx] = time_steps
        max_length = max(max_length, time_steps)
        if n_mels is None:
            n_mels = log_mel.size(0)

        if label_column is not None:
            if label_column not in sample:
                raise KeyError(
                    f"Column '{label_column}' not found in Hugging Face dataset for {dataset_name}."
                )
            raw_labels.append(sample[label_column])

    if n_mels is None:
        raise RuntimeError(f"No audio samples found in Hugging Face dataset {dataset_name}.")

    features = torch.zeros((dataset_length, n_mels, max_length), dtype=torch.float32)

    for idx in tqdm(
        range(dataset_length), desc=f"Materializing {dataset_name}", leave=False
    ):
        sample = dataset[idx]
        audio_sample = sample[audio_column]
        waveform = torch.tensor(audio_sample["array"], dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)

        sample_rate = int(audio_sample["sampling_rate"])
        log_mel = _compute_log_mel(waveform, sample_rate).squeeze(0)

        time_steps = log_mel.size(-1)
        features[idx, :, :time_steps] = log_mel

    payload: Dict[str, torch.Tensor] = {
        "features": features,
        "lengths": lengths_tensor,
    }

    label_names: Optional[List[str]] = None
    label_mapping: Optional[Dict[str, int]] = None
    if label_column is not None:
        feature_info = dataset.features.get(label_column)

        if isinstance(feature_info, ClassLabel):
            label_names = list(feature_info.names)
            labels_tensor = torch.tensor([int(x) for x in raw_labels], dtype=torch.long)
        else:
            normalized: List[Any] = []
            for value in raw_labels:
                if isinstance(value, numbers.Integral):
                    normalized.append(int(value))
                else:
                    normalized.append(str(value))

            if all(isinstance(v, int) for v in normalized):
                labels_tensor = torch.tensor(normalized, dtype=torch.long)
            else:
                existing_mapping = dataset_cfg.get("label_mapping")
                if existing_mapping is not None:
                    label_mapping = {str(k): int(v) for k, v in existing_mapping.items()}
                else:
                    label_mapping = {}
                    for value in normalized:
                        key = str(value)
                        if key not in label_mapping:
                            label_mapping[key] = len(label_mapping)

                labels_tensor = torch.tensor(
                    [label_mapping[str(v)] for v in normalized], dtype=torch.long
                )

        payload["labels"] = labels_tensor

    metadata: Dict[str, Any] = {
        "repo_id": repo_id,
        "config_name": config_name,
        "split": split,
        "revision": revision,
        "fingerprint": fingerprint,
        "audio_column": audio_column,
        "num_samples": len(dataset),
        "cache_directory": str(dataset_root),
        "labels_available": label_column is not None,
    }

    if label_column is not None:
        metadata["label_column"] = label_column
    if label_names is not None:
        metadata["label_names"] = label_names
    if label_mapping is not None:
        metadata["label_mapping"] = label_mapping
    if expected_fingerprint is not None:
        metadata["expected_fingerprint"] = expected_fingerprint
    if (
        label_column is not None
        and label_name_column
        and label_name_column in dataset.column_names
    ):
        metadata["label_name_column"] = label_name_column
        mapping: Dict[str, Any] = {}
        for raw_label, label_name in zip(raw_labels, dataset[label_name_column]):
            key = str(raw_label)
            mapping.setdefault(key, label_name)
        metadata["label_name_mapping"] = mapping

    return payload, metadata


def _prepare_urbansound8k(
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    del checksum_spec  # Unused: retained for API compatibility
    cfg = dict(dataset_cfg or {})
    cfg.setdefault("repo_id", "urbansound8k")
    cfg.setdefault("split", "train")
    cfg.setdefault("audio_column", "audio")
    cfg.setdefault("label_column", "classID")
    cfg.setdefault("label_name_column", "class")
    return _prepare_hf_audio_dataset(paths, "UrbanSound8K", cfg)


def _prepare_esc50(
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    del checksum_spec  # Unused: retained for API compatibility
    cfg = dict(dataset_cfg or {})
    cfg.setdefault("repo_id", "ashraq/esc50")
    cfg.setdefault("split", "train")
    cfg.setdefault("audio_column", "audio")
    cfg.setdefault("label_column", "target")
    cfg.setdefault("label_name_column", "category")
    return _prepare_hf_audio_dataset(paths, "ESC50", cfg)


def _prepare_yesno(
    paths: DatasetPaths,
    checksum_spec: Union[str, Mapping[str, str], None],
    dataset_cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    cfg = dict(dataset_cfg or {})
    dataset_kwargs = {k: v for k, v in cfg.items() if k not in {"split"}}

    _ensure_torchaudio_backend()

    dataset = _load_torch_dataset(
        torchaudio.datasets.YESNO,
        "YESNO",
        paths,
        checksum_spec,
        ["yes_no*tar.gz", "yesno*tar.gz"],
        **dataset_kwargs,
    )

    samples = _load_yesno_samples(dataset)

    log_mels: List[torch.Tensor] = []
    lengths: List[int] = []
    label_tensors: List[torch.Tensor] = []

    for waveform, sample_rate, labels in tqdm(samples, desc="Processing YESNO", leave=False):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        log_mel = _compute_log_mel(waveform, int(sample_rate)).squeeze(0)
        log_mels.append(log_mel)
        lengths.append(log_mel.size(-1))
        if labels is not None:
            label_tensors.append(torch.tensor(labels, dtype=torch.long))

    if not log_mels:
        raise RuntimeError("No audio samples found in YESNO dataset.")

    n_mels = log_mels[0].size(0)
    max_length = max(lengths)
    features = torch.zeros((len(log_mels), n_mels, max_length), dtype=torch.float32)
    for idx, (log_mel, length) in enumerate(zip(log_mels, lengths)):
        features[idx, :, :length] = log_mel

    payload: Dict[str, torch.Tensor] = {
        "features": features,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }

    metadata: Dict[str, Any] = {
        "split": cfg.get("split", "unspecified"),
        "labels_available": bool(label_tensors),
        "num_samples": len(log_mels),
        "source_url": getattr(dataset, "url", None),
    }

    if label_tensors:
        payload["labels"] = torch.stack(label_tensors)

    return payload, metadata


AUDIO_DATASETS = {
    "UrbanSound8K": _prepare_urbansound8k,
    "ESC50": _prepare_esc50,
    "YESNO": _prepare_yesno,
}


# ---------------------------------------------------------------------------
# Dataset packaging utilities
# ---------------------------------------------------------------------------


class VisionTensorDataset(Dataset):
    """Simple dataset wrapper for image tensors and optional labels."""

    def __init__(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"image": self.features[idx]}
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        return sample


class AudioTensorDataset(Dataset):
    """Dataset wrapper for padded log-mel spectrograms with optional labels."""

    def __init__(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        self.features = features
        self.lengths = lengths
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"audio": self.features[idx], "length": self.lengths[idx]}
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        return sample


def _split_payload(
    payload: Dict[str, torch.Tensor], split: float, seed: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features = payload["features"]
    num_samples = features.size(0)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    split_idx = int(num_samples * split)
    val_idx = indices[:split_idx]
    train_idx = indices[split_idx:]

    def _subset(idxs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {key: tensor[idxs] for key, tensor in payload.items()}

    return _subset(train_idx), _subset(val_idx)


def _compute_statistics(tensor: torch.Tensor) -> Dict[str, List[float]]:
    if tensor.dim() == 4:
        # Vision: [N, C, H, W]
        mean = tensor.mean(dim=(0, 2, 3))
        std = tensor.std(dim=(0, 2, 3))
    elif tensor.dim() == 3:
        # Audio: [N, F, T]
        mean = tensor.mean(dim=(0, 2))
        std = tensor.std(dim=(0, 2))
    else:
        raise ValueError("Unsupported tensor shape for statistics computation")
    return {"mean": mean.tolist(), "std": std.tolist()}


def _save_processed(
    tensor_dict: Dict[str, torch.Tensor],
    split_name: Literal["train", "val"],
    modality: Literal["vision", "audio"],
    dataset_name: str,
    paths: DatasetPaths,
) -> Path:
    filename = f"{dataset_name.lower()}_{modality}_{split_name}.pt"
    output_path = paths.processed / filename
    torch.save(tensor_dict, output_path)
    return output_path


def _save_statistics(stats: Dict[str, List[float]], modality: str, dataset_name: str, paths: DatasetPaths) -> Path:
    output_path = paths.processed / f"{dataset_name.lower()}_{modality}_stats.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return output_path


def _save_metadata(
    metadata: Mapping[str, Any],
    modality: str,
    dataset_name: str,
    paths: DatasetPaths,
) -> Optional[Path]:
    if not metadata:
        return None
    output_path = paths.processed / f"{dataset_name.lower()}_{modality}_metadata.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output_path


def _build_dataloader(
    payload: Dict[str, torch.Tensor],
    modality: Literal["vision", "audio"],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    if modality == "vision":
        dataset = VisionTensorDataset(
            payload["features"], payload.get("labels")
        )
    elif modality == "audio":
        dataset = AudioTensorDataset(
            payload["features"],
            payload["lengths"],
            payload.get("labels"),
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported modality '{modality}'.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _resolve_dataset_spec(
    cfg: Config,
    modality: Literal["vision", "audio"],
    default_name: str,
) -> Tuple[str, Union[str, Mapping[str, str], None], Mapping[str, Any]]:
    data_cfg = cfg.data or {}
    datasets_cfg = data_cfg.get("datasets")

    name = default_name
    checksum_spec: Union[str, Mapping[str, str], None] = ""
    options: Mapping[str, Any] = {}

    if isinstance(datasets_cfg, Mapping):
        entry = datasets_cfg.get(modality)
        if entry is not None:
            if isinstance(entry, str):
                name = entry
            else:
                name = entry.get("name", name)
                checksum_spec = entry.get("checksum", checksum_spec)
                options = entry.get("options", options)

    legacy_key = f"{modality}_dataset"
    if legacy_key in data_cfg:
        name = data_cfg[legacy_key]

    checksum_spec = checksum_spec or data_cfg.get("checksum", {}).get(name, "")
    if not options:
        options = data_cfg.get("dataset_options", {}).get(name, {})

    return name, checksum_spec, dict(options or {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_datasets(cfg: Config) -> Dict[str, Dict[str, DataLoader]]:
    set_seed(cfg.seed)

    base_path = DatasetPaths.from_root(Path(cfg.data["root"]))
    dataloaders: Dict[str, Dict[str, DataLoader]] = {}
    modalities = cfg.modalities or {"vision": True, "audio": True}

    if modalities.get("vision", False):
        dataset_name, checksum_spec, dataset_options = _resolve_dataset_spec(
            cfg, "vision", "CIFAR10"
        )
        preprocess_fn = VISION_DATASETS.get(dataset_name)
        if preprocess_fn is None:
            raise ValueError(f"Unsupported vision dataset '{dataset_name}'.")
        payload, metadata = preprocess_fn(
            base_path,
            checksum_spec,
            dataset_options,
        )
        if metadata:
            _save_metadata(metadata, "vision", dataset_name, base_path)
        train_payload, val_payload = _split_payload(payload, split=0.05, seed=cfg.seed)
        stats = _compute_statistics(train_payload["features"])
        _save_statistics(stats, "vision", dataset_name, base_path)
        _save_processed(train_payload, "train", "vision", dataset_name, base_path)
        _save_processed(val_payload, "val", "vision", dataset_name, base_path)
        dataloaders["vision"] = {
            "train": _build_dataloader(
                train_payload,
                modality="vision",
                batch_size=cfg.data["batch_size"]["vision"],
                num_workers=cfg.data["num_workers"],
                shuffle=True,
            ),
            "val": _build_dataloader(
                val_payload,
                modality="vision",
                batch_size=cfg.data["batch_size"]["vision"],
                num_workers=cfg.data["num_workers"],
                shuffle=False,
            ),
        }

    if modalities.get("audio", False):
        dataset_name, checksum_spec, dataset_options = _resolve_dataset_spec(
            cfg, "audio", "YESNO"
        )
        preprocess_fn = AUDIO_DATASETS.get(dataset_name)
        if preprocess_fn is None:
            raise ValueError(f"Unsupported audio dataset '{dataset_name}'.")
        payload, metadata = preprocess_fn(
            base_path,
            checksum_spec,
            dataset_options,
        )
        if metadata:
            _save_metadata(metadata, "audio", dataset_name, base_path)
        train_payload, val_payload = _split_payload(payload, split=0.05, seed=cfg.seed)
        stats = _compute_statistics(train_payload["features"])
        _save_statistics(stats, "audio", dataset_name, base_path)
        _save_processed(train_payload, "train", "audio", dataset_name, base_path)
        _save_processed(val_payload, "val", "audio", dataset_name, base_path)
        dataloaders["audio"] = {
            "train": _build_dataloader(
                train_payload,
                modality="audio",
                batch_size=cfg.data["batch_size"]["audio"],
                num_workers=cfg.data["num_workers"],
                shuffle=True,
            ),
            "val": _build_dataloader(
                val_payload,
                modality="audio",
                batch_size=cfg.data["batch_size"]["audio"],
                num_workers=cfg.data["num_workers"],
                shuffle=False,
            ),
        }

    return dataloaders


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_from_args()
    loaders = prepare_datasets(cfg)
    for modality, splits in loaders.items():
        for split_name, loader in splits.items():
            print(
                f"Prepared {modality} {split_name} loader "
                f"with {len(loader.dataset)} samples and batch size {loader.batch_size}"
            )


if __name__ == "__main__":
    main()


__all__ = [
    "DatasetPaths",
    "VisionTensorDataset",
    "AudioTensorDataset",
    "prepare_datasets",
    "verify_checksum_file",
]
