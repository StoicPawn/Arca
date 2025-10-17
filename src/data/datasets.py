"""Synthetic co-occurrence dataset generation pipeline."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import signal

import librosa

from ..utils.config import Config, build_arg_parser, load_config
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import set_global_seed


LOGGER = get_logger(__name__)

SAMPLE_RATE = 16_000
IMAGE_SIZE = 96
N_MELS = 64
CLIP_LENGTH_DEFAULT = (1.5, 3.0)
EVENT_DURATION_RANGE = (0.45, 0.85)
TEMPORAL_SCALES_DEFAULT = (0.12, 0.35, 0.7)
MAX_OBJECTS_DEFAULT = 3
MAX_EVENTS_DEFAULT = 3

COLOR_TABLE: Dict[str, Tuple[int, int, int]] = {
    "red": (219, 68, 55),
    "green": (15, 157, 88),
    "blue": (66, 133, 244),
}

COLOR_TO_TIMBRE: Dict[str, str] = {
    "red": "saw",
    "green": "square",
    "blue": "sine",
}

SHAPE_TO_INTERVAL: Dict[str, int] = {
    "square": 0,
    "triangle": 7,
    "circle": 4,
}

SCALE_VALUES: Dict[str, float] = {
    "small": 0.38,
    "medium": 0.55,
    "large": 0.72,
}

SCALE_TO_AMPLITUDE: Dict[str, float] = {
    "small": 0.35,
    "medium": 0.55,
    "large": 0.75,
}


@dataclass
class SplitResult:
    train_path: pathlib.Path
    val_path: pathlib.Path
    test_path: pathlib.Path
    manifest_path: pathlib.Path
    stats_path: pathlib.Path


def ensure_dirs(config: Config) -> Tuple[pathlib.Path, pathlib.Path]:
    data_root = config.data_root
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def _apply_color_jitter(color: Tuple[int, int, int], jitter: float, rng: np.random.Generator) -> Tuple[int, int, int]:
    if jitter <= 0:
        return color
    base = np.asarray(color, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, jitter, size=3)
    jittered = np.clip(base + noise, 0.0, 1.0)
    return tuple(int(round(channel * 255.0)) for channel in jittered)


def _render_object(
    shape: str,
    color: Tuple[int, int, int],
    scale: float,
    position: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    cx = position[0] * IMAGE_SIZE
    cy = position[1] * IMAGE_SIZE
    extent = max(4.0, scale * IMAGE_SIZE * 0.45)

    if shape == "square":
        bbox = [cx - extent, cy - extent, cx + extent, cy + extent]
        draw.rectangle(bbox, fill=color)
    elif shape == "circle":
        bbox = [cx - extent, cy - extent, cx + extent, cy + extent]
        draw.ellipse(bbox, fill=color)
    elif shape == "triangle":
        angle = rng.uniform(0.0, 2.0 * math.pi)
        points: List[Tuple[float, float]] = []
        for k in range(3):
            theta = angle + (2.0 * math.pi * k) / 3.0
            points.append((cx + extent * math.cos(theta), cy + extent * math.sin(theta)))
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unsupported shape '{shape}'")

    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)


def _generate_waveform(
    timbre: str,
    frequency: float,
    duration: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    num_samples = max(1, int(round(duration * SAMPLE_RATE)))
    t = np.linspace(0.0, duration, num_samples, endpoint=False)
    phase = rng.uniform(0.0, 2.0 * math.pi)

    if timbre == "sine":
        waveform = np.sin(2.0 * math.pi * frequency * t + phase)
    elif timbre == "saw":
        waveform = signal.sawtooth(2.0 * math.pi * frequency * t + phase)
    elif timbre == "square":
        waveform = signal.square(2.0 * math.pi * frequency * t + phase)
    else:
        raise ValueError(f"Unsupported timbre '{timbre}'")

    envelope = np.hanning(num_samples)
    waveform = waveform * envelope
    waveform += rng.normal(0.0, 0.0025, size=waveform.shape)
    waveform = np.clip(waveform * amplitude, -1.0, 1.0)
    return waveform.astype(np.float32)


def _compute_log_mel(audio: np.ndarray) -> np.ndarray:
    win_length = int(0.025 * SAMPLE_RATE)
    hop_length = int(0.010 * SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=win_length * 2,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=N_MELS,
        power=2.0,
        center=True,
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    return log_mel.astype(np.float32)


def _temporal_kernel(
    vision_times: Sequence[float],
    audio_times: Sequence[float],
    clip_length: float,
    scales: Sequence[float],
) -> np.ndarray:
    kernel = np.zeros((len(vision_times), len(audio_times)), dtype=np.float32)
    for i, v_time in enumerate(vision_times):
        for j, a_time in enumerate(audio_times):
            delta = abs(v_time - a_time)
            weights = [math.exp(-delta / max(scale, 1e-6)) for scale in scales]
            long_term = max(0.0, 1.0 - delta / max(clip_length, 1e-6))
            kernel[i, j] = float((sum(weights) / len(weights)) * 0.7 + long_term * 0.3)
    return kernel


class SyntheticCooccurrenceBuilder:
    def __init__(self, options: Dict[str, object], kernel_scales: Sequence[float], seed: int) -> None:
        self.seed = seed
        self.kernel_scales = tuple(float(s) for s in kernel_scales) or TEMPORAL_SCALES_DEFAULT

        self.clip_length_range = tuple(
            float(x) for x in options.get("clip_length", CLIP_LENGTH_DEFAULT)
        )
        if len(self.clip_length_range) != 2:
            raise ValueError("clip_length must have exactly two entries [min, max]")

        self.max_objects = int(options.get("max_objects", MAX_OBJECTS_DEFAULT))
        self.max_events = int(options.get("max_events", MAX_EVENTS_DEFAULT))
        if self.max_events != self.max_objects:
            LOGGER.warning(
                "max_events (%d) differs from max_objects (%d); enforcing equality for polyphony",
                self.max_events,
                self.max_objects,
            )
            self.max_events = self.max_objects

        self.jitter = {
            "color": float(options.get("jitter", {}).get("color", 0.0)),
            "pitch": float(options.get("jitter", {}).get("pitch", 0.0)),
            "loudness": float(options.get("jitter", {}).get("loudness", 0.0)),
            "position": float(options.get("jitter", {}).get("position", 0.0)),
            "scale": float(options.get("jitter", {}).get("scale", 0.0)),
        }

        requested_shapes = options.get("shapes")
        requested_colors = options.get("colors")

        self.shapes: List[str] = sorted(
            set(requested_shapes) if requested_shapes else SHAPE_TO_INTERVAL.keys()
        )
        self.colors: List[str] = sorted(
            set(requested_colors) if requested_colors else COLOR_TO_TIMBRE.keys()
        )

        invalid_shapes = [shape for shape in self.shapes if shape not in SHAPE_TO_INTERVAL]
        invalid_colors = [color for color in self.colors if color not in COLOR_TO_TIMBRE]
        if invalid_shapes:
            raise ValueError(f"Unsupported shapes requested: {invalid_shapes}")
        if invalid_colors:
            raise ValueError(f"Unsupported colors requested: {invalid_colors}")

        self.shape_to_index = {shape: idx for idx, shape in enumerate(self.shapes)}
        self.color_to_index = {color: idx for idx, color in enumerate(self.colors)}
        self.scale_labels = list(SCALE_VALUES.keys())
        self.scale_to_index = {label: idx for idx, label in enumerate(self.scale_labels)}

        holdout_pairs = options.get("holdout_pairs", [])
        self.holdout_pairs: List[Tuple[str, str]] = [tuple(pair) for pair in holdout_pairs]

        self.base_frequency = float(options.get("base_frequency", 220.0))

    def _rng(self, split_seed: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + split_seed)

    def _sample_pairs(
        self,
        rng: np.random.Generator,
        pool: Sequence[Tuple[str, str]],
        count: int,
    ) -> List[Tuple[str, str]]:
        if not pool:
            raise ValueError("No latent pairs available to sample from")
        return [pool[rng.integers(0, len(pool))] for _ in range(count)]

    def _generate_clip(
        self,
        rng: np.random.Generator,
        split: str,
        allow_pairs: Sequence[Tuple[str, str]],
        require_holdout: bool,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        clip_length = rng.uniform(*self.clip_length_range)
        num_objects = rng.integers(1, self.max_objects + 1)

        pairs: List[Tuple[str, str]] = []
        if require_holdout and self.holdout_pairs:
            pairs.append(self.holdout_pairs[rng.integers(0, len(self.holdout_pairs))])
        remaining = max(0, num_objects - len(pairs))
        pairs.extend(self._sample_pairs(rng, allow_pairs, remaining))
        rng.shuffle(pairs)

        base_times = np.linspace(clip_length * 0.35, clip_length * 0.65, num_objects)

        objects: List[Dict[str, object]] = []
        audio_events: List[Dict[str, object]] = []

        for idx, (shape, color) in enumerate(pairs):
            scale_label = self.scale_labels[rng.integers(0, len(self.scale_labels))]
            scale_base = SCALE_VALUES[scale_label]
            scale = float(np.clip(
                scale_base + rng.normal(0.0, self.jitter["scale"] * scale_base),
                0.25,
                0.9,
            ))

            pos_jitter = self.jitter["position"]
            position = (
                float(np.clip(0.3 + rng.normal(0.0, pos_jitter), 0.15, 0.85)),
                float(np.clip(0.3 + rng.normal(0.0, pos_jitter), 0.15, 0.85)),
            )

            color_rgb = COLOR_TABLE[color]
            jittered_color = _apply_color_jitter(color_rgb, self.jitter["color"], rng)
            image = _render_object(shape, jittered_color, scale, position, rng)

            timbre = COLOR_TO_TIMBRE[color]
            interval = SHAPE_TO_INTERVAL[shape]
            base_freq = self.base_frequency * (2.0 ** (interval / 12.0))
            freq = float(
                max(55.0, base_freq * (1.0 + rng.normal(0.0, self.jitter["pitch"])) )
            )
            duration = float(
                np.clip(
                    rng.uniform(*EVENT_DURATION_RANGE)
                    + rng.normal(0.0, 0.08),
                    0.3,
                    clip_length - 0.1,
                )
            )
            start_time = float(
                np.clip(
                    base_times[idx] + rng.normal(0.0, 0.05 * clip_length),
                    0.0,
                    max(0.01, clip_length - duration),
                )
            )
            vision_time = float(np.clip(start_time + rng.normal(0.0, 0.04), 0.0, clip_length))

            amplitude = float(
                np.clip(
                    SCALE_TO_AMPLITUDE[scale_label]
                    * (1.0 + rng.normal(0.0, self.jitter["loudness"]))
                    * scale,
                    0.1,
                    1.0,
                )
            )

            waveform = _generate_waveform(timbre, freq, duration, amplitude, rng)
            log_mel = _compute_log_mel(waveform)

            objects.append(
                {
                    "image": image,
                    "shape": shape,
                    "color": color,
                    "scale_label": scale_label,
                    "scale": scale,
                    "position": position,
                    "time": vision_time,
                }
            )
            audio_events.append(
                {
                    "spectrogram": log_mel,
                    "start": start_time,
                    "duration": duration,
                    "timbre": timbre,
                    "frequency": freq,
                }
            )

        manifest_entry = {
            "clip_length": clip_length,
            "objects": [
                {
                    "shape": obj["shape"],
                    "color": obj["color"],
                    "scale_label": obj["scale_label"],
                    "scale": obj["scale"],
                    "position": obj["position"],
                    "time": obj["time"],
                }
                for obj in objects
            ],
            "audio_events": [
                {
                    "timbre": evt["timbre"],
                    "start": evt["start"],
                    "duration": evt["duration"],
                    "frequency": evt["frequency"],
                }
                for evt in audio_events
            ],
        }

        payload = {
            "clip_length": clip_length,
            "objects": objects,
            "audio_events": audio_events,
        }
        return manifest_entry, payload

    def build_split(
        self,
        split: str,
        count: int,
        allow_pairs: Sequence[Tuple[str, str]],
        require_holdout: bool,
        split_seed: int,
    ) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
        rng = self._rng(split_seed)
        manifest: List[Dict[str, object]] = []
        payloads: List[Dict[str, object]] = []

        for idx in range(count):
            entry, payload = self._generate_clip(rng, split, allow_pairs, require_holdout)
            entry["clip_id"] = f"{split}_{idx:05d}"
            manifest.append(entry)
            payloads.append(payload)

        data = self._pack_split(payloads)
        return manifest, data

    def _pack_split(self, payloads: Iterable[Dict[str, object]]) -> Dict[str, np.ndarray]:
        payload_list = list(payloads)
        num_samples = len(payload_list)
        max_event_frames = 0
        for payload in payload_list:
            for event in payload["audio_events"]:
                max_event_frames = max(max_event_frames, event["spectrogram"].shape[1])
        max_event_frames = max(1, max_event_frames)

        vision = np.zeros(
            (num_samples, self.max_objects, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )
        vision_mask = np.zeros((num_samples, self.max_objects), dtype=bool)
        vision_times = np.zeros((num_samples, self.max_objects), dtype=np.float32)
        vision_shapes = np.full((num_samples, self.max_objects), -1, dtype=np.int64)
        vision_colors = np.full((num_samples, self.max_objects), -1, dtype=np.int64)
        vision_scales = np.full((num_samples, self.max_objects), -1, dtype=np.int64)

        audio = np.zeros(
            (num_samples, self.max_events, N_MELS, max_event_frames), dtype=np.float32
        )
        audio_mask = np.zeros((num_samples, self.max_events, max_event_frames), dtype=bool)
        audio_event_mask = np.zeros((num_samples, self.max_events), dtype=bool)
        audio_times = np.zeros((num_samples, self.max_events), dtype=np.float32)
        audio_durations = np.zeros((num_samples, self.max_events), dtype=np.float32)

        temporal_kernel = np.zeros(
            (num_samples, self.max_objects, self.max_events), dtype=np.float32
        )
        clip_lengths = np.zeros(num_samples, dtype=np.float32)

        audio_sum = np.zeros(N_MELS, dtype=np.float64)
        audio_sq_sum = np.zeros(N_MELS, dtype=np.float64)
        audio_weight = 0.0

        for sample_idx, payload in enumerate(payload_list):
            clip_length = float(payload["clip_length"])
            clip_lengths[sample_idx] = clip_length

            objects = payload["objects"]
            audio_events = payload["audio_events"]

            for obj_idx, obj in enumerate(objects[: self.max_objects]):
                vision_mask[sample_idx, obj_idx] = True
                vision[sample_idx, obj_idx] = obj["image"].astype(np.float32)
                vision_times[sample_idx, obj_idx] = float(obj["time"])
                vision_shapes[sample_idx, obj_idx] = self.shape_to_index[obj["shape"]]
                vision_colors[sample_idx, obj_idx] = self.color_to_index[obj["color"]]
                vision_scales[sample_idx, obj_idx] = self.scale_to_index[obj["scale_label"]]

            for event_idx, event in enumerate(audio_events[: self.max_events]):
                audio_event_mask[sample_idx, event_idx] = True
                spec = event["spectrogram"]
                length = spec.shape[1]
                audio[sample_idx, event_idx, :, :length] = spec
                audio_mask[sample_idx, event_idx, :length] = True
                audio_times[sample_idx, event_idx] = float(event["start"])
                audio_durations[sample_idx, event_idx] = float(event["duration"])

                audio_sum += spec.sum(axis=1)
                audio_sq_sum += (spec ** 2).sum(axis=1)
                audio_weight += float(length)

            valid_vision_times = [obj["time"] for obj in objects[: self.max_objects]]
            valid_audio_times = [event["start"] for event in audio_events[: self.max_events]]
            kernel_matrix = _temporal_kernel(
                valid_vision_times,
                valid_audio_times,
                clip_length,
                self.kernel_scales,
            )
            rows, cols = kernel_matrix.shape
            temporal_kernel[sample_idx, :rows, :cols] = kernel_matrix

        audio_mean = (audio_sum / max(audio_weight, 1.0)).astype(np.float32)
        audio_var = audio_sq_sum / max(audio_weight, 1.0) - np.square(audio_mean)
        audio_std = np.sqrt(np.clip(audio_var, 1e-6, None)).astype(np.float32)

        return {
            "vision": vision,
            "vision_mask": vision_mask,
            "vision_times": vision_times,
            "vision_shape_idx": vision_shapes,
            "vision_color_idx": vision_colors,
            "vision_scale_idx": vision_scales,
            "audio": audio,
            "audio_mask": audio_mask,
            "audio_event_mask": audio_event_mask,
            "audio_times": audio_times,
            "audio_durations": audio_durations,
            "temporal_kernel": temporal_kernel,
            "clip_length": clip_lengths,
            "audio_mean": audio_mean,
            "audio_std": audio_std,
            "temporal_scales": np.asarray(self.kernel_scales, dtype=np.float32),
        }


def _save_npz(path: pathlib.Path, arrays: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def _save_json(path: pathlib.Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def run_pipeline(config: Config) -> Dict[str, SplitResult]:
    log_dir = config.logging_dir / "logs"
    configure_logging(log_dir)
    LOGGER.info("Logging initialised at %s", log_dir)

    seed = int(config.raw.get("seed", 42))
    set_global_seed(seed)

    _, processed_dir = ensure_dirs(config)

    datasets_cfg = config.raw.get("data", {}).get("datasets", {})
    if "cooccurrence" not in datasets_cfg:
        raise ValueError("Configuration missing data.datasets.cooccurrence section")

    cooccur_cfg = datasets_cfg["cooccurrence"]
    options = dict(cooccur_cfg.get("options", {}))
    kernel_scales = config.raw.get("model", {}).get("kernel_scales", TEMPORAL_SCALES_DEFAULT)

    builder = SyntheticCooccurrenceBuilder(options, kernel_scales, seed)

    train_clips = int(options.get("train_clips", 0))
    val_clips = int(options.get("val_clips", 0))
    test_clips = int(options.get("test_clips", 0))

    all_pairs = [(shape, color) for shape in builder.shapes for color in builder.colors]
    holdout_pairs = builder.holdout_pairs
    train_pairs = [pair for pair in all_pairs if pair not in holdout_pairs]
    if not train_pairs:
        raise ValueError("No latent pairs available for training after applying holdouts")

    LOGGER.info(
        "Generating synthetic co-occurrence dataset | train=%d val=%d test=%d",
        train_clips,
        val_clips,
        test_clips,
    )

    manifests: Dict[str, List[Dict[str, object]]] = {}
    split_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    if train_clips > 0:
        train_manifest, train_arrays = builder.build_split(
            "train",
            train_clips,
            train_pairs,
            require_holdout=False,
            split_seed=17,
        )
        manifests["train"] = train_manifest
        split_arrays["train"] = train_arrays

    if val_clips > 0:
        val_allow = train_pairs + holdout_pairs if holdout_pairs else all_pairs
        val_manifest, val_arrays = builder.build_split(
            "val",
            val_clips,
            val_allow,
            require_holdout=bool(holdout_pairs),
            split_seed=29,
        )
        manifests["val"] = val_manifest
        split_arrays["val"] = val_arrays

    if test_clips > 0:
        test_allow = train_pairs + holdout_pairs if holdout_pairs else all_pairs
        test_manifest, test_arrays = builder.build_split(
            "test",
            test_clips,
            test_allow,
            require_holdout=bool(holdout_pairs),
            split_seed=53,
        )
        manifests["test"] = test_manifest
        split_arrays["test"] = test_arrays

    train_path = processed_dir / "cooccurrence_train.npz"
    val_path = processed_dir / "cooccurrence_val.npz"
    test_path = processed_dir / "cooccurrence_test.npz"
    manifest_path = processed_dir / "cooccurrence_manifest.json"
    stats_path = processed_dir / "cooccurrence_stats.json"

    if "train" in split_arrays:
        _save_npz(train_path, split_arrays["train"])
        LOGGER.info("Saved training split to %s", train_path)
    if "val" in split_arrays:
        _save_npz(val_path, split_arrays["val"])
        LOGGER.info("Saved validation split to %s", val_path)
    if "test" in split_arrays:
        _save_npz(test_path, split_arrays["test"])
        LOGGER.info("Saved test split to %s", test_path)

    manifest_blob = {
        "seed": seed,
        "splits": manifests,
    }
    _save_json(manifest_path, manifest_blob)

    stats_blob = {
        "seed": seed,
        "clip_length_range": list(builder.clip_length_range),
        "event_duration_range": list(EVENT_DURATION_RANGE),
        "temporal_scales": list(builder.kernel_scales),
        "shapes": list(builder.shapes),
        "colors": list(builder.colors),
        "scale_labels": list(builder.scale_labels),
        "shape_to_interval": {k: SHAPE_TO_INTERVAL[k] for k in builder.shapes},
        "color_to_timbre": {k: COLOR_TO_TIMBRE[k] for k in builder.colors},
        "scale_values": {k: SCALE_VALUES[k] for k in builder.scale_labels},
        "scale_to_amplitude": {k: SCALE_TO_AMPLITUDE[k] for k in builder.scale_labels},
        "max_objects": builder.max_objects,
        "max_events": builder.max_events,
        "image_size": IMAGE_SIZE,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "holdout_pairs": [list(pair) for pair in holdout_pairs],
    }
    _save_json(stats_path, stats_blob)

    return {
        "cooccurrence": SplitResult(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            manifest_path=manifest_path,
            stats_path=stats_path,
        )
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()

