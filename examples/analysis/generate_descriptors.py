"""Generate descriptors for a Deep Potential training dataset."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from deepmd.infer.deep_pot import DeepPot

from .common import evaluate_descriptors, load_deepmd_dataset


def parse_arguments() -> argparse.Namespace:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the descriptors learned by a Deep Potential model for a "
            "training dataset stored in DeepMD's numpy format."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the DeepMD dataset directory (contains set.* folders).",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the frozen_model.pb file produced by deepmd-kit.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where descriptor arrays and metadata will be stored.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help=(
            "Optional name of the model head to use when the graph exports "
            "multiple outputs (for example: Target_FTS)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames evaluated together when calling the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing descriptor files in the output directory.",
    )
    return parser.parse_args()


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    """Prepare the output directory while respecting the overwrite flag."""

    path.mkdir(parents=True, exist_ok=True)
    descriptor_path = path / "descriptor_frames.npy"
    if descriptor_path.exists() and not overwrite:
        raise FileExistsError(
            f"Descriptor file {descriptor_path} already exists. "
            "Use --overwrite to regenerate the descriptors."
        )


def save_metadata(output_dir: Path, dataset_info: dict[str, Any]) -> None:
    """Persist auxiliary metadata as a JSON file for reproducibility."""

    metadata_path = output_dir / "descriptor_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2)



def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ensure_output_directory(args.output, args.overwrite)

    logging.info("Loading dataset from %s", args.dataset)
    dataset = load_deepmd_dataset(args.dataset)

    logging.info("Loading model graph from %s", args.model)
    model = DeepPot(str(args.model), head=args.head)

    logging.info("Evaluating descriptors in batches of %d frames", args.batch_size)
    descriptor_frames = evaluate_descriptors(dataset, model, batch_size=args.batch_size)
    logging.info("Descriptor tensor shape: %s", descriptor_frames.shape)

    descriptor_path = args.output / "descriptor_frames.npy"
    np.save(descriptor_path, descriptor_frames)
    logging.info("Descriptor tensor saved to %s", descriptor_path)

    mean_descriptor = descriptor_frames.mean(axis=1)
    energies_per_atom = dataset.energies / dataset.n_atoms

    summary_path = args.output / "descriptor_summary.npz"
    np.savez_compressed(
        summary_path,
        mean_descriptor=mean_descriptor,
        energy_per_atom=energies_per_atom,
    )
    logging.info("Descriptor summary saved to %s", summary_path)

    metadata = {
        "dataset": str(args.dataset.resolve()),
        "model": str(args.model.resolve()),
        "n_frames": dataset.n_frames,
        "n_atoms": dataset.n_atoms,
        "descriptor_shape": descriptor_frames.shape,
        "head": args.head,
        "batch_size": args.batch_size,
    }
    save_metadata(args.output, metadata)
    logging.info("Metadata written to descriptor_metadata.json")


if __name__ == "__main__":
    main()
