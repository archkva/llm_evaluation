from __future__ import annotations

import zipfile
from pathlib import Path

import gdown
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "benchmark" / "models"

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "UPD_cefr_model_extended_dataset": {
        "zip_name": "UPD_cefr_model_extended_dataset.zip",
        "drive_url": (
            "https://drive.google.com/file/d/"
            "10KafaU6K6brSFpCfmyXIgJiUvjm9ySY4/view?usp=sharing"
        ),
    },
    "UPD_support_model": {
        "zip_name": "UPD_support_model.zip",
        "drive_url": (
            "https://drive.google.com/file/d/"
            "1HIF9_2Dkp1oiGmwVUazA6Ea5ZmJBNG7a/view?usp=sharing"
        ),
    },
}


def _model_is_ready(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "config.json").exists()


def _download_model_zip(url: str, zip_path: Path) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Downloading model archive to {zip_path}...")
    gdown.download(url=url, output=str(zip_path), fuzzy=True, quiet=False)

    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Failed to download model archive from {url}")


def _extract_model_zip(zip_path: Path, model_path: Path) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(
            members,
            desc=f"Extracting {zip_path.name}",
            unit="file",
            leave=False,
        ):
            zip_ref.extract(member, model_path)

    if not _model_is_ready(model_path):
        raise FileNotFoundError(
            f"Extracted archive does not contain a valid model at {model_path}"
        )

    tqdm.write(f"Model ready at {model_path}")


def ensure_model(model_name: str) -> Path:
    """
    Ensure a benchmark model is available locally.

    Checks for an extracted model under benchmark/models/, then for a local
    zip archive, and downloads from Google Drive (see README) if missing.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    spec = MODEL_REGISTRY[model_name]
    model_path = MODEL_DIR / model_name
    zip_path = MODEL_DIR / spec["zip_name"]

    if _model_is_ready(model_path):
        tqdm.write(f"Model directory already exists: {model_path}")
        return model_path

    if not zip_path.is_file():
        _download_model_zip(spec["drive_url"], zip_path)

    _extract_model_zip(zip_path, model_path)
    return model_path
