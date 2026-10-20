"""Fetch a dataset's pooled features, manifest and (LIBERO-X) split files from the Hugging Face Hub into data/<dataset>/.

The Hub repo (``DatasetSpec.hf_repo``) stores the pooled features as ``pooled/features.npz``; here they end up as
``data/<dataset>/features.npz``, where ``load_features`` reads them. The raw rollouts are not downloaded.
"""

import os
from pathlib import Path
from typing import Callable

from ..config import DatasetSpec, get


def hub_patterns(spec: DatasetSpec) -> list[str]:
    """Files of the Hub repo that are needed: the pooled features, the manifest, and for LIBERO-X the held-out task lists."""
    patterns = ["pooled/features.npz", "manifest.csv"]
    if "task" in spec.split_kinds:
        patterns.append("splits/*")  # fnmatch: also matches the nested leave_*_out folders
    return patterns


def download_dataset(dataset: str, fetch: Callable | None = None) -> Path:
    """Download into data/<dataset>/ and return that folder. An existing features.npz is never overwritten.

    ``fetch`` has the signature of ``huggingface_hub.snapshot_download`` (the default); tests pass a stand-in.
    """
    spec = get(dataset)
    folder = spec.data_dir
    target = folder / "features.npz"
    if target.exists():
        raise FileExistsError(f"{target} already exists; remove it to download again")
    if fetch is None:
        from huggingface_hub import snapshot_download as fetch
    folder.mkdir(parents=True, exist_ok=True)
    fetch(repo_id=spec.hf_repo, repo_type="dataset", allow_patterns=hub_patterns(spec), local_dir=str(folder))
    downloaded = folder / "pooled" / "features.npz"
    if not downloaded.exists():
        raise FileNotFoundError(f"{spec.hf_repo} has no pooled/features.npz")
    os.replace(downloaded, target)
    (folder / "pooled").rmdir()
    return folder
