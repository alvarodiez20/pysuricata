from __future__ import annotations

import glob
import gzip
import os
import pickle
from typing import Any, List, Mapping, Optional, Tuple


class CheckpointManager:
    """Lightweight helper to manage streaming checkpoints on disk.

    Stores gzipped pickles per chunk and (optionally) an HTML snapshot.
    Keeps only a limited number of the most recent checkpoints.
    """

    def __init__(
        self,
        directory: str,
        prefix: str = "pysuricata_ckpt",
        keep: int = 3,
        write_html: bool = False,
    ) -> None:
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self.prefix = prefix
        self.keep = max(1, int(keep))
        self.write_html = write_html

    def _glob(self, ext: str) -> List[str]:
        return sorted(
            glob.glob(os.path.join(self.directory, f"{self.prefix}_chunk*.{ext}"))
        )

    def _path_for(self, chunk_idx: int, ext: str) -> str:
        return os.path.join(self.directory, f"{self.prefix}_chunk{chunk_idx:06d}.{ext}")

    def rotate(self) -> None:
        pkls = self._glob("pkl.gz")
        if len(pkls) <= self.keep:
            return
        to_remove = pkls[: len(pkls) - self.keep]
        for p in to_remove:
            try:
                os.remove(p)
            except Exception:
                pass
            html_p = p.replace(".pkl.gz", ".html")
            try:
                if os.path.exists(html_p):
                    os.remove(html_p)
            except Exception:
                pass

    def save(
        self, chunk_idx: int, state: Mapping[str, Any], html: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        pkl_path = self._path_for(chunk_idx, "pkl.gz")
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        html_path = None
        if self.write_html and html is not None:
            html_path = self._path_for(chunk_idx, "html")
            with open(html_path, "w", encoding="utf-8") as hf:
                hf.write(html)
        self.rotate()
        return pkl_path, html_path


def make_state_snapshot(
    *,
    kinds: Any,
    accs: Mapping[str, Any],
    row_kmv: Any,
    total_missing_cells: int,
    approx_mem_bytes: int,
    chunk_idx: int,
    first_columns: list[str],
    sample_section_html: str,
    cfg: Any,
) -> dict[str, Any]:
    """Build a pickle‑friendly snapshot dictionary for checkpointing.

    Keeps only lightweight, serializable state required to resume/report.
    """
    return {
        "version": 1,
        "timestamp": __import__("time").time(),
        "chunk_idx": int(chunk_idx),
        "first_columns": list(first_columns),
        "sample_section_html": sample_section_html,
        "kinds": kinds,
        "accs": dict(accs),
        "row_kmv": row_kmv,
        "total_missing_cells": int(total_missing_cells),
        "approx_mem_bytes": int(approx_mem_bytes),
        "config": {
            "title": getattr(cfg, "title", None),
            "chunk_size": getattr(cfg, "chunk_size", None),
            "numeric_sample_k": getattr(cfg, "numeric_sample_k", None),
            "uniques_k": getattr(cfg, "uniques_k", None),
            "topk_k": getattr(cfg, "topk_k", None),
            "compute_correlations": getattr(cfg, "compute_correlations", None),
            "corr_threshold": getattr(cfg, "corr_threshold", None),
        },
    }


def maybe_make_manager(cfg: Any, output_file: Optional[str]) -> Optional[CheckpointManager]:
    """Factory for an optional checkpoint manager based on config.

    Returns None when checkpointing is disabled via config.
    """
    try:
        every = int(getattr(cfg, "checkpoint_every_n_chunks", 0))
    except Exception:
        every = 0
    if every <= 0:
        return None
    base_dir = getattr(cfg, "checkpoint_dir", None) or (
        os.path.dirname(output_file) if output_file else os.getcwd()
    )
    return CheckpointManager(
        base_dir,
        prefix=getattr(cfg, "checkpoint_prefix", "pysuricata_ckpt"),
        keep=int(getattr(cfg, "checkpoint_max_to_keep", 3)),
        write_html=bool(getattr(cfg, "checkpoint_write_html", False)),
    )
