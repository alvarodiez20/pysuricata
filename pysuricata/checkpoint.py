from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, List
import gzip
import glob
import os
import pickle


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

