import glob
import gzip
import os
import pickle
from dataclasses import dataclass

import pytest

from pysuricata.checkpoint import (
    CheckpointManager,
    make_state_snapshot,
    maybe_make_manager,
)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Provide a temporary directory for checkpoints."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return str(d)


def test_checkpoint_manager_init(temp_checkpoint_dir):
    # Test directory creation and initialization
    mgr = CheckpointManager(
        temp_checkpoint_dir, prefix="test_ckpt", keep=2, write_html=True
    )
    assert mgr.directory == temp_checkpoint_dir
    assert mgr.prefix == "test_ckpt"
    assert mgr.keep == 2
    assert mgr.write_html is True


def test_checkpoint_manager_save_and_rotate(temp_checkpoint_dir):
    mgr = CheckpointManager(
        temp_checkpoint_dir, prefix="rotate_test", keep=2, write_html=True
    )
    state = {"data": [1, 2, 3]}

    # Save 3 checkpoints (only 2 should be kept)
    for i in range(3):
        mgr.save(i, state, html=f"<html>chunk {i}</html>")

    # Check files
    pkls = sorted(glob.glob(os.path.join(temp_checkpoint_dir, "*.pkl.gz")))
    htmls = sorted(glob.glob(os.path.join(temp_checkpoint_dir, "*.html")))

    assert len(pkls) == 2
    assert len(htmls) == 2

    # Verify that the oldest ones (chunk 0) were deleted, leaving chunk 1 and 2
    assert "rotate_test_chunk000001.pkl.gz" in pkls[0]
    assert "rotate_test_chunk000002.pkl.gz" in pkls[1]
    assert "rotate_test_chunk000001.html" in htmls[0]
    assert "rotate_test_chunk000002.html" in htmls[1]

    # Verify the contents of the latest pickle
    with gzip.open(pkls[-1], "rb") as f:
        loaded_state = pickle.load(f)
    assert loaded_state == state


def test_checkpoint_manager_save_no_html(temp_checkpoint_dir):
    mgr = CheckpointManager(temp_checkpoint_dir, write_html=False)
    state = {"test": 123}

    # Passing html=... should be ignored since write_html=False
    pkl_path, html_path = mgr.save(1, state, html="<html>ignore this</html>")

    assert os.path.exists(pkl_path)
    assert html_path is None
    htmls = glob.glob(os.path.join(temp_checkpoint_dir, "*.html"))
    assert len(htmls) == 0


def test_rotate_error_handling(temp_checkpoint_dir, monkeypatch):
    mgr = CheckpointManager(temp_checkpoint_dir, keep=1)

    # Save 2 files to force rotation
    p1, _ = mgr.save(1, {"a": 1})

    # Mock os.remove to raise exception to test the except blocks in rotate()
    def mock_remove(path):
        raise PermissionError("Mock error")

    monkeypatch.setattr(os, "remove", mock_remove)

    # This should not raise an exception because of the try/except block
    mgr.save(2, {"a": 2}, html="<html></html>")


def test_make_state_snapshot():
    @dataclass
    class MockConfig:
        title = "Test Title"
        chunk_size = 100
        numeric_sample_k = 1000
        uniques_k = 2048
        topk_k = 50
        compute_correlations = True
        corr_threshold = 0.5

    cfg = MockConfig()

    kinds = {"col1": "numeric"}
    accs = {"col1": {"_count": 10}}
    row_kmv = None

    snapshot = make_state_snapshot(
        kinds=kinds,
        accs=accs,
        row_kmv=row_kmv,
        total_missing_cells=5,
        approx_mem_bytes=1024,
        chunk_idx=10,
        first_columns=["col1", "col2"],
        sample_section_html="<div>sample</div>",
        cfg=cfg,
    )

    assert snapshot["version"] == 1
    assert "timestamp" in snapshot
    assert snapshot["chunk_idx"] == 10
    assert snapshot["first_columns"] == ["col1", "col2"]
    assert snapshot["sample_section_html"] == "<div>sample</div>"
    assert snapshot["kinds"] == kinds
    assert snapshot["accs"] == accs
    assert snapshot["row_kmv"] is None
    assert snapshot["total_missing_cells"] == 5
    assert snapshot["approx_mem_bytes"] == 1024

    # Check config
    assert snapshot["config"]["title"] == "Test Title"
    assert snapshot["config"]["chunk_size"] == 100
    assert snapshot["config"]["numeric_sample_k"] == 1000
    assert snapshot["config"]["compute_correlations"] is True


def test_maybe_make_manager_disabled():
    @dataclass
    class MockConfig:
        checkpoint_every_n_chunks = 0

    assert maybe_make_manager(MockConfig(), None) is None

    # Test completely missing attributes
    class EmptyConfig:
        pass

    assert maybe_make_manager(EmptyConfig(), None) is None


def test_maybe_make_manager_enabled():
    @dataclass
    class MockConfig:
        checkpoint_every_n_chunks = 5
        checkpoint_dir = "/tmp/checkpoints"
        checkpoint_prefix = "my_prefix"
        checkpoint_max_to_keep = 5
        checkpoint_write_html = True

    mgr = maybe_make_manager(MockConfig(), None)

    assert isinstance(mgr, CheckpointManager)
    assert mgr.directory == "/tmp/checkpoints"
    assert mgr.prefix == "my_prefix"
    assert mgr.keep == 5
    assert mgr.write_html is True


def test_maybe_make_manager_fallback(tmp_path):
    @dataclass
    class MockConfig:
        checkpoint_every_n_chunks = 1
        # No checkpoint_dir, no prefix etc.

    out_dir = tmp_path / "output"
    out_file = out_dir / "report.html"

    # Should fallback to os.path.dirname(output_file)
    mgr1 = maybe_make_manager(MockConfig(), str(out_file))
    assert mgr1.directory == str(out_dir)
    assert mgr1.prefix == "pysuricata_ckpt"  # Default
    assert mgr1.keep == 3  # Default
    assert mgr1.write_html is False  # Default

    # Should fallback to os.getcwd() if output_file is None
    mgr2 = maybe_make_manager(MockConfig(), None)
    assert mgr2.directory == os.getcwd()


def test_rotate_html_error_handling(temp_checkpoint_dir, monkeypatch):
    mgr = CheckpointManager(temp_checkpoint_dir, keep=1, write_html=True)

    # Save 2 files to force rotation
    p1, h1 = mgr.save(1, {"a": 1}, html="<html>1</html>")

    # Mock os.remove to raise exception *only* for the html file in rotate()
    original_remove = os.remove

    def mock_remove(path):
        if path.endswith(".html"):
            raise PermissionError("Mock error for HTML")
        original_remove(path)

    monkeypatch.setattr(os, "remove", mock_remove)

    # This should not raise an exception
    mgr.save(2, {"a": 2}, html="<html>2</html>")


def test_maybe_make_manager_invalid_every():
    @dataclass
    class MockConfig:
        @property
        def checkpoint_every_n_chunks(self):
            raise ValueError("Invalid attribute access")

    # Should fallback to `every = 0` and return None
    assert maybe_make_manager(MockConfig(), None) is None
