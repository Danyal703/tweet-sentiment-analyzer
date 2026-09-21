import hashlib
import json

import pytest

from src import common


def test_corrupted_cache_is_rejected_before_use(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "sources.json").write_text(json.dumps({"example.txt": {
        "url": "https://example.invalid/data", "sha256": hashlib.sha256(b"expected").hexdigest()
    }}))
    (data / "raw/example.txt").write_bytes(b"corrupted")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        common.load_source("example.txt")
