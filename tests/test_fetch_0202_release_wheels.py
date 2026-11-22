"""Release-wheel preparation checks both registry metadata and bytes."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_0202_release_wheels.py"
_SPEC = importlib.util.spec_from_file_location("fetch_0202_release_wheels", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("corrupt", ["none", "metadata", "bytes"])
def test_fetch_requires_pinned_metadata_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: str
) -> None:
    payload = b"trusted wheel bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(_MODULE, "_WHEELS", {"0.19.5": digest})
    metadata = {
        "urls": [
            {
                "filename": "cogkura-0.19.5-py3-none-any.whl",
                "digests": {"sha256": "0" * 64 if corrupt == "metadata" else digest},
                "url": "https://files.pythonhosted.org/wheel",
            }
        ]
    }

    def fake_open(url: str, *, timeout: int) -> io.BytesIO:
        del timeout
        if url.startswith("https://pypi.org/"):
            return io.BytesIO(json.dumps(metadata).encode())
        assert url == "https://files.pythonhosted.org/wheel"
        return io.BytesIO(b"tampered" if corrupt == "bytes" else payload)

    monkeypatch.setattr(_MODULE, "urlopen", fake_open)
    if corrupt == "none":
        _MODULE.fetch_wheels(tmp_path)
        assert (tmp_path / "cogkura-0.19.5-py3-none-any.whl").read_bytes() == payload
    else:
        with pytest.raises(RuntimeError, match="metadata|hash"):
            _MODULE.fetch_wheels(tmp_path)
        assert not (tmp_path / "cogkura-0.19.5-py3-none-any.whl").exists()
