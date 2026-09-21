"""Verified downloads and consistent experiment metadata."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def load_source(name):
    """Use the local cache or fetch exactly the bytes recorded in sources.json."""
    manifest = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    info = manifest[name]
    path = ROOT / "data/raw" / name
    if path.exists():
        payload = path.read_bytes()
    else:
        try:
            request = Request(info["url"], headers={"User-Agent": "portfolio-reproduction/1.0"})
            with urlopen(request, timeout=90) as response:
                payload = response.read()
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not download {name}. Connect to the internet and retry. Source: {info['url']}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != info["sha256"]:
        raise ValueError(f"Checksum mismatch for {name}; expected {info['sha256']}, got {digest}. Do not silently substitute a different dataset.")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_run(metrics, packages):
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "metrics.json", metrics)
    write_json(reports / "run_metadata.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "sources": json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8")),
    })
    return reports


def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#f7f9fc", "axes.facecolor": "#ffffff",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#23344d",
        "text.color": "#23344d", "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 120,
        "savefig.dpi": 160,
    })
