"""Structure checks for the Colab notebooks (they cannot run locally: they need Colab + Drive)."""
import ast
import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[1] / "notebooks"
COMMON = [
    "nvidia-smi", "torch.cuda.is_available()", "drive.mount(", "/content/drive/MyDrive/LiverCancer_AI", "extractall(",
    "pip -q install ultralytics==", "data.yaml", "amp=True", "device=0", "resume=True", "RESUME",
    "m.box.mp", "m.box.mr", "def resumable(", "resume=True, data=str(DATA_YAML)", "m.box.map50", "m.box.map)", "best.pt", "last.pt",
    "results.png", "sha256", "for conf in (0.001, 0.25)", "_bundle.zip", "already exists on Drive",
    "last_backup.pt", "add_callback('on_model_save', backup_last)", "def resume_source(",
]
PILOT = ["'yolo11n.pt', 5,", "save_period=1"]
FULL = ["patience=PATIENCE", "'optimizer': 'SGD'", "EPOCHS, PATIENCE, IMGSZ, BATCH, SEED = 50, 20",
        "experiments.csv", "yolo11s.pt", "'T4' not in torch.cuda.get_device_name(0)", "'patience': PATIENCE"]


def notebooks():
    for mod in ("ct", "mri"):
        yield mod, False, NB / f"03_train_yolo_{mod}_colab.ipynb"
        yield mod, True, NB / f"04_train_yolo_{mod}_full_colab.ipynb"


@pytest.mark.parametrize("mod,full,path", list(notebooks()), ids=lambda x: str(x))
def test_notebook_structure(mod, full, path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["cells"]
    code = []
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown") and "id" in c
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))))
            code.append(src)
    text = "\n".join(code)
    for key in COMMON + (FULL if full else PILOT):
        assert key in text, f"{path.name}: missing {key!r}"
    zip_name = "yolo_ct_full.zip" if (full and mod == "ct") else f"yolo_{mod}_pilot.zip"
    assert zip_name in text and f"/content/yolo_{mod}" in text
    assert "SHA256SUMS*.txt" in text
    assert "split='test'" not in text and 'split="test"' not in text, "notebook must not evaluate on Test"
    assert "'test' not in cfg" in text


def test_nbformat_schema():
    nbformat = pytest.importorskip("nbformat")
    for _, _, path in notebooks():
        nbformat.validate(nbformat.read(path, as_version=4))
