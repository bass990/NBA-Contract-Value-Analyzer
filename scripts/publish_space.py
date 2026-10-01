"""Publish the Streamlit dashboard to the Hugging Face Space.

    hf auth login                    # once, with a token that has write access to the Space
    python scripts/publish_space.py  # uploads app.py, requirements.txt, README.md, src/, data/processed/{model.pkl,predictions_latest.csv}

Uploads exactly the layout described in huggingface-spaces/deployment_notes.md in one
commit and prints the Space URL. Files already in the Space that are not in this set
are left alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE = "bass990/NBA-Contract-Value-Analyzer"


def files() -> dict[str, Path]:
    out = {
        "app.py": ROOT / "huggingface-spaces" / "app.py",
        "requirements.txt": ROOT / "huggingface-spaces" / "requirements.txt",
        "README.md": ROOT / "huggingface-spaces" / "README.md",
        "data/processed/model.pkl": ROOT / "data" / "processed" / "model.pkl",
        "data/processed/predictions_latest.csv": ROOT / "data" / "processed" / "predictions_latest.csv",
    }
    for py in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        out[py.relative_to(ROOT).as_posix()] = py
    return out


def main() -> int:
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        print(f"not logged in to Hugging Face ({str(exc)[:80]}). Run: hf auth login")
        return 1
    fs = files()
    missing = [p for p in fs.values() if not p.exists()]
    if missing:
        print("missing:", *missing, sep="\n  ")
        return 1
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in fs.items()]
    info = api.create_commit(repo_id=SPACE, repo_type="space", operations=ops,
                             commit_message="Dashboard v1.1: predictions_latest.csv regenerated from the retrained bundle (R2 0.733, MAE $1.41M)")
    print(f"published {len(ops)} files as {user}: {info.commit_url}\nhttps://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
