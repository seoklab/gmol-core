from pathlib import Path


def safe_filename(filename: str) -> str:
    return "".join(
        [c if c.isalnum() or c in ["_", ".", "-"] else "_" for c in filename]
    )


def get_project_root() -> Path:
    return Path(__file__).parents[3]
