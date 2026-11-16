from pathlib import Path


def get_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise FileNotFoundError("Could not find the project root")
