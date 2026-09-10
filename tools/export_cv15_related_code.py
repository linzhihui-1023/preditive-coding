#!/usr/bin/env python3
"""
Export all code related to the current C-V15 experiment from a local checkout.

Usage:
    python tools/export_cv15_related_code.py
    python tools/export_cv15_related_code.py --repo /home/lin/predify
    python tools/export_cv15_related_code.py --repo /home/lin/predify --output /home/lin/cv15_all_related_code.zip

What it does:
1. Starts from the C-V15 model/training/entry/contract files.
2. Recursively follows Python imports that resolve inside this repository.
3. Preserves the original repository directory structure.
4. Includes C-V15 result JSON files when present.
5. Writes EXPORT_MANIFEST.md with branch/commit and the exact exported file list.
6. Creates a ZIP archive.

It does NOT include model checkpoints (.pt/.pth), datasets, caches, or external packages.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


SEED_FILES = [
    "predify2021/model_factory/deeplabv3plus_resnet50/task_space_proposal_conditioned_soft_acceptance.py",
    "predify2021/mce_scores/c_v15_proposal_conditioned_soft_acceptance_training.py",
    "predify2021/mce_scores/train_kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance.py",
    "predify2021/mce_scores/check_c_v15_proposal_conditioned_soft_acceptance_contracts.py",
    "predify2021/mce_scores/check_c_v15_dual_reference_acceptance_diagnostics.py",
]

RESULT_DIRS = [
    "results/kitti_step_task_space_prior_c_v15_proposal_conditioned_soft_acceptance",
]

RESULT_FILENAMES = {
    "epoch_001.json",
    "epoch_002.json",
    "epoch_003.json",
    "summary.json",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "datasets",
    "data",
    "cache",
    "caches",
    "experiments",
}

EXCLUDED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".engine",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".mp4", ".avi", ".mov",
}


def run_git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / ".git").exists() and (candidate / "predify2021").exists():
            return candidate
    if (start / "predify2021").exists():
        return start
    raise FileNotFoundError(
        f"Cannot find repository root from {start}. "
        "Pass --repo explicitly, e.g. --repo /home/lin/predify"
    )


def module_to_candidates(repo: Path, module: str) -> list[Path]:
    if not module:
        return []
    rel = Path(*module.split("."))
    candidates = [
        repo / rel.with_suffix(".py"),
        repo / rel / "__init__.py",
    ]
    return [p for p in candidates if p.exists() and p.is_file()]


def package_for_file(repo: Path, file_path: Path) -> list[str]:
    rel = file_path.resolve().relative_to(repo.resolve())
    return list(rel.parts[:-1])


def resolve_relative_module(
    repo: Path,
    source_file: Path,
    level: int,
    module: Optional[str],
) -> list[Path]:
    package = package_for_file(repo, source_file)
    up = max(level - 1, 0)
    if up > len(package):
        return []
    base = package[: len(package) - up]
    if module:
        base += module.split(".")
    return module_to_candidates(repo, ".".join(base))


def imported_internal_files(repo: Path, source_file: Path) -> set[Path]:
    try:
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found: set[Path] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for candidate in module_to_candidates(repo, alias.name):
                    found.add(candidate.resolve())

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                for candidate in resolve_relative_module(
                    repo, source_file, node.level, node.module
                ):
                    found.add(candidate.resolve())

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    package = package_for_file(repo, source_file)
                    up = max(node.level - 1, 0)
                    if up <= len(package):
                        base = package[: len(package) - up]
                        if node.module:
                            base += node.module.split(".")
                        base += alias.name.split(".")
                        for candidate in module_to_candidates(repo, ".".join(base)):
                            found.add(candidate.resolve())
            else:
                module = node.module or ""
                for candidate in module_to_candidates(repo, module):
                    found.add(candidate.resolve())

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    dotted = f"{module}.{alias.name}" if module else alias.name
                    for candidate in module_to_candidates(repo, dotted):
                        found.add(candidate.resolve())

    return found


def is_exportable(repo: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def add_package_init_files(repo: Path, files: set[Path]) -> set[Path]:
    expanded = set(files)
    for file_path in list(files):
        rel_parent = file_path.resolve().relative_to(repo.resolve()).parent
        current = repo
        for part in rel_parent.parts:
            current = current / part
            init_file = current / "__init__.py"
            if init_file.exists():
                expanded.add(init_file.resolve())
    return expanded


def collect_transitive_python_dependencies(
    repo: Path,
    seeds: Iterable[str],
) -> tuple[set[Path], list[str]]:
    queue: list[Path] = []
    missing: list[str] = []

    for rel in seeds:
        path = (repo / rel).resolve()
        if path.exists():
            queue.append(path)
        else:
            missing.append(rel)

    collected: set[Path] = set()

    while queue:
        path = queue.pop()
        if path in collected:
            continue
        if not is_exportable(repo, path):
            continue
        collected.add(path)
        if path.suffix == ".py":
            for dependency in imported_internal_files(repo, path):
                if dependency not in collected and is_exportable(repo, dependency):
                    queue.append(dependency)

    return add_package_init_files(repo, collected), missing


def collect_results(repo: Path) -> set[Path]:
    result_files: set[Path] = set()
    for rel_dir in RESULT_DIRS:
        directory = repo / rel_dir
        if not directory.exists():
            continue
        for name in RESULT_FILENAMES:
            path = directory / name
            if path.exists() and path.is_file():
                result_files.add(path.resolve())
    return result_files


def copy_preserving_tree(repo: Path, files: Iterable[Path], stage: Path) -> None:
    for source in sorted(set(files)):
        rel = source.resolve().relative_to(repo.resolve())
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_manifest(
    repo: Path,
    stage: Path,
    code_files: set[Path],
    result_files: set[Path],
    missing_seeds: list[str],
) -> None:
    branch = run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    commit = run_git(repo, "rev-parse", "HEAD")
    status = run_git(repo, "status", "--short")

    def rels(paths: Iterable[Path]) -> list[str]:
        return sorted(str(p.resolve().relative_to(repo.resolve())) for p in paths)

    lines = [
        "# C-V15 Related Code Export",
        "",
        f"- Exported at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Repository: `{repo}`",
        f"- Git branch: `{branch}`",
        f"- Git commit: `{commit}`",
        f"- Working tree dirty: `{'yes' if status else 'no'}`",
        f"- Python/source files: `{len(code_files)}`",
        f"- Result JSON files: `{len(result_files)}`",
        "",
        "## Seed files",
        "",
    ]
    lines += [f"- `{item}`" for item in SEED_FILES]

    if missing_seeds:
        lines += [
            "",
            "## Missing seed files",
            "",
            *[f"- `{item}`" for item in missing_seeds],
        ]

    lines += [
        "",
        "## Exported code / internal Python dependencies",
        "",
        *[f"- `{item}`" for item in rels(code_files)],
        "",
        "## Exported C-V15 result files",
        "",
        *([f"- `{item}`" for item in rels(result_files)] or ["- None found"]),
        "",
        "## Exclusions",
        "",
        "- Model checkpoints (`.pt/.pth/.ckpt`)",
        "- Datasets and caches",
        "- External Python packages",
        "- Images/videos/binary artifacts",
        "",
        "The code set is the transitive closure of repository-local Python imports "
        "starting from the C-V15 seed files above.",
        "",
    ]
    (stage / "EXPORT_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def make_zip(stage: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Local repository root or a path inside it. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ZIP path. Default: <repo>/exports/c_v15_all_related_code_<shortsha>.zip",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any C-V15 seed file is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = find_repo_root(args.repo)
    code_files, missing = collect_transitive_python_dependencies(repo, SEED_FILES)
    result_files = collect_results(repo)

    if args.strict and missing:
        print("Missing seed files:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    shortsha = run_git(repo, "rev-parse", "--short", "HEAD")
    if not shortsha or shortsha == "unknown":
        shortsha = "unknown"

    output = (
        args.output.resolve()
        if args.output is not None
        else (repo / "exports" / f"c_v15_all_related_code_{shortsha}.zip").resolve()
    )

    with tempfile.TemporaryDirectory(prefix="cv15_export_") as tmp:
        stage = Path(tmp) / "c_v15_all_related_code"
        stage.mkdir(parents=True, exist_ok=True)
        copy_preserving_tree(repo, code_files | result_files, stage)
        write_manifest(repo, stage, code_files, result_files, missing)
        make_zip(stage, output)

    print(f"Repository : {repo}")
    print(f"Code files : {len(code_files)}")
    print(f"Result JSON: {len(result_files)}")
    print(f"Output ZIP : {output}")

    if missing:
        print("Missing seeds:")
        for item in missing:
            print(f"  - {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
