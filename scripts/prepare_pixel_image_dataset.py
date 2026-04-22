#!/usr/bin/env python3
"""Prepare a persistent pixel-space class-folder dataset.

The output layout is always:
  OUTPUT_ROOT/train/<class_name>/<image files>

It accepts either a zip archive or an already-extracted directory. Archive
metadata and macOS files are removed, unsupported files are dropped, and common
one-folder archive wrappers are flattened.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAC_FILENAMES = {".DS_Store", "Thumbs.db"}
MAC_DIRNAMES = {"__MACOSX"}


def _is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _clean_tree(root):
    removed = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.name in MAC_FILENAMES or path.name.startswith("._"):
            path.unlink(missing_ok=True)
            removed += 1
        elif path.is_dir() and path.name in MAC_DIRNAMES:
            shutil.rmtree(path)
            removed += 1
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file() and not _is_image(path):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _unwrap_single_dir(root):
    source = root
    while True:
        entries = [p for p in source.iterdir() if not p.name.startswith(".")]
        if len(entries) != 1 or not entries[0].is_dir():
            return source
        source = entries[0]


def _select_source_root(extracted_root):
    source = _unwrap_single_dir(extracted_root)
    if (source / "EuroSAT").is_dir() and (source / "EuroSATallBands").is_dir():
        source = source / "EuroSAT"
    return source


def _copy_images(class_dir, output_class_dir):
    output_class_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(p for p in class_dir.rglob("*") if _is_image(p)):
        rel = image_path.relative_to(class_dir)
        if len(rel.parts) > 1:
            target_name = "__".join(rel.parts)
        else:
            target_name = image_path.name
        target = output_class_dir / target_name
        stem = target.stem
        suffix = target.suffix
        dedupe_idx = 1
        while target.exists():
            target = output_class_dir / f"{stem}_{dedupe_idx}{suffix}"
            dedupe_idx += 1
        shutil.copy2(image_path, target)
        count += 1
    return count


def _normalize_to_train(source_root, output_root, overwrite=False):
    train_out = output_root / "train"
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    train_out.mkdir(parents=True, exist_ok=True)

    if (source_root / "train").is_dir():
        class_parent = source_root / "train"
    else:
        class_parent = source_root

    class_dirs = [p for p in sorted(class_parent.iterdir()) if p.is_dir()]
    counts = {}
    if class_dirs:
        for class_dir in class_dirs:
            count = _copy_images(class_dir, train_out / class_dir.name)
            if count > 0:
                counts[class_dir.name] = count
    else:
        default_dir = train_out / "default"
        count = _copy_images(class_parent, default_dir)
        if count > 0:
            counts["default"] = count

    for class_dir in list(train_out.iterdir()):
        if class_dir.is_dir() and not any(_is_image(p) for p in class_dir.rglob("*")):
            shutil.rmtree(class_dir)

    if not counts:
        raise ValueError(f"No image files found under {source_root}")
    return counts


def _extract_zip(zip_path, extract_root):
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = Path(member.filename)
            if any(part in MAC_DIRNAMES for part in name.parts):
                continue
            if name.name in MAC_FILENAMES or name.name.startswith("._"):
                continue
            zf.extract(member, extract_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Dataset zip or extracted directory.")
    parser.add_argument("output_root", help="Persistent output dataset root.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with tempfile.TemporaryDirectory(prefix="imeanflow_pixel_data_") as tmp:
        tmp_root = Path(tmp)
        if input_path.is_file():
            if input_path.suffix.lower() != ".zip":
                raise ValueError(f"Only .zip files are supported as archive inputs: {input_path}")
            extract_root = tmp_root / "extracted"
            extract_root.mkdir()
            print(f"Unzipping {input_path}...")
            _extract_zip(input_path, extract_root)
            removed = _clean_tree(extract_root)
            source_root = _select_source_root(extract_root)
        else:
            removed = _clean_tree(input_path)
            source_root = _select_source_root(input_path)

        counts = _normalize_to_train(source_root, output_root, overwrite=args.overwrite)

    print(f"Output root: {output_root}")
    print(f"Removed unsupported/archive metadata files: {removed}")
    print(f"Classes: {len(counts)}")
    print(f"Images: {sum(counts.values())}")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
