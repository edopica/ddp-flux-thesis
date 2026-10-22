"""
Phase 2 static-scaling prepared-dataset builder (PIL + numpy only).

Decodes Tiny ImageNet-200 exactly once into the frozen uint8 NCHW arrays and
computes the OD-5 normalization constants (stick figure below).

Frozen rules implemented (specification.md 3.1-3.2; OD-5):
  * every JPEG is decoded once with PIL: Image.open(path).convert("RGB");
  * train canonical order: classes in wnids.txt order, filenames sorted inside
    each train/<wnid>/images/ directory (Python sorted(), Unicode code point
    order); class index = position in wnids.txt (0-based);
  * val order: sorted filenames in val/images/, labels from
    val/val_annotations.txt;
  * stats: float64 per-image channel sums and sums-of-squares over the decoded
    [0,1] training images, accumulated sequentially in canonical order (no
    parallel reduction), before augmentation/padding/tail dropping; mean = S/N,
    std = sqrt(max(0, E[x^2] - mean^2)); stored as float32. Validation and test
    images are never used.

Outputs in --out-dir:
  train_images.npy uint8 (100000, 3, 64, 64) C-order
  train_labels.npy int64 (100000,)
  val_images.npy   uint8 (10000, 3, 64, 64) C-order
  val_labels.npy   int64 (10000,)
  class_wnids.txt  byte-identical copy of the source wnids.txt
  stats.json       {"mean": [3], "std": [3], "pixel_count_per_channel": ..., ...}
  dataset_manifest.json  per-file sha256, decoder identity/version, generator
                         info, source commit/state, canonical-order description

Refuses to overwrite an existing dataset_manifest.json unless --force.

Reproduction (Leonardo compute node; the pinned venv has PIL):
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/prepare_dataset.py \\
    --dataset-root /leonardo_work/IscrC_UNMASKED/epicazio/datasets/tiny-imagenet-200 \\
    --out-dir /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared

Deterministic stdout lines: DATASET_PREP <key>=<value>.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

import dataset_adapter as da

PNG_HACK_NOTE = (
    "Tiny ImageNet train files carry a .JPEG extension but are PNG-encoded; "
    "PIL.Image.open sniffs the container, so the recorded decoder identity is "
    "PIL.Image.open(path).convert('RGB') regardless of the file extension."
)

CANONICAL_TRAIN_ORDER = (
    "classes in wnids.txt order (class index = 0-based position), then files "
    "sorted with Python sorted() (Unicode code point order) inside "
    "train/<wnid>/images/ filtered to *.jpeg/*.JPEG case-insensitively"
)
CANONICAL_VAL_ORDER = (
    "sorted filenames (Python sorted(), Unicode code point order) in "
    "val/images/, label from val/val_annotations.txt"
)


def _jpeg_files(directory: str):
    names = [n for n in os.listdir(directory) if n.lower().endswith(".jpeg")]
    return sorted(names)


def _read_wnids(dataset_root: str):
    path = os.path.join(dataset_root, "wnids.txt")
    with open(path, encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]
    if len(wnids) != da.N_CLASSES:
        raise SystemExit("wnids.txt has %d entries, expected %d" % (len(wnids), da.N_CLASSES))
    if len(set(wnids)) != da.N_CLASSES:
        raise SystemExit("wnids.txt entries are not unique")
    return wnids


def _validate_listing(dataset_root: str, wnids):
    """Counts before decoding: 500 files per train class, 10000 val files."""
    for c, wnid in enumerate(wnids):
        d = os.path.join(dataset_root, "train", wnid, "images")
        n = len(_jpeg_files(d))
        if n != 500:
            raise SystemExit("train class %s has %d files, expected 500" % (wnid, n))
    val_dir = os.path.join(dataset_root, "val", "images")
    val_files = _jpeg_files(val_dir)
    if len(val_files) != da.N_VAL:
        raise SystemExit("val/images has %d files, expected %d" % (len(val_files), da.N_VAL))
    return val_files


def _read_val_annotations(dataset_root: str, wnids):
    path = os.path.join(dataset_root, "val", "val_annotations.txt")
    class_index = {w: i for i, w in enumerate(wnids)}
    annot = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise SystemExit("val_annotations.txt line %d malformed: %r" % (lineno, line))
            fname, wnid = parts[0].strip(), parts[1].strip()
            if wnid not in class_index:
                raise SystemExit("val_annotations.txt line %d unknown wnid %r" % (lineno, wnid))
            annot[fname] = class_index[wnid]
    if len(annot) != da.N_VAL:
        raise SystemExit("val_annotations.txt has %d entries, expected %d" % (len(annot), da.N_VAL))
    return annot


def _decode(path: str, pil_image, formats=None):
    img = pil_image.open(path)
    fmt = img.format or "unknown"
    if formats is not None:
        formats[fmt] = formats.get(fmt, 0) + 1
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if arr.shape != (64, 64, 3):
        raise SystemExit("decoded %s to shape %s, expected (64, 64, 3)" % (path, arr.shape))
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def build_train(dataset_root: str, out_dir: str, wnids, pil_image):
    images = np.lib.format.open_memmap(
        os.path.join(out_dir, "train_images.npy"), mode="w+",
        dtype=np.uint8, shape=(da.N_TRAIN, 3, 64, 64))
    labels = np.empty(da.N_TRAIN, dtype=np.int64)
    sum64 = np.zeros(3, dtype=np.float64)
    sumsq64 = np.zeros(3, dtype=np.float64)
    formats = {}
    idx = 0
    for c, wnid in enumerate(wnids):
        d = os.path.join(dataset_root, "train", wnid, "images")
        for fname in _jpeg_files(d):
            chw = _decode(os.path.join(d, fname), pil_image, formats)
            images[idx] = chw
            labels[idx] = c
            x = chw.astype(np.float64) / 255.0
            sum64 += x.sum(axis=(1, 2))
            sumsq64 += (x * x).sum(axis=(1, 2))
            idx += 1
            if idx % 10000 == 0:
                print("DATASET_PREP train_decoded=%d" % idx, flush=True)
    if idx != da.N_TRAIN:
        raise SystemExit("decoded %d train images, expected %d" % (idx, da.N_TRAIN))
    images.flush()
    del images
    np.save(os.path.join(out_dir, "train_labels.npy"), labels)
    pixel_count = da.N_TRAIN * 64 * 64
    mean64 = sum64 / float(pixel_count)
    var64 = sumsq64 / float(pixel_count) - mean64 * mean64
    std64 = np.sqrt(np.maximum(var64, 0.0))
    mean32 = mean64.astype(np.float32)
    std32 = std64.astype(np.float32)
    print("DATASET_PREP train_mean=%s" % np.array2string(mean32, precision=9), flush=True)
    print("DATASET_PREP train_std=%s" % np.array2string(std32, precision=9), flush=True)
    return mean32, std32, int(pixel_count), mean64, std64, formats


def build_val(dataset_root: str, out_dir: str, val_files, annot, pil_image):
    images = np.lib.format.open_memmap(
        os.path.join(out_dir, "val_images.npy"), mode="w+",
        dtype=np.uint8, shape=(da.N_VAL, 3, 64, 64))
    labels = np.empty(da.N_VAL, dtype=np.int64)
    formats = {}
    missing = [f for f in val_files if f not in annot]
    if missing:
        raise SystemExit("val files without annotations: %s" % missing[:5])
    for i, fname in enumerate(val_files):
        images[i] = _decode(os.path.join(dataset_root, "val", "images", fname), pil_image, formats)
        labels[i] = annot[fname]
        if (i + 1) % 5000 == 0:
            print("DATASET_PREP val_decoded=%d" % (i + 1), flush=True)
    images.flush()
    del images
    np.save(os.path.join(out_dir, "val_labels.npy"), labels)
    return labels, formats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the frozen uint8 prepared dataset and OD-5 statistics")
    p.add_argument("--dataset-root", required=True, help="extracted tiny-imagenet-200 root (wnids.txt, train/, val/)")
    p.add_argument("--out-dir", required=True, help="prepared dataset directory ($PREP)")
    p.add_argument("--force", action="store_true", help="overwrite an existing dataset_manifest.json")
    p.add_argument("--source-commit", default=None)
    p.add_argument("--source-state-sha256", default=None)
    args = p.parse_args(argv)

    from PIL import Image  # lazy: keeps pure-numpy modules importable locally

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "dataset_manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        raise SystemExit("refusing to overwrite %s (use --force)" % manifest_path)

    wnids = _read_wnids(args.dataset_root)
    val_files = _validate_listing(args.dataset_root, wnids)
    annot = _read_val_annotations(args.dataset_root, wnids)
    shutil.copyfile(os.path.join(args.dataset_root, "wnids.txt"),
                    os.path.join(args.out_dir, "class_wnids.txt"))

    image_mod = Image
    print("DATASET_PREP decoder=PIL.Image.open.convert(RGB) pillow_version=%s" % getattr(image_mod, "__version__", "unknown"), flush=True)
    mean32, std32, pixel_count, mean64, std64, train_formats = build_train(args.dataset_root, args.out_dir, wnids, image_mod)
    _, val_formats = build_val(args.dataset_root, args.out_dir, val_files, annot, image_mod)
    print("DATASET_PREP containers_train=%s containers_val=%s"
          % (sorted(train_formats.items()), sorted(val_formats.items())), flush=True)

    stats = {
        "mean": [float(x) for x in mean32],
        "std": [float(x) for x in std32],
        "pixel_count_per_channel": pixel_count,
        "dtype": "float32",
        "method": (
            "per-channel RGB mean and population standard deviation (divide by N) from all "
            "100000 original training images, decoded once with PIL Image.open(path).convert('RGB'), "
            "scaled to [0,1], float64 accumulation, computed before augmentation/padding/tail dropping; "
            "validation and test images never used; mean = S/N; std = sqrt(max(0, E[x^2] - mean^2)); "
            "constants stored as float32"
        ),
        "reduction": (
            "per-image float64 channel sums and sums-of-squares accumulated sequentially in canonical "
            "train order (wnids.txt order, sorted filenames); no parallel reduction"
        ),
        "decimal_mean_f64": [float(x) for x in mean64],
        "decimal_std_f64": [float(x) for x in std64],
        "canonical_train_order": CANONICAL_TRAIN_ORDER,
    }
    da.write_json(os.path.join(args.out_dir, "stats.json"), stats)

    files = []
    for name in ("train_images.npy", "train_labels.npy", "val_images.npy", "val_labels.npy",
                 "class_wnids.txt", "stats.json"):
        path = os.path.join(args.out_dir, name)
        entry = {"path": name, "bytes": os.path.getsize(path), "sha256": da.sha256_file(path)}
        if name.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            entry["dtype"] = str(arr.dtype)
            entry["shape"] = [int(x) for x in arr.shape]
            del arr
        files.append(entry)

    manifest = {
        "producer": "scripts/static_scaling/prepare_dataset.py",
        "created": da.utc_now(),
        "synthetic": False,
        **da.source_info(args.source_commit, args.source_state_sha256),
        "dataset": {
            "name": "tiny-imagenet-200",
            "root": os.path.abspath(args.dataset_root),
            "classes": da.N_CLASSES,
            "train_images": da.N_TRAIN,
            "val_images": da.N_VAL,
            "image_shape_chw": [3, 64, 64],
            "source_url": "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
            "expected_archive_sha256": "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821",
            "archive_note": "archive bytes are verified at download time (2026-10-15 record); this run re-verified layout and counts only",
        },
        "decoder": {
            "identity": "PIL.Image.open(path).convert('RGB')",
            "library": "Pillow",
            "version": getattr(image_mod, "__version__", "unknown"),
            "formats_train": {k: int(v) for k, v in sorted(train_formats.items())},
            "formats_val": {k: int(v) for k, v in sorted(val_formats.items())},
            "note": PNG_HACK_NOTE if set(train_formats) == {"PNG"} else (
                "detected container formats are recorded in formats_train/formats_val; "
                "PIL.Image.open sniffs the container, so the .JPEG extension is not authoritative"),
        },
        "generator": {"library": "numpy", "version": np.__version__, "module": "scripts/static_scaling/prepare_dataset.py"},
        "canonical_train_order": CANONICAL_TRAIN_ORDER,
        "canonical_val_order": CANONICAL_VAL_ORDER,
        "stats": stats,
        "files": files,
    }
    da.write_json(manifest_path, manifest)
    print("DATASET_PREP wrote %s files=%d pixel_count_per_channel=%d" % (manifest_path, len(files), pixel_count))
    print("DATASET_PREP DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
