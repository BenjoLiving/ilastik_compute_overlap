"""Overlap between two Ilastik classification masks (PNN and PV).

Run it with:   uv run main.py      (or:  python main.py)

Everything you might want to change is in the SETTINGS block just below.
See README.md for what this does and how to read the results.
"""

# =============================================================================
# SETTINGS: change these, then run the script. Nothing else needs editing.
# =============================================================================

# Folder holding the PNN Ilastik output, and folder holding the PV Ilastik output.
# Use a full path (e.g. "/Users/me/data/PNN") or one relative to this file.
PNN_FOLDER = "PNN_PV/PFC_PNN_TrainingCrops"
PV_FOLDER = "PNN_PV/PFC_TrainingCrops"

# Where results go. Each run makes its own new folder inside this one.
OUTPUT_FOLDER = "results"

# A short label for this run. It becomes part of the results folder name.
RUN_NAME = "PNN-vs-PV"

# Object-level analysis only (Method 2):
MIN_SIZE_PX = 0           # ignore objects smaller than this many pixels (0 = keep everything)
MIN_FRAC_OVERLAP = 0.0    # fraction of an object that must be covered to count (0 = any overlap counts)

# Advanced: the Ilastik mask files end with "_<PNN or PV>_" followed by this.
MASK_FILE_ENDING = "Object Predictions.tiff"

# =============================================================================
# END OF SETTINGS
# =============================================================================

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve(path_text):
    """Full paths are used as given; relative ones are relative to this file."""
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else SCRIPT_DIR / path


# ---------------------------------------------------------------- finding files

def index_masks(folder, stain):
    """Return {section_id: path_to_mask} for every mask file in the folder."""
    if not folder.is_dir():
        sys.exit(f"ERROR: the {stain} folder does not exist:\n  {folder}\nCheck the SETTINGS at the top of main.py.")
    ending = f"_{stain}_{MASK_FILE_ENDING}"
    return {f.name.removesuffix(ending): f
            for f in sorted(folder.iterdir())
            if f.name.endswith(ending)}


def load_mask(path):
    """Read an Ilastik mask as True/False (True = stained)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"ERROR: could not read image file:\n  {path}")
    return img > 0


# ------------------------------------------------- Method 1: whole-section overlap

def total_overlap(pnn_mask, pv_mask):
    """Pixel counts and overlap percentages for one pair of masks."""
    pnn_pixels = int(pnn_mask.sum())
    pv_pixels = int(pv_mask.sum())
    shared_pixels = int((pnn_mask & pv_mask).sum())            # stained in BOTH
    combined_pixels = pnn_pixels + pv_pixels - shared_pixels   # stained in EITHER
    return {
        "PNN_pixels": pnn_pixels,
        "PV_pixels": pv_pixels,
        "shared_pixels": shared_pixels,
        "combined_pixels": combined_pixels,
        "pct_PNN_covered_by_PV": 100 * shared_pixels / pnn_pixels if pnn_pixels else np.nan,
        "pct_PV_covered_by_PNN": 100 * shared_pixels / pv_pixels if pv_pixels else np.nan,
        "pct_of_combined_shared": 100 * shared_pixels / combined_pixels if combined_pixels else np.nan,
    }


# -------------------------------------------------- Method 2: object-level overlap

def label_mask(mask):
    """Give every separate blob its own number (1, 2, 3...). 0 is background.
    connectivity=8 means pixels touching diagonally count as the same object."""
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def object_overlap_table(section_id, stain, own, other_mask, other_labels):
    """One row per object in `own`, describing how much the other stain overlaps it."""
    n, labels, stats, centroids = own

    # Pixels where this stain AND the other stain are both on
    both = other_mask & (labels > 0)
    own_ids = labels[both]
    partner_ids = other_labels[both]

    # Overlapping pixels per object (index 0 is background, so drop it)
    overlap_px = np.bincount(own_ids, minlength=n)[1:]

    # How many different objects of the other stain does each object touch?
    pairs = np.unique(np.stack([own_ids, partner_ids]), axis=1)
    n_partners = np.bincount(pairs[0], minlength=n)[1:]

    size_px = stats[1:, cv2.CC_STAT_AREA]
    return pd.DataFrame({
        "section_id": section_id,
        "stain": stain,
        "object_id": np.arange(1, n),
        "size_px": size_px,
        "centroid_x": centroids[1:, 0],
        "centroid_y": centroids[1:, 1],
        "overlap_px": overlap_px,
        "frac_overlapped": overlap_px / size_px,
        "n_partner_objects": n_partners,
    })


def summarize_objects(objects, min_size_px, min_frac_overlap):
    """For each section and stain: how many objects are overlapped by the other stain?"""
    kept = objects[objects["size_px"] >= min_size_px].copy()
    kept["overlapped"] = (kept["overlap_px"] > 0) & (kept["frac_overlapped"] >= min_frac_overlap)
    out = (kept.groupby(["section_id", "stain"])
               .agg(n_objects=("object_id", "count"), n_overlapped=("overlapped", "sum"))
               .reset_index())
    out["pct_overlapped"] = 100 * out["n_overlapped"] / out["n_objects"]
    return out


# ------------------------------------------------------------------ the picture

def overlap_image(pnn_mask, pv_mask):
    """PV goes in the red channel, PNN in the green channel. Both on = yellow."""
    rgb = np.zeros((*pnn_mask.shape, 3), dtype=np.uint8)
    rgb[pv_mask, 0] = 255
    rgb[pnn_mask, 1] = 255
    return rgb


# --------------------------------------------------------------- run record

def sha256_of(path):
    """A fingerprint of the file's contents. If the file changes, the fingerprint changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=SCRIPT_DIR).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # not a git repo, or no commits yet


def write_manifest(run_dir, pnn_index, pv_index, matched_ids):
    rows = []
    for stain, index in (("PNN", pnn_index), ("PV", pv_index)):
        for sid, path in index.items():
            info = path.stat()
            rows.append({
                "section_id": sid,
                "stain": stain,
                "used_in_analysis": sid in matched_ids,  # False = no partner file in the other folder
                "file_path": str(path.resolve()),
                "size_bytes": info.st_size,
                "last_modified": datetime.fromtimestamp(info.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_of(path),
            })
    pd.DataFrame(rows).sort_values(["section_id", "stain"]).to_csv(run_dir / "input_manifest.csv", index=False)


def write_run_info(run_dir, run_time, pnn_folder, pv_folder, matched_ids, only_pnn, only_pv):
    run_info = {
        "run_name": RUN_NAME,
        "run_time": run_time.isoformat(timespec="seconds"),
        "pnn_folder": str(pnn_folder),
        "pv_folder": str(pv_folder),
        "mask_file_ending": MASK_FILE_ENDING,
        "sections_matched": len(matched_ids),
        "sections_only_in_pnn_folder": only_pnn,
        "sections_only_in_pv_folder": only_pv,
        "settings": {
            "object_connectivity": 8,
            "min_size_px": MIN_SIZE_PX,
            "min_frac_overlap": MIN_FRAC_OVERLAP,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "git_commit": git_commit(),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)


def write_readme(run_dir, run_time, pnn_folder, pv_folder, n_sections):
    text = f"""OVERLAP ANALYSIS: {RUN_NAME}
Run on {run_time.isoformat(timespec="seconds")}

WHAT THIS IS
Overlap between two Ilastik classification masks (PNN and PV) for each brain section.
Only sections found in BOTH folders were analysed ({n_sections} sections).

INPUTS (not copied here; see input_manifest.csv for every file and its fingerprint)
  PNN masks: {pnn_folder}
  PV masks : {pv_folder}
  Mask files used: "<section>_<PNN or PV>_{MASK_FILE_ENDING}"

SETTINGS (Method 2 only)
  Objects were found with 8-connectivity (diagonal pixels count as touching).
  MIN_SIZE_PX      = {MIN_SIZE_PX}   (objects smaller than this were ignored in object_overlap_summary.csv)
  MIN_FRAC_OVERLAP = {MIN_FRAC_OVERLAP}   (0 = an object counts as overlapped if ANY of it is covered)

FILES
  overlap_summary.csv           Method 1: whole-section overlap, one row per section
  object_overlap_summary.csv    Method 2: how many objects are overlapped, per section and stain
  object_overlap/               Method 2 detail: one CSV per section, one row per object (NOT thresholded)
  overlap_images/               One picture per section: red = PV, green = PNN, yellow = both
  input_manifest.csv            Every input file used, with size, date and SHA-256 fingerprint
  run_info.json                 Settings, software versions and folder paths for this run
  main_snapshot.py              Copy of the script (with its settings) that made these results

METHOD 1 COLUMNS (overlap_summary.csv)
  pct_PNN_covered_by_PV    Of all the PNN-stained area, the % that is also PV-stained
  pct_PV_covered_by_PNN    Of all the PV-stained area, the % that is also PNN-stained
  pct_of_combined_shared   Of the area stained by EITHER, the % stained by BOTH
  PNN_pixels, PV_pixels    Number of stained pixels in each mask
  shared_pixels            Pixels stained in both masks
  combined_pixels          Pixels stained in either mask

METHOD 2 COLUMNS (object_overlap/ files)
  stain                    Which stain this object belongs to (PNN or PV)
  object_id                Number given to the object within its mask (not Ilastik's own ID)
  size_px                  Size of the object in pixels
  centroid_x, centroid_y   Centre of the object in the image (pixels)
  overlap_px               Pixels of this object that are covered by the OTHER stain
  frac_overlapped          overlap_px / size_px (0 = none covered, 1 = fully covered)
  n_partner_objects        How many objects of the OTHER stain touch this object

METHOD 2 COLUMNS (object_overlap_summary.csv)
  n_objects                Objects of this stain kept after the size threshold
  n_overlapped             How many of them are overlapped by the other stain
  pct_overlapped           n_overlapped / n_objects, as a percentage

NOTES
  The two Method 2 directions are not mirror images: a stain's objects are compared
  against the other stain, so PNN and PV can give different percentages.
  Object counts can differ slightly from Ilastik's own tables because objects are
  re-found from the exported mask.
"""
    (run_dir / "README.txt").write_text(text)


# ------------------------------------------------------------------------ main

def main():
    run_time = datetime.now()
    pnn_folder = resolve(PNN_FOLDER)
    pv_folder = resolve(PV_FOLDER)

    # 1. Find the mask files in each folder and match them up by section
    pnn_index = index_masks(pnn_folder, "PNN")
    pv_index = index_masks(pv_folder, "PV")
    matched_ids = sorted(set(pnn_index) & set(pv_index))
    only_pnn = sorted(set(pnn_index) - set(pv_index))
    only_pv = sorted(set(pv_index) - set(pnn_index))

    print(f"PNN masks: {len(pnn_index)} | PV masks: {len(pv_index)} | Matched: {len(matched_ids)}")
    if only_pnn:
        print("WARNING: only in the PNN folder (skipped):", *only_pnn, sep="\n  ")
    if only_pv:
        print("WARNING: only in the PV folder (skipped):", *only_pv, sep="\n  ")
    if not matched_ids:
        sys.exit("ERROR: no sections matched between the two folders. Nothing to do.")

    # 2. Make a new folder for this run (never overwrites an earlier one)
    run_dir = resolve(OUTPUT_FOLDER) / f"{run_time:%Y-%m-%d_%H%M}_{RUN_NAME}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        sys.exit(f"ERROR: this run folder already exists (re-run in a minute or change RUN_NAME):\n  {run_dir}")
    (run_dir / "object_overlap").mkdir()
    (run_dir / "overlap_images").mkdir()

    # 3. Analyse one section at a time
    method1_rows = []
    object_tables = []
    for i, sid in enumerate(matched_ids, start=1):
        print(f"[{i}/{len(matched_ids)}] {sid}")
        pnn_mask = load_mask(pnn_index[sid])
        pv_mask = load_mask(pv_index[sid])
        if pnn_mask.shape != pv_mask.shape:
            sys.exit(f"ERROR: the PNN and PV masks for {sid} are different sizes "
                     f"({pnn_mask.shape} vs {pv_mask.shape}), so they cannot be compared.")

        method1_rows.append({"section_id": sid, **total_overlap(pnn_mask, pv_mask)})

        pnn_labelled = label_mask(pnn_mask)
        pv_labelled = label_mask(pv_mask)
        table = pd.concat([
            object_overlap_table(sid, "PNN", pnn_labelled, pv_mask, pv_labelled[1]),
            object_overlap_table(sid, "PV", pv_labelled, pnn_mask, pnn_labelled[1]),
        ], ignore_index=True)
        table.to_csv(run_dir / "object_overlap" / f"{sid}_objects.csv", index=False)
        object_tables.append(table)

        cv2.imwrite(str(run_dir / "overlap_images" / f"{sid}_overlap.png"),
                    cv2.cvtColor(overlap_image(pnn_mask, pv_mask), cv2.COLOR_RGB2BGR))  # OpenCV saves BGR

    # 4. Summary tables
    pd.DataFrame(method1_rows).to_csv(run_dir / "overlap_summary.csv", index=False)
    all_objects = pd.concat(object_tables, ignore_index=True)
    summarize_objects(all_objects, MIN_SIZE_PX, MIN_FRAC_OVERLAP).to_csv(
        run_dir / "object_overlap_summary.csv", index=False)

    # 5. Record exactly what was done
    write_manifest(run_dir, pnn_index, pv_index, matched_ids)
    write_run_info(run_dir, run_time, pnn_folder, pv_folder, matched_ids, only_pnn, only_pv)
    shutil.copy2(Path(__file__).resolve(), run_dir / "main_snapshot.py")
    write_readme(run_dir, run_time, pnn_folder, pv_folder, len(matched_ids))

    print(f"\nDone. Results saved to:\n  {run_dir}")


if __name__ == "__main__":
    main()
