# Ilastik mask overlap (PNN vs PV)

## What it does

You stained brain sections for two things (PNN and PV) and classified each one with Ilastik. This program takes the two folders of Ilastik output, pairs up the files for each brain section, and measures **how much the two stains overlap**. It saves the numbers as spreadsheets (CSV files), plus a picture of the overlap for every section.

It looks at overlap in two ways:

1. **Whole section:** of all the PNN-stained area, how much is also PV-stained? (and the other way round)
2. **Cell by cell:** for every individual PNN or PV object, is any of it covered by the other stain? You can ignore tiny objects if you only care about bigger ones.

Your original files are never changed or copied.

## What you need

- The two folders of Ilastik output. For each section, the program uses the file ending in `Object Predictions.tiff`. The PNN and PV files for the same section must have the same name apart from `PNN` / `PV`, and must be the same image size.
- [uv](https://docs.astral.sh/uv/), which sets up Python and the right packages for you.

## How to use it

1. Open `main.py` and edit the **SETTINGS** block at the very top:
   - `PNN_FOLDER` and `PV_FOLDER`: where your two folders are.
   - `RUN_NAME`: a short label for this run, e.g. `"cohort2"`.
   - `MIN_SIZE_PX` and `MIN_FRAC_OVERLAP`: optional filters for the cell-by-cell results. The defaults mean no filtering.
2. In a terminal, in this project folder, run:

   ```
   uv run main.py
   ```

3. Your results appear in `results/<date>_<time>_<run name>/`.

Every run makes a **new** folder, so earlier results are never overwritten. If the program says a section has no partner, a file is missing or named differently in one of the folders.

## What you get

Inside each results folder:

| File | What it is |
|---|---|
| `overlap_summary.csv` | Whole-section overlap, one row per section |
| `object_overlap_summary.csv` | Cell-by-cell: how many objects are overlapped, per section and stain |
| `object_overlap/` | Every individual object, one CSV per section |
| `overlap_images/` | One picture per section: **red** = PV, **green** = PNN, **yellow** = both |
| `README.txt` | Explains every column in the CSVs |
| `input_manifest.csv`, `run_info.json`, `main_snapshot.py` | A record of exactly which files, settings and code made these results, so you can trust them months later |

### Reading the whole-section results

| Column | Meaning |
|---|---|
| `pct_PNN_covered_by_PV` | Of all the PNN-stained area, the % that is also PV-stained |
| `pct_PV_covered_by_PNN` | Of all the PV-stained area, the % that is also PNN-stained |
| `pct_of_combined_shared` | Of the area stained by *either*, the % stained by *both* |

These are different questions, so pick the one that matches yours. The stain with less area will usually show a higher "covered" percentage.

## Good to know

- The program finds objects itself from the masks, so counts can differ slightly from Ilastik's own tables.
- `overlap_walkthrough.ipynb` is the same analysis explained step by step, if you want to see how it works.
