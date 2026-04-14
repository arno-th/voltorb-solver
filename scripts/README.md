# Developer Scripts

These scripts are one-off development tools for managing and debugging the image templates used by the parser. Run them from the repository root.

---

## `build_templates.py`

Converts raw template screenshots (colour PNG/JPG) into grayscale PNGs ready for use by the parser.

**When to run:** after capturing new raw template images into `assets/templates/raw/` and before running the solver, so the parser picks up the updated templates.

```bash
python scripts/build_templates.py
```

This uses the defaults: reads from `assets/templates/raw/`, writes to `assets/templates/`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input DIR` | `assets/templates/raw` | Source directory of raw screenshots |
| `--output DIR` | `assets/templates` | Destination directory for processed templates |
| `--suffix STR` | *(empty)* | String appended to each output filename before `.png` |

**Example — write to a staging directory with a suffix:**
```bash
python scripts/build_templates.py --input assets/templates/raw --output /tmp/tpl_test --suffix _gray
```

---

## `score_template_match.py`

Scores how well a candidate image matches a template using OpenCV normalised cross-correlation (`TM_CCOEFF_NORMED`). Useful for diagnosing why the parser is failing to match a particular clue or tile crop.

**Requires:** `opencv-python` (already a project dependency).

```bash
python scripts/score_template_match.py --template TEMPLATE --candidate CANDIDATE
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--template PATH` | *(required)* | Template image (e.g. `assets/templates/game_tile_0.png`) |
| `--candidate PATH` | *(required)* | Candidate crop to compare against the template |
| `--mode {same,sliding}` | `same` | `same`: resize candidate to template size then compare. `sliding`: find best-match location inside a larger candidate |
| `--binary` | off | Apply Otsu binarization to both images before matching |
| `--blur` | off | Apply a slight Gaussian blur before matching |

**Score interpretation:**

| Score | Quality |
|-------|---------|
| ≥ 0.92 | excellent |
| ≥ 0.82 | strong |
| ≥ 0.68 | moderate |
| ≥ 0.50 | weak |
| < 0.50 | poor |

The parser's acceptance threshold is **0.85** for clue templates and **0.75/0.78** for tile templates.

**Example — check a clue crop against a known template:**
```bash
python scripts/score_template_match.py \
  --template assets/templates/clue_row_v_2.png \
  --candidate assets/templates/raw/clue_unknown/clue_row_v_<hash>_<timestamp>.png \
  --mode same
```
