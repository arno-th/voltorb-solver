# Voltorb Solver

Desktop helper for Pokemon HeartGold/SoulSilver Voltorb Flip.

## Current Status

Initial MVP implemented:

- Manual row/column clue input.
- Manual tile reveal/update.
- Bomb probability per tile (`P(Voltorb)`).
- Per-tile value distribution (`P(0..3)`).
- Two move recommendation lists:
  - Safest moves (lowest bomb probability).
  - Best expected-value moves.
- Auto recalculation on any state update.
- Image upload + manual crop + OCR attempt for clue import (with warnings and manual correction support).

## Run

```bash
python -m pip install -e .
python -m voltorb_solver.main
```

## Test

```bash
python -m pip install -e .[dev]
pytest -q
```

## Notes

- OCR quality depends on local Tesseract installation and screenshot clarity.
- Linux install example: `sudo apt install tesseract-ocr`.
- If the app cannot find Tesseract, set `TESSERACT_CMD` to the executable path (for example `export TESSERACT_CMD=/usr/bin/tesseract`).
- If OCR fails, you can still use full manual entry.
