# Voltorb Solver

Live-overlay helper for Pokemon HeartGold/SoulSilver Voltorb Flip.

## Current Status

Overlay-first MVP implemented:

- Always-on-top transparent overlay that can draw detected game regions on your desktop.
- Control window with screen capture and screenshot load actions.
- Screenshot region parser for Voltorb Flip layouts.
- Annotated image output with per-component labels: board tiles `(0,0)` to `(4,4)`, row clues `r0..r4`, and column clues `c0..c4`.

## Run

```bash
python -m pip install -e .
python -m voltorb_solver.main
```

To label one screenshot directly from CLI:

```bash
python -m voltorb_solver.image_import.screen_parser GameBoard.png labeled.png
```

## Test

```bash
python -m pip install -e .[dev]
pytest -q
```

## Notes

- The region parser is geometric and heuristic-based. It does not OCR clue digits yet.
- If game panel detection fails, the parser falls back to panel-relative defaults and reports warnings.
