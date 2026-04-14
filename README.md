# Voltorb Solver

Live-overlay solver and auto-player for Pokémon HeartGold/SoulSilver Voltorb Flip.

## Features

- **Probability solver** — computes per-cell bomb probability and expected value from the row/column clues.
- **Move advisor** — ranks safe tiles by expected value; flags guaranteed-safe cells and deprioritises risky ones.
- **Screen capture** — captures the emulator window (or a saved screenshot) and parses the board automatically.
- **Template-based clue parsing** — reads clue digits via template matching with a pixel-token hole-count tiebreaker.
- **Always-on-top overlay** — draws detected board regions, cell coordinates, and probability hints on top of the emulator window.
- **Auto-play** — automatically clicks the highest-value safe tile using `xdotool`; detects game-clear and game-failed screens to advance rounds.
- **Stats tracking** — tracks lifetime and session win/bomb counts (including solver miscalculations), persisted to `~/.local/share/voltorb-solver/stats.json`.
- **Undo support** — step back through board states within a round.
- **X11 only** — overlay and auto-click require an X11 session (`xdotool`, `xwininfo`, `wmctrl`, `xprop`).

## System Dependencies

The following tools must be installed separately (Python packages alone are not sufficient):

```bash
# Debian/Ubuntu/Mint
sudo apt install x11-utils xdotool wmctrl
```

| Tool | Required for |
|------|-------------|
| `xwininfo` (x11-utils) | Emulator window selection and geometry |
| `xdotool` | Auto-clicking tiles in the emulator |
| `wmctrl` | Window focus before clicking (optional fallback) |
| `xprop` (x11-utils) | Reading emulator window class for identification |

## Run

```bash
python -m pip install -e .
python -m voltorb_solver.main
```

To annotate a screenshot directly from the CLI:

```bash
python -m voltorb_solver.image_import.screen_parser assets/GameBoard.png labeled.png
```

## Test

```bash
python -m pip install -e .[dev]
pytest -q
```

## Notes

- If game panel detection fails, the parser falls back to panel-relative defaults and reports warnings.
- Template coverage is incremental: unmatched clue fields are saved to `assets/templates/raw/clue_unknown/` for manual labelling and re-training.
