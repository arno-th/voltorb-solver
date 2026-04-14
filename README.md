# Voltorb Solver

Live-overlay solver and auto-player for Pokémon HeartGold/SoulSilver Voltorb Flip.
This is a proof-of-concept app only, built almost solely through Github Copilot. It was built to test the limits of AI paired programming.

## Demo

<video src="assets/demo.mp4" controls width="800">Demo of Voltorb Solver overlay running on top of a Pokémon HeartGold emulator, showing automatic board detection, probability hints, and auto-play advancing rounds.</video>

The overlay attaches to the emulator window, detects the board via screen capture, and highlights the recommended tile in real time. Auto-play advances the gamestate by clicking on the current best tile.

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

## Requirements

Python 3.11 or later.

## Run

```bash
python -m pip install -e .
python -m voltorb_solver.main
```

## Usage

**Manual mode** — use the overlay as a hint panel: select a tile on the board widget or click a cell in the emulator window, and the advisor highlights the recommended next move based on current probabilities. You advance the game yourself.

**Auto-play mode** — enable *Auto-play* in the control panel. After each successful parse the app automatically clicks the highest-value safe tile via `xdotool`, detects game-clear and game-failed textboxes, and advances rounds without manual input.

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
- `opencv-python` and `pynput` are optional at runtime. Without `opencv-python`, screen capture and automatic board detection are unavailable (manual clue entry still works). Without `pynput`, the global hotkey for triggering a capture is unavailable.
