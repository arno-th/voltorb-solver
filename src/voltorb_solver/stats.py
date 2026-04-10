from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_STATS_PATH = Path.home() / ".local" / "share" / "voltorb-solver" / "stats.json"


@dataclass
class RoundCounts:
    wins: int = 0
    bombs_hit: int = 0
    miscalc_bombs: int = 0  # subset of bombs_hit where solver thought the tile was safe (P=0%)

    @property
    def rounds_played(self) -> int:
        return self.wins + self.bombs_hit

    @property
    def unlucky_bombs(self) -> int:
        return self.bombs_hit - self.miscalc_bombs

    @property
    def win_rate(self) -> float:
        if self.rounds_played == 0:
            return 0.0
        return self.wins / self.rounds_played


class StatsManager:
    """Tracks lifetime (persisted) and session (in-memory) game statistics."""

    def __init__(self) -> None:
        self.lifetime = RoundCounts()
        self.session = RoundCounts()
        self._load()

    def record_win(self) -> None:
        self.lifetime.wins += 1
        self.session.wins += 1
        self._save()

    def record_bomb(self, *, is_miscalc: bool = False) -> None:
        self.lifetime.bombs_hit += 1
        self.session.bombs_hit += 1
        if is_miscalc:
            self.lifetime.miscalc_bombs += 1
            self.session.miscalc_bombs += 1
        self._save()

    def _load(self) -> None:
        try:
            data = json.loads(_STATS_PATH.read_text())
            lt = data.get("lifetime", {})
            self.lifetime = RoundCounts(
                wins=int(lt.get("wins", 0)),
                bombs_hit=int(lt.get("bombs_hit", 0)),
                miscalc_bombs=int(lt.get("miscalc_bombs", 0)),
            )
        except Exception:
            pass

    def _save(self) -> None:
        try:
            _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STATS_PATH.write_text(json.dumps({"lifetime": asdict(self.lifetime)}, indent=2))
        except Exception:
            pass
