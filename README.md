# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jekojek/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jekojek/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5/25/50/75/95%) per trophy. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

Exophase refreshes its counts in jumps, not continuously: when a refresh lands, the tracked-player count leaps while earners barely move, so every rate is diluted by roughly the same factor. The program therefore models the two counts apart and divides again at the target:

- **Players tracked N(t)** — grows close to a power law in log time (launch night: 638 → 3,868 → ~13,300 players, slope ≈ 1.05). The fit comes from the case's own distinct snapshots, bracketed by a recent-pace and a decaying-growth scenario.
- **Earners E(t) per trophy** — a log-time fit per trophy, with the slope shrunk toward its group's median (tier + last-rate band), because one trophy's counts are noisy at 2-decimal precision. Story-opening trophies climb fast; hard and platinum trophies barely move.
- **Spread** — built from the N-scenario spread, the E-fit residual, and a term that widens with the horizon, so a 90-minute practice forecast is tight while the 4.5-day final forecast honestly admits how far N and E can wander. Quantiles are taken on the *displayed* rate (rounded to 0.01 points), which matters at the 0.01% floor where most rare trophies land.
- **Fallback** — anything unusable falls back to a persistence forecast, so a case is never left invalid.

Standard library only, deterministic, offline, ~0.1 s per case.

## Validation (local, proxy answers)

Practice answers are sealed server-side, so the proxy answer used here is earners-from-the-last-pre-refresh-read / 3,868 (the post-refresh player count), with the 100% trophy kept at 99.99%:

| check | result |
| --- | --- |
| practice p00 proxy loss (house baseline: 1.69) | **0.055** |
| practice p01 proxy loss (house baseline: 1.69) | **0.044** |
| synthetic final, pure power-law world (persistence: 0.223) | **0.038**, coverage 1.00 |
| synthetic final, growth-decay world (persistence: 0.313) | **0.080**, coverage 1.00 |
| determinism (2 runs) | byte-identical |
| edge cases (1 read, null rates, 0.00/100.00, empty history) | all valid outputs |
| runtime | ~0.07 s (limit 8 s) |

---

<sub>Entered on [make.fun](https://make.fun/?ref=xesesfrd&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
