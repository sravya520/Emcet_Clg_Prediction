# Backtest: do the Safe / Moderate / Reach labels hold up?

Every number on this page was produced by `python -m copilot.engine.backtest`. Nothing here is written by hand.

## The limits

| Band | Ratio range | Meaning |
|---|---|---|
| Safe | up to **0.76** | comfortably inside last year's line |
| Moderate | **0.76 - 1.15** | near the line |
| Reach | **1.15 - 1.51** | past the line, still possible |
| (not shown) | above **1.51** | too far to be worth listing |

Tuned on **2023->2024**. Tested once on **2024->2025 (SC excluded)**.

Each line is drawn where the chance of getting in, for a student sitting right at that line, falls to a set level on the tuning fold: Safe >= 90%, Moderate >= 50%, Reach >= 20%. Measured **at** the line, not averaged over everything below it. That distinction is load-bearing: an earlier version averaged, which pushed the Safe line out to 1.11 and squeezed the Moderate band down to 0.01 wide. Averaging lets a band's comfortable middle hide a weak edge; measuring at the edge does not.

## Fold A - tuning

These numbers are **not evidence**. The limits were fitted to this fold, so a good score here is circular by construction. It is shown for completeness only.

### Fold A (tuning): 2023 -> 2024

- Options matched in both years: **21,216**
- Options dropped (no successor in 2024): **4,155**
- Simulated student ranks: **720** (every 250th rank)
- Student-option pairs the app would have shown: **12,712,395**

| band     | ratio range   |   times shown |   times right | how often right   |
|:---------|:--------------|--------------:|--------------:|:------------------|
| Safe     | 0.00 - 0.76   |       7208266 |       7000825 | 97.1%             |
| Moderate | 0.76 - 1.15   |       3704467 |       2697476 | 72.8%             |
| Reach    | 1.15 - 1.51   |       1799662 |        534553 | 29.7%             |

## Fold B - hold-out (the real result)

This fold was evaluated **once**, after the limits were written to disk. They were not adjusted afterwards.

### Fold B (hold-out): 2024 -> 2025

- Options matched in both years: **17,959**
- Options dropped (no successor in 2025): **4,643**
- Simulated student ranks: **720** (every 250th rank)
- Student-option pairs the app would have shown: **10,479,793**
- Categories excluded: **SC, SC-I, SC-II, SC-III**

| band     | ratio range   |   times shown |   times right | how often right   |
|:---------|:--------------|--------------:|--------------:|:------------------|
| Safe     | 0.00 - 0.76   |       6434923 |       6292751 | 97.8%             |
| Moderate | 0.76 - 1.15   |       2974029 |       2012134 | 67.7%             |
| Reach    | 1.15 - 1.51   |       1070841 |        363020 | 33.9%             |

## What SC students are told

SC is missing from Fold B on purpose. In 2025 the state replaced SC with SC-I / SC-II / SC-III, so a 2024 SC row has no like-for-like successor. Mapping them would mean inventing a correspondence the government never published. The app therefore shows every SC, SC-I, SC-II and SC-III student a warning that their bands are untested.

## Disclosure: the hold-out was run twice

Full transparency, because it affects how much weight this result deserves. The first version of the tuning code measured each band's hit rate as an average over everything below the line. That produced a broken set of limits - a Moderate band 0.01 wide - and the hold-out was evaluated once with them before the flaw was noticed.

The flaw was visible in the tuning fold alone (a band that narrow is obviously wrong), and the fix was derived from the tuning fold's curve, not from the hold-out. But the hold-out is no longer perfectly untouched, and pretending otherwise would be the exact kind of quiet overclaim this project is built to avoid. Treat the hold-out numbers as *one* re-run after a methodology fix, not as a first-ever look.

The next genuinely clean test will be the 2025 -> 2026 fold, once the 2026 statement is published.

## Honest limits of this test

- It assumes an option that existed in both years is the same option. Colleges do change what they offer.
- It excludes special reservation categories (PWD, NCC, Sports, CAP, Scouts & Guides), because the source statements exclude them too.
- Closing ranks are end-of-web-counselling and exclude spot admissions, so real intake goes slightly further than these numbers show.
- It measures whether a rank would have been enough, not whether the student would have been offered or accepted that seat.
