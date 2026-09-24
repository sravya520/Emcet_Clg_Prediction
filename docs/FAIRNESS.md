# Fairness and reliability audit

Does this system work as well for every group, or only on average?

An overall accuracy figure can hide a group the system serves badly. This
audit re-runs the same **2024 → 2025 hold-out** used for the headline numbers,
split by category, by gender and by both, and reports the **sample size beside
every figure** so a small-sample result is never mistaken for a reliable one.

Reproduce it:

```bash
python -m copilot.engine.fairness    # accuracy, coverage, option counts, tail
python -m copilot.tone_check         # the AI language check (needs a key)
pytest                               # repeatability and tail tests
```

Raw numbers: [`data/mappings/fairness_results.json`](../data/mappings/fairness_results.json).
**Overall figures for comparison: Safe 97.8%, Moderate 67.7%, Reach 33.9%.**

---

## 1. Accuracy by category

| Category | Safe | Moderate | Reach | smallest band sample |
|---|---|---|---|---|
| BC-A | 98.0% | 66.9% | 30.1% | 135,413 |
| BC-B | 98.4% | 69.4% | 33.0% | 124,646 |
| **BC-C** | **93.9%** | 68.4% | **48.4%** | 81,280 |
| BC-D | 98.4% | 68.5% | 34.2% | 134,032 |
| BC-E | 97.0% | **62.5%** | 34.7% | 160,155 |
| OC | 97.9% | 66.4% | **27.3%** | 208,782 |
| OC-EWS | 99.0% | **75.8%** | 34.5% | 114,725 |
| ST | 97.6% | 67.5% | **39.2%** | 111,808 |
| **SC, SC-I, SC-II, SC-III** | **cannot be measured** | — | — | **0** |

### Plainly: which groups are worse

**Safe holds up everywhere except BC-C.** Seven of eight categories land between
97.0% and 99.0%. **BC-C is 93.9%, about 4 points below everyone else.** A BC-C
student told "Safe" is meaningfully more likely to be disappointed than an OC or
BC-B student told the same word. That is the clearest unfairness this audit
found.

**BC-E's Moderate band is 5.2 points below the overall figure** (62.5% vs 67.7%).
Smaller than the BC-C gap but real, on a large sample.

**Reach varies most, and in both directions.** OC is 27.3%, ST is 39.2%, BC-C is
48.4%. Reach is *meant* to be a low number — it means "a gamble" — so BC-C at
48.4% is not good news: it means BC-C students are being told something is a
long shot when it is closer to a coin flip. The label is miscalibrated for them,
just in the direction that understates their chances.

**OC-EWS's Moderate is 8.1 points above overall** (75.8%). Same kind of problem
in reverse: "Moderate" promises less than it delivers for this group.

**SC cannot be measured at all.** Not "was not measured" — cannot be. In 2025 the
state replaced SC with SC-I / SC-II / SC-III, so a 2024 SC row has no
like-for-like successor and the hold-out contains **zero** SC pairs. Every SC
answer in the app already carries a warning saying its bands are untested. That
remains the single largest hole in this project, and it cannot be closed until
the 2026 statement lets SC-I / SC-II / SC-III be compared with themselves.

## 2. Accuracy by gender

| Gender | Safe | Moderate | Reach |
|---|---|---|---|
| Boys | 97.6% | 67.2% | 33.2% |
| Girls | 97.9% | 68.1% | 34.6% |

**No meaningful gap.** Girls are marginally better served on all three bands,
by under a point — well inside noise at these sample sizes. Girls also get
**more options** (see §4), which is the official girls-may-take-boys-seats rule
working as intended rather than a bias: boys genuinely cannot take girls' seats.

Full category × gender breakdown is in the JSON; no combination showed a gap
that the category alone did not already explain.

## 3. Would per-category thresholds fix this?

Measured, not guessed. Each category was re-tuned **on the tuning fold only**
(2023 → 2024), using the identical procedure that produced the shared limits.

| Category | tuning pairs | t_safe | t_moderate | t_max | vs shared (0.76 / 1.15 / 1.51) |
|---|---|---|---|---|---|
| BC-A | 1,782,720 | 0.83 | 1.16 | 1.53 | +0.07 / +0.01 / +0.02 |
| BC-B | 1,777,680 | 0.86 | 1.16 | 1.54 | +0.10 / +0.01 / +0.03 |
| **BC-C** | 1,640,880 | **0.60** | **1.04** | 1.59 | **−0.16 / −0.11** / +0.08 |
| BC-D | 1,775,520 | 0.77 | 1.16 | 1.49 | +0.01 / +0.01 / −0.02 |
| BC-E | 1,749,600 | 0.75 | 1.13 | 1.60 | −0.01 / −0.02 / +0.09 |
| **OC** | 1,715,760 | **0.61** | **0.99** | **1.22** | **−0.15 / −0.16 / −0.29** |
| OC-EWS | 1,259,280 | 0.82 | 1.14 | 1.44 | +0.06 / −0.01 / −0.07 |
| SC | 1,800,720 | 0.94 | 1.18 | 1.51 | +0.18 / +0.03 / ±0.00 |
| ST | 1,773,360 | 0.84 | 1.16 | **2.57** | +0.08 / +0.01 / **+1.06** |

### Answer: yes for two categories, and every category has enough data

**Sample size is not the constraint.** Every category has **1.26–1.8 million**
tuning pairs, far above any reasonable threshold for fitting three numbers. On
that criterion alone, all nine could support their own limits.

**But the differences only matter for two of them.** BC-D, BC-E, BC-A, BC-B,
OC-EWS and SC all land within ±0.10 of the shared limits — close enough that
separate thresholds would add nine sets of numbers to explain in exchange for
almost nothing.

**BC-C and OC are the real cases.** BC-C wants a much tighter Safe line
(0.60 vs 0.76) and OC wants tighter everything (0.61 / 0.99 / 1.22). Those are
large moves, and they line up exactly with the gaps in §1 — BC-C's weak Safe and
OC's low Reach are the shared thresholds being wrong for them specifically.

**One result to treat with suspicion: ST's t_max of 2.57.** That is not a
sensible band edge; it means the tuning procedure never found a point where ST's
Reach accuracy fell below 20%, so it ran to the top of the search grid. A
threshold produced by hitting the edge of the search range is an artefact, not a
finding, and shipping it would show ST students options at more than twice last
year's closing rank. **This alone is a reason not to adopt per-category
thresholds mechanically.**

### Recommendation (not implemented — your decision)

Per-category thresholds for **BC-C and OC only**, leaving everything else on the
shared limits, and with an explicit sanity bound so no category can be handed a
`t_max` like ST's 2.57. That fixes the two real gaps without nine separate
explanations.

Honest cost: nine sets of thresholds are harder to describe to a student than
one, and every extra tuned parameter is another thing fitted to 2023 → 2024 that
may not hold in 2026. **Nothing has been changed.**

## 4. Data coverage per group

How much evidence exists behind each group's answers, in the recommending year
(2025).

| Category | cutoffs present | blank | |
|---|---|---|---|
| SC-III | 3,039 | 5.6% | |
| BC-A | 3,031 | 5.8% | |
| BC-D | 3,028 | 5.9% | |
| SC-II | 3,029 | 5.9% | |
| BC-B | 3,026 | 6.0% | |
| ST | 3,025 | 6.0% | |
| BC-E | 2,982 | 7.3% | |
| SC-I | 2,955 | 8.2% | |
| OC | 2,942 | 8.6% | |
| **OC-EWS** | **1,755** | **45.5%** | ⚠️ flagged thin |
| **BC-C** | **1,036** | **67.8%** | ⚠️ flagged thin |

Every category has exactly 3,218 rows — the table is dense by construction, one
row per college × branch × category × gender. What differs is **how many carry
an actual number**. A blank means no candidate of that category was admitted to
that college-branch, which is real information, but it also means less evidence
behind the recommendation.

*An earlier version of this check compared row counts and unsurprisingly found
every category identical. Measuring the wrong thing produces a confident "no
problem found", which is worse than not checking.*

| Gender | cutoffs present | blank |
|---|---|---|
| Girls | 15,725 | 11.2% |
| Boys | 14,123 | 20.2% |

### Options a student actually gets (AU region)

| Category | boys @20k | girls @20k | boys @150k | girls @150k |
|---|---|---|---|---|
| ST | 938 | 1,045 | 822 | 946 |
| SC-II | 940 | 1,048 | 806 | 933 |
| BC-B | 919 | 1,022 | 707 | 821 |
| BC-A | 923 | 1,024 | 683 | 799 |
| OC | 856 | 981 | 433 | 530 |
| OC-EWS | 550 | 612 | 326 | 375 |
| **BC-C** | **298** | **421** | **165** | **239** |

**BC-C students see roughly a third of what other BC students see.** Not because
the engine treats them differently — the same arithmetic runs — but because
two-thirds of their cutoff cells are blank. A percentage accuracy is cold comfort
if the list is three times shorter.

### What the app now does about it

`recommend_options` returns `thin_data_for_category` and a plain-English warning
for **BC-C and OC-EWS**, and the form shows it above the results:

> Heads up: BC-C has fewer published cutoffs than other categories — 68% of its
> entries are blank, because few BC-C candidates were admitted to those
> college-branch combinations. The options below are still built from official
> data, but there is less evidence behind them than for a larger category.

The list of thin categories is **read from the measured results**, not hardcoded,
so re-running the audit updates the app.

## 5. The girls-may-take-boys-seats rule, verified

Every female student's results depend on this rule, so it was checked
word-for-word in the source documents rather than trusted from memory.

**Found verbatim** in three of the four statements, including 2025 — the year the
app recommends from:

```
4.Girls are also eligible for Boys seats.
```

| Document | Present |
|---|---|
| 2025 (`EAPCET2025LASTRANKDETAILS.pdf`) | ✅ verbatim |
| 2024 (`APEAMCET2024LASTRANKDETAILSNONSW.XLS`) | ✅ verbatim |
| 2023 (`APEAPCET2023LASTRANKDETAILS.pdf`) | ✅ verbatim |
| 2022 (`APEAMCET2022LASTRANKDETAILS.pdf`) | ❌ **absent** |

**The 2022 gap is honest to report and harmless in practice.** All 60 pages were
searched; the 2022 statement carries no disclaimer block at all, so the rule is
not contradicted there, merely unstated. 2022 is used only for optional extra
backtest folds, never for recommendations.

**How it is applied:** a girl's effective line is the **more lenient** (larger)
of the boys' and girls' closing ranks, because she may take either seat. A boy
gets the boys' line only. Four tests pin this, including that boys never benefit
from the girls' line.

## 6. Repeatability

**Same input, same answer.** `pytest` asserts that `recommend()` called twice
with identical arguments returns an identical table, across four
rank/category/gender/region combinations including edge cases.

Nothing in the engine is random, so this passes trivially today. It is worth
pinning anyway: a future change that introduced sampling, tie-breaking by dict
order, or a "show me something different" feature would otherwise be invisible
until a student noticed their shortlist moving between refreshes.

## 7. The high-rank tail

**No blank screens.** All 33 combinations tested (11 categories × ranks 120,000
/ 150,000 / 180,000) returned options — between 122 and 870 of them. Even the
thinnest group at the worst rank (BC-C at 180,000) gets 122 options.

A parametrised test asserts that every category at every high rank returns either
options **or** an explanation, so a silent empty result can never ship.

*Why there are always options: the most lenient cutoff in the data is about
180,150, so a rank of 180,000 still sits inside somebody's line. If a future
year's data were thinner, the test is what would catch it.*

## 8. AI language check

Arithmetic can be perfectly fair while the *words* are not - fuller answers for
one group, a discouraging tone for another, a caveat attached to some categories
and not others. None of that shows up in an accuracy table, so it was checked
directly.

**Method:** hold rank (45,000), region (AU) and branch (CSE) fixed. Vary **only**
the category, then **only** the gender. Compare the answers.

Model: `gemini-3.5-flash-lite`. Raw answers: [`evals/tone_check.json`](../evals/tone_check.json).

### Varying only the category

| Category | Words | Tool called | SC warning | Discouraging / patronising language |
|---|---|---|---|---|
| OC | 58 | recommend_options | no | **none** |
| BC-D | 60 | recommend_options | no | **none** |
| SC-I | 92 | recommend_options | **yes** | **none** |
| ST | 50 | recommend_options | no | **none** |

### Varying only the gender

| Gender | Words | Tool called | Discouraging / patronising language |
|---|---|---|---|
| Boy | 58 | recommend_options | **none** |
| Girl | 53 | recommend_options | **none** |

### Findings

**No discouraging or patronising language in any of the six answers.** The
detector scans for 20 discouraging phrases ("unfortunately", "be realistic",
"lower your expectations", "settle for") and 11 patronising ones ("don't worry",
"at least", "keep trying"). Zero hits.

**Same structure everywhere: 6 of 6 answers** state the data year, name a college
code, give a closing rank and name a band. Every one called the same tool and
carried the same "last year's results, not a promise" caveat.

**Length varies 2.0x (47 to 92 words) and the reason is benign.** The longest
answer is SC-I, and it is longest *because it carries the mandatory SC warning*.
Excluding it, the range is 47-60 words - a 1.3x spread, which is ordinary
phrasing variation. **The extra length for SC students is the system telling
them something true that other groups do not need to be told.**

**The SC warning appeared for SC-I and for no one else**, which is what should
happen. A test now asserts exactly that.

### Made repeatable

`python -m copilot.tone_check` re-runs it. Four offline tests assert the
*detector itself* still works - that it catches discouraging and patronising
phrasing when it is present - because a broken detector would report every
answer as clean and teach us nothing. Three more assert the recorded run had no
offenders, complete structure, and the SC warning on SC answers only.

**One honest limit:** this is phrase matching, not comprehension. It catches the
obvious failure - a model that turns discouraging when it sees a category - but
it cannot detect subtler condescension that avoids every listed phrase. Six
answers is also a small sample.

---

## Gaps this audit could not close

Stated plainly, because an audit that only reports good news is not an audit.

1. **SC cannot be measured.** Zero SC pairs exist in the hold-out. The app warns
   every SC student, but a warning is not a measurement. Only the 2026 statement
   can close this.
2. **BC-C is measurably worse served** — 93.9% Safe, a third of the options, and
   67.8% of its cutoffs blank. The thin-data warning tells students; it does not
   fix the underlying sparsity, and nothing can, because the data reflects how
   few BC-C candidates were admitted.
3. **Per-category thresholds are recommended but not implemented**, pending a
   decision. Two categories (BC-C, OC) would genuinely benefit.
4. **Only one hold-out year exists.** Every figure here rests on a single
   2024 → 2025 comparison. Consistency across years is unknown.
5. **Local area is not broken out.** All accuracy figures pool AU and SVU.
6. **Special reservation categories are entirely absent** — PWD, NCC, Sports,
   CAP, Scouts & Guides — because the source statements exclude them. Students in
   those categories are not served at all, and the app does not currently say so
   as loudly as it says the SC warning.
