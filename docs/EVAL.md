# Agent evaluation

Produced by `python -m copilot.evaluate` against the real Gemini API. Every number is counted from an actual run; none is estimated.

- Model: **gemini-3.5-flash**
- Questions: **3** (3 answered, 0 errored)

## Results

| Measure | Result |
|---|---|
| Correct tool chosen | **100.0%** |
| Out-of-scope handled honestly | **None%** (0 questions) |
| SC answers carrying the warning | **None%** (0 questions) |
| Answer in the right format | **100.0%** |
| Turns where the checker removed something | **0** of 3 (0.0%) |
| Total items removed | **0** |
| Average time per answer | **22.97s** |
| Median / slowest | 13.11s / 46.42s |
| Average steps per answer | 2 |

## Cost

Free tier, so the money cost of this run was **nothing**. The real constraint is the daily request limit, not price.

## Every question

| id | kind | tool expected | tool called | ok | format | honest | SC warn | removed | secs |
|---|---|---|---|---|---|---|---|---|---|
| n01 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 46.42 |
| n02 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 13.11 |
| n03 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 9.38 |

## Notes

No question errored.
