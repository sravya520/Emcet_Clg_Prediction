# Agent evaluation

Produced by `python -m copilot.evaluate` against the real Gemini API. Every number is counted from an actual run; none is estimated.

- Model: **gemini-3.5-flash-lite**
- Questions: **30** (30 answered, 0 errored)
- Coverage: **100.0%**
- Code version: `f38ebf6`

> **Run note:** **FINAL RUN (2026-09-23), gemini-3.5-flash-lite, 30/30 answered.** Run in two sittings: a first attempt reached 21/30 before dropped network connections (6 ConnectError, 3 RemoteProtocolError - not rate limits), and `--resume` completed the remaining 9 keeping the 21 already answered. Both sittings used the same model and the same agent code; the only commit between them added an index page to the API root, which the agent never touches. One known false positive remains in this run: question `o03` had the checker strip the college code `ADIT`, which the student had typed in their own question. It is the same family as the student's-own-rank bug fixed in 08d7b13 - that fix trusted numbers from the question but not codes. Left in rather than re-run so the before/after is visible.

## Quality - the headline numbers

These describe how well the agent behaves, and do not depend on how fast the network was on the day.

| Measure | Result |
|---|---|
| Correct tool chosen | **93.3%** |
| Out-of-scope handled honestly | **100.0%** (7 questions) |
| SC answers carrying the warning | **100.0%** (3 questions) |
| Answer in the right format | **100.0%** |
| Turns where the checker removed something | **1** of 30 (3.3%) |
| Total items removed | **1** |
| Average steps per answer | 1.83 |

## Latency - reported separately, because the conditions matter

**Where this ran:** local development machine, agent called in-process (NOT the deployed Render service).

| | seconds |
|---|---|
| Median | **56.36** |
| Average | **76.91** |
| Fastest | 17.86 |
| Slowest | 215.39 (`d02`) |

The average sits well above the median because a few answers spent most of their time inside Tenacity's backoff, waiting out dropped connections and rate limits, rather than waiting for the model. **That waiting is included on purpose** - it is what a user would actually have experienced - but it means these figures describe the network on the day as much as the agent. No data points were removed to make the average look better.

## Cost

Free tier, so the money cost of this run was **nothing**. The real constraint is the daily request limit, not price.

## Every question

| id | kind | tool expected | tool called | ok | format | honest | SC warn | removed | secs |
|---|---|---|---|---|---|---|---|---|---|
| n01 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 35.86 |
| n02 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 51.3 |
| n03 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 39.91 |
| n04 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 39.41 |
| n05 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 57.14 |
| f01 | filter_branch | recommend_options | recommend_options | yes | yes | - | - | 0 | 101.81 |
| f02 | filter_branch | recommend_options | recommend_options | yes | yes | - | - | 0 | 23.39 |
| f03 | filter_branch | recommend_options | recommend_options | yes | yes | - | - | 0 | 31.09 |
| f04 | filter_district | recommend_options | recommend_options | yes | yes | - | - | 0 | 96.86 |
| f05 | filter_district | recommend_options | recommend_options | yes | yes | - | - | 0 | 108.2 |
| c01 | compare | compare_options | compare_options | yes | yes | - | - | 0 | 153.34 |
| c02 | compare | compare_options | get_option_details, get_option_details | **NO** | yes | - | - | 0 | 96.75 |
| c03 | compare | compare_options | compare_options | yes | yes | - | - | 0 | 43.8 |
| d01 | details | get_option_details | get_option_details | yes | yes | - | - | 0 | 64.02 |
| d02 | details | get_option_details | get_option_details | yes | yes | - | - | 0 | 215.39 |
| e01 | explain | explain_bands | explain_bands | yes | yes | - | - | 0 | 39.0 |
| e02 | explain | explain_bands | explain_bands | yes | yes | - | - | 0 | 47.2 |
| o01 | out_of_scope | (none) | get_option_details | yes | yes | yes | - | 0 | 127.92 |
| o02 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 48.31 |
| o03 | out_of_scope | (none) | (none) | yes | yes | yes | - | 1 | 49.47 |
| o04 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 48.3 |
| o05 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 79.11 |
| s01 | sc | recommend_options | recommend_options | yes | yes | - | yes | 0 | 195.06 |
| s02 | sc | recommend_options | recommend_options | yes | yes | - | yes | 0 | 55.58 |
| s03 | sc | recommend_options | explain_bands | **NO** | yes | - | yes | 0 | 35.92 |
| t01 | tricky | (none) | recommend_options | yes | yes | - | - | 0 | 82.73 |
| t02 | tricky | (none) | (none) | yes | yes | yes | - | 0 | 17.86 |
| t03 | tricky | recommend_options | recommend_options | yes | yes | - | - | 0 | 94.38 |
| t04 | tricky | recommend_options, get_option_details, compare_options | get_option_details | yes | yes | - | - | 0 | 159.0 |
| t05 | tricky | (none) | recommend_options | yes | yes | yes | - | 0 | 69.11 |

## Notes

No question errored.
