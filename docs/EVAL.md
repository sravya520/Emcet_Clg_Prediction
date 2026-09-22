# Agent evaluation

Produced by `python -m copilot.evaluate` against the real Gemini API. Every number is counted from an actual run; none is estimated.

- Model: **gemini-3.5-flash-lite**
- Questions: **30** (30 answered, 0 errored)
- Coverage: **100.0%**

## Results

| Measure | Result |
|---|---|
| Correct tool chosen | **93.3%** |
| Out-of-scope handled honestly | **100.0%** (7 questions) |
| SC answers carrying the warning | **100.0%** (3 questions) |
| Answer in the right format | **100.0%** |
| Turns where the checker removed something | **2** of 30 (6.7%) |
| Total items removed | **2** |
| Average time per answer | **18.55s** |
| Median / slowest | 13.41s / 57.98s |
| Average steps per answer | 1.93 |

## Cost

Free tier, so the money cost of this run was **nothing**. The real constraint is the daily request limit, not price.

## Every question

| id | kind | tool expected | tool called | ok | format | honest | SC warn | removed | secs |
|---|---|---|---|---|---|---|---|---|---|
| n01 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 42.62 |
| n02 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 20.55 |
| n03 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 18.08 |
| n04 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 19.77 |
| n05 | normal | recommend_options | recommend_options | yes | yes | - | - | 0 | 18.7 |
| f01 | filter_branch | recommend_options | recommend_options | yes | yes | - | - | 0 | 11.78 |
| f02 | filter_branch | recommend_options | recommend_options | yes | yes | - | - | 0 | 10.69 |
| f03 | filter_branch | recommend_options | recommend_options | yes | yes | - | - | 0 | 5.39 |
| f04 | filter_district | recommend_options | recommend_options | yes | yes | - | - | 0 | 6.59 |
| f05 | filter_district | recommend_options | recommend_options | yes | yes | - | - | 0 | 6.31 |
| c01 | compare | compare_options | get_option_details, get_option_details, compare_options | yes | yes | - | - | 0 | 5.64 |
| c02 | compare | compare_options | get_option_details, get_option_details | **NO** | yes | - | - | 0 | 3.83 |
| c03 | compare | compare_options | compare_options | yes | yes | - | - | 0 | 2.81 |
| d01 | details | get_option_details | get_option_details | yes | yes | - | - | 0 | 4.08 |
| d02 | details | get_option_details | get_option_details | yes | yes | - | - | 0 | 2.53 |
| e01 | explain | explain_bands | explain_bands | yes | yes | - | - | 0 | 5.19 |
| e02 | explain | explain_bands | explain_bands | yes | yes | - | - | 0 | 2.89 |
| o01 | out_of_scope | (none) | get_option_details | yes | yes | yes | - | 0 | 3.86 |
| o02 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 2.03 |
| o03 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 11.16 |
| o04 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 22.05 |
| o05 | out_of_scope | (none) | (none) | yes | yes | yes | - | 0 | 21.58 |
| s01 | sc | recommend_options | recommend_options | yes | yes | - | yes | 0 | 57.98 |
| s02 | sc | recommend_options | recommend_options | yes | yes | - | yes | 0 | 51.3 |
| s03 | sc | recommend_options | explain_bands | **NO** | yes | - | yes | 0 | 39.25 |
| t01 | tricky | (none) | recommend_options | yes | yes | - | - | 0 | 34.88 |
| t02 | tricky | (none) | (none) | yes | yes | yes | - | 0 | 15.05 |
| t03 | tricky | recommend_options | recommend_options | yes | yes | - | - | 1 | 31.62 |
| t04 | tricky | recommend_options, get_option_details, compare_options | get_option_details | yes | yes | - | - | 1 | 32.67 |
| t05 | tricky | (none) | recommend_options | yes | yes | yes | - | 0 | 45.48 |

## Notes

No question errored.
