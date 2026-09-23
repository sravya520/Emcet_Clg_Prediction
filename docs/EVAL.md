# Agent evaluation

Produced by `python -m copilot.evaluate` against the real Gemini API. Every number is counted from an actual run; none is estimated.

- Model: **gemini-3.5-flash-lite**
- Questions: **30** (21 answered, 9 errored)
- Coverage: **70.0%**
- Code version: `8edd556`

> ## INCOMPLETE RUN - THESE NUMBERS ARE NOT REPORTABLE
>
> Only 21 of 30 questions were answered (70.0% coverage). The percentages below are computed over the answered subset only and must not be quoted as the agent's accuracy.
>
> Failures by kind: `{'RemoteProtocolError': 3, 'ConnectError': 6}`

## Results

| Measure | Result |
|---|---|
| Correct tool chosen | **95.2%** |
| Out-of-scope handled honestly | **100.0%** (1 questions) |
| SC answers carrying the warning | **None%** (0 questions) |
| Answer in the right format | **100.0%** |
| Turns where the checker removed something | **0** of 21 (0.0%) |
| Total items removed | **0** |
| Average time per answer | **76.12s** |
| Median / slowest | 57.14s / 215.39s |
| Average steps per answer | 1.95 |

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
| o01 | out_of_scope | (none) | (none) | **NO** | **NO** | - | - | 0 | 53.59 |
| o02 | out_of_scope | (none) | (none) | **NO** | **NO** | - | - | 0 | 12.06 |
| o03 | out_of_scope | (none) | (none) | **NO** | **NO** | - | - | 0 | 4.05 |
| o04 | out_of_scope | (none) | (none) | **NO** | **NO** | - | - | 0 | 14.5 |
| o05 | out_of_scope | (none) | (none) | **NO** | **NO** | - | - | 0 | 13.81 |
| s01 | sc | recommend_options | (none) | **NO** | **NO** | - | - | 0 | 12.91 |
| s02 | sc | recommend_options | (none) | **NO** | **NO** | - | - | 0 | 10.78 |
| s03 | sc | recommend_options | (none) | **NO** | **NO** | - | - | 0 | 11.7 |
| t01 | tricky | (none) | recommend_options | yes | yes | - | - | 0 | 82.73 |
| t02 | tricky | (none) | (none) | yes | yes | yes | - | 0 | 17.86 |
| t03 | tricky | recommend_options | recommend_options | yes | yes | - | - | 0 | 94.38 |
| t04 | tricky | recommend_options, get_option_details, compare_options | get_option_details | yes | yes | - | - | 0 | 159.0 |
| t05 | tricky | (none) | (none) | **NO** | **NO** | - | - | 0 | 41.76 |

## Notes

Questions that errored:
- `o01`: RemoteProtocolError: Server disconnected without sending a response.
- `o02`: ConnectError: [Errno 11001] getaddrinfo failed
- `o03`: ConnectError: [Errno 11001] getaddrinfo failed
- `o04`: RemoteProtocolError: Server disconnected without sending a response.
- `o05`: ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)
- `s01`: ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)
- `s02`: ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)
- `s03`: ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)
- `t05`: RemoteProtocolError: Server disconnected without sending a response.
