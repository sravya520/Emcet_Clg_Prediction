# Data validation report

- Exam state: **AP EAPCET**, MPC stream
- Total rows: **117,838**
- Years: **2022, 2023, 2024, 2025**
- Counselling phase: **end_of_web_counselling**

**5 passed, 0 failed, 6 informational.**

---

## [PASS] Required columns present

All required columns are present.

## [PASS] Rows and category columns per year

Category x gender column counts match the source layouts.

|   year |   long_rows |   source_rows |   colleges |   branches |   ranks_present |   category_columns |   pct_rank_missing |
|-------:|------------:|--------------:|-----------:|-----------:|----------------:|-------------------:|-------------------:|
|   2022 |       27036 |          1502 |        297 |         61 |           24861 |                 18 |               8.04 |
|   2023 |       27234 |          1513 |        300 |         64 |           25269 |                 18 |               7.22 |
|   2024 |       28170 |          1565 |        273 |         69 |           25505 |                 18 |               9.46 |
|   2025 |       35398 |          1609 |        274 |         73 |           29848 |                 22 |              15.68 |

## [PASS] No negative or zero closing ranks

All 105,483 present ranks are positive (min 1,572, max 180,163).

## [INFO] Percentage missing per column, per year

A blank closing rank means no candidate of that category was admitted to that college-branch. It is kept as missing, never imputed.

|   year |   counselling_phase |   college_code |   college_name |   district |   inst_region |   local_area |   local_area_derived |   college_type |   branch_code |   branch_name |   name_status |   category |   gender |   closing_rank |   fee_inr |   place |   coed |   affiliation |   estd |   source_url |   source_file |   source_page |   source_sno |
|-------:|--------------------:|---------------:|---------------:|-----------:|--------------:|-------------:|---------------------:|---------------:|--------------:|--------------:|--------------:|-----------:|---------:|---------------:|----------:|--------:|-------:|--------------:|-------:|-------------:|--------------:|--------------:|-------------:|
|   2022 |                   0 |              0 |              0 |          0 |             0 |            0 |                    0 |              0 |             0 |          3.66 |             0 |          0 |        0 |           8.04 |      0    |       0 |      0 |             0 |   2.93 |            0 |             0 |             0 |            0 |
|   2023 |                   0 |              0 |              0 |          0 |             0 |            0 |                    0 |              0 |             0 |          4.76 |             0 |          0 |        0 |           7.22 |      0    |       0 |      0 |             0 |   0    |            0 |             0 |             0 |            0 |
|   2024 |                   0 |              0 |              0 |          0 |             0 |            0 |                    0 |              0 |             0 |          5.18 |             0 |          0 |        0 |           9.46 |      5.69 |       0 |      0 |             0 |   0    |            0 |             0 |           100 |            0 |
|   2025 |                   0 |              0 |              0 |          0 |             0 |            0 |                    0 |              0 |             0 |          4.72 |             0 |          0 |        0 |          15.68 |    100    |     100 |    100 |           100 | 100    |            0 |             0 |             0 |            0 |

## [PASS] No duplicate rows per key

Key ['year', 'college_code', 'branch_code', 'local_area', 'category', 'gender'] is unique across all 117,838 rows.

## [INFO] Colleges appearing in one year but not the next

Colleges open, close and change code between years. This is expected; it bounds how many college-branch pairs the backtest can match.

| pair         |   in_both |   only_earlier |   only_later | examples_only_earlier                | examples_only_later                      |
|:-------------|----------:|---------------:|-------------:|:-------------------------------------|:-----------------------------------------|
| 2022 -> 2023 |       268 |             29 |           32 | ANCP, ANNP, APCS, ASIT, AVNP, CHBR   | ASNT, AUCPSF, BCOP, BIPB, BITS, GKPS     |
| 2023 -> 2024 |       250 |             50 |           23 | ACES, ADCP, ADTP, AITS, ANUPSF, ARMN | ADTPPU, AITSPU, APCS, ASKWOC, BCET, BLMP |
| 2024 -> 2025 |       244 |             29 |           30 | ACET, ACPS, APCS, BALA, BCOP, BLMP   | ACEV, ADCP, AIPS, ANCP, ANNP, ANRG       |

## [INFO] Categories present per year

2025 replaces SC with SC-I / SC-II / SC-III. An SC row from 2024 has no one-to-one successor in 2025, so SC cannot be validated on the 2024 -> 2025 fold.

|   year |   BC-A |   BC-B |   BC-C |   BC-D |   BC-E |   OC |   OC-EWS |   SC |   SC-I |   SC-II |   SC-III |   ST |
|-------:|-------:|-------:|-------:|-------:|-------:|-----:|---------:|-----:|-------:|--------:|---------:|-----:|
|   2022 |   3004 |   3004 |   3004 |   3004 |   3004 | 3004 |     3004 | 3004 |      0 |       0 |        0 | 3004 |
|   2023 |   3026 |   3026 |   3026 |   3026 |   3026 | 3026 |     3026 | 3026 |      0 |       0 |        0 | 3026 |
|   2024 |   3130 |   3130 |   3130 |   3130 |   3130 | 3130 |     3130 | 3130 |      0 |       0 |        0 | 3130 |
|   2025 |   3218 |   3218 |   3218 |   3218 |   3218 | 3218 |     3218 |    0 |   3218 |    3218 |     3218 | 3218 |

## [INFO] Fee coverage per year

The 2025 statement carries no fee column at all. 2024 is short of 100% because state-wide colleges carry the fee on only one of their two local-area rows.

|   year |   present |   total |   pct_present |
|-------:|----------:|--------:|--------------:|
|   2022 |     27036 |   27036 |         100   |
|   2023 |     27234 |   27234 |         100   |
|   2024 |     26568 |   28170 |          94.3 |
|   2025 |         0 |   35398 |           0   |

## [INFO] Local area derived from the college's own region

2022 fills the local-area column only for state-wide colleges. For the rest we fall back to the college's own region and flag the row.

|   year |   derived |   total |   pct |
|-------:|----------:|--------:|------:|
|   2022 |     25326 |   27036 |  93.7 |
|   2023 |         0 |   27234 |   0   |
|   2024 |         0 |   28170 |   0   |
|   2025 |         0 |   35398 |   0   |

## [INFO] Branch names

77 distinct branch codes. 41 have no name and show as the raw code. All 77 named codes are marked UNOFFICIAL: no official code-to-name list was found in any counselling document.

## [PASS] Local areas limited to the two AP regions

Found ['AU', 'SVU']. AP has two local areas (Andhra University and Sri Venkateswara University). OU is Telangana and must never appear.

---

Generated by `python -m copilot.data.validate`. Every number here comes from the ingested table, not from a note written by hand.
