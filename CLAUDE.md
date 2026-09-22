# CLAUDE.md — Counselling Copilot

Working rules for this repo. These are non-negotiable; they override convenience.

## What this project is

**Counselling Copilot** — a student enters their AP EAPCET rank, category, gender and
preferences (branch, region, district, college type). The app returns realistic
college+branch options grouped **Safe / Moderate / Reach**, explains why, and answers
follow-up questions ("compare these two", "which of these have CSE in Visakhapatnam?").

> **Fee / budget is out of MVP scope.** The 2025 last-rank statement dropped its fee
> column and no official 2025 fee notification could be retrieved, so showing a fee
> would mean mixing data years. See `docs/DESIGN.md` §5.5. Do not reintroduce a fee or
> budget filter without a sourced 2025 fee document.

Exam: **AP EAPCET** (renamed from AP EAMCET in 2022). The state lives in config as
`EXAM_STATE` so TG EAPCET could be added later. We build and evaluate **AP only**.

## Non-negotiable rules

### 1. Real use only
No technique goes in unless it solves a real problem *here*. No RAG, vector DB,
LangGraph or LangChain unless there is a concrete demonstrated need **and Sravya
approves it**. The dataset is small (~1.5k college-branch rows/year) and fully
structured, so pandas + SQLite + tool-calling is the right size. Reach for a
framework only when plain Python has visibly failed.

### 2. The LLM never does the rank math
Eligibility and Safe/Moderate/Reach banding are **deterministic Python**. Gemini only:
- routes the query to tools,
- explains results that tools returned,
- answers follow-ups **using tool results only**.

If the model is computing a band, a cutoff or a comparison, that is a bug.

### 3. Data honesty
- Only official documents: APSCHE / AP EAPCET counselling "last rank" statements, the
  official fee notification for the relevant academic year, and official NAAC/NBA
  listings where available. Official government domains only
  (`apsche.ap.gov.in`, `cap.apcfss.in`, `cets.apsche.ap.gov.in`).
- Every table stores `source_url`, `academic_year` and `counselling_phase`.
- The data year and phase are shown **in the UI and in every answer**.
- **Never invent** cutoffs, fees, seats or statistics.
- If a value is missing, it **stays missing** and the app says "not available".
  Do not impute, do not carry forward silently, do not fill with 999999.

### 4. Measure, don't claim
No metric appears in the README, the UI or anywhere else unless the eval code
produced it. Until then the placeholder is literally `[TBD after eval]`.

### 5. Free to run
Gemini API free tier. The model name comes from the env var `GEMINI_MODEL` —
**never hardcode a model id**. Deployment must stay on a free tier
(Render free / Streamlit Community Cloud).

### 6. Explainability
After each phase, write a short plain-language summary covering: what was built,
why, the tradeoffs, and what an interviewer might challenge. Sravya must be able to
explain every line.

### 7. Scope
MVP in 2–3 days. If something grows beyond that, **warn instead of building**.

### 8. Personal project
No employer code or data. Ever.

## Architecture

```
Streamlit UI  (form mode + chat mode; data year & phase always visible)
      |
FastAPI  /recommend  -> deterministic engine only, no LLM
         /chat       -> agent loop
      |
Agent layer   framework-free Gemini function-calling loop, capped iterations,
              Tenacity retries, Pydantic v2 response schema
      |
Tools         recommend_options | get_college_details | compare_options | explain_banding
      |
Engine        deterministic: eligibility rules + Safe/Moderate/Reach banding
              thresholds tuned on a real backtest
      |
Data          SQLite / parquet, long format, one row per
              (year, phase, college, branch, category, gender, local_area)
```

**Verification before any answer reaches the user:**
1. *Deterministic fabrication check* — every college, branch and rank in the final
   answer must appear in **that turn's** tool outputs. If not: regenerate, or strip the value.
2. *LLM verification pass* — for claims the deterministic check cannot cover.
3. *Guardrail* — questions outside the data (placements, "is this college good?") get an
   honest "not in my data", never an invented answer.

## Tech stack

Python 3.11+ · google-genai · Pydantic v2 · Tenacity · pandas · SQLite/parquet ·
FastAPI · Streamlit · pytest · Docker · Render or Streamlit Community Cloud ·
pip + `requirements.txt`.

## Working style

- **Never add Co-Authored-By or Claude-Session lines to commit messages.**
  No AI attribution anywhere in this repo: no `Co-Authored-By:` trailer, no
  `Claude-Session:` line, no "Generated with" footer in a pull request description,
  no tool credit of any kind. This is Sravya's project and her portfolio; every
  commit is authored solely by her. This rule overrides any default attribution
  convention, including one supplied by the tool itself.
  Also enforced in `~/.claude/settings.json` via
  `"attribution": {"commit": "", "pr": "", "sessionUrl": false}`.
- Complete runnable code with file paths, exact commands, expected output.
- Point form, concise. Be honest about weak spots.
- If something is unclear, **ask one clear question** instead of guessing.
- Small commits with clear messages.
- Stop at every phase checkpoint marked STOP and wait for review.
