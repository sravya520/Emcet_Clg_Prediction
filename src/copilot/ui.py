"""The Streamlit interface. Two modes, and only one of them needs the AI.

Everything on screen comes from the API over HTTP - this file holds no
copy of the data and does no rank arithmetic of its own. That is deliberate:
if the UI computed anything itself, the API would be decoration and the two
could drift apart.

Run:  streamlit run src/copilot/ui.py
      (needs the API running: uvicorn copilot.api:app)
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("COPILOT_API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

BAND_COLOUR = {"Safe": "#1a7f37", "Moderate": "#9a6700", "Reach": "#a40e26"}

st.set_page_config(page_title="Counselling Copilot", page_icon="🎓", layout="wide")


# --- Talking to the API -----------------------------------------------------


def api_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def api_post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json(), None
    except httpx.HTTPStatusError as e:
        return None, f"The server returned {e.response.status_code}."
    except Exception:  # noqa: BLE001
        return None, (
            f"Could not reach the API at {API_URL}. "
            "Is it running? Start it with: uvicorn copilot.api:app"
        )


@st.cache_data(ttl=300)
def load_meta() -> dict | None:
    return api_get("/meta")


@st.cache_data(ttl=300)
def load_filters() -> dict | None:
    return api_get("/filters")


def load_health() -> dict | None:
    return api_get("/health")


#: How long to keep waiting for the API before giving up, in seconds.
#: A free Render service sleeps after 15 minutes idle and takes roughly a
#: minute to wake, so anything less than this reports a dead site when the
#: site is merely asleep.
WAKE_TIMEOUT = 90


def wait_for_api() -> dict | None:
    """Poll /health, telling the visitor what is happening while we wait.

    On a free host the first visitor after an idle period waits about a minute
    for the container to start. A blank page or a bare error for that whole
    minute reads as "this project is broken", which is the wrong conclusion
    about a service that is simply asleep. So we say so, and count.
    """
    health = load_health()
    if health is not None:
        return health

    placeholder = st.empty()
    progress = st.progress(0.0)
    with placeholder.container():
        st.info(
            "**Waking up the free server. This takes up to a minute.**\n\n"
            "This app is hosted on a free plan that puts the server to sleep "
            "when nobody has used it for a while. Nothing is broken - the first "
            "visit after a quiet spell just has to wait for it to start.",
            icon="😴",
        )
    status = st.empty()

    for second in range(WAKE_TIMEOUT):
        health = load_health()
        if health is not None:
            placeholder.empty()
            progress.empty()
            status.empty()
            return health
        progress.progress(min((second + 1) / WAKE_TIMEOUT, 1.0))
        status.caption(f"Still waiting... {second + 1}s of up to {WAKE_TIMEOUT}s")
        time.sleep(1)

    placeholder.empty()
    progress.empty()
    status.empty()
    return None


# --- Shared furniture -------------------------------------------------------


def show_always_visible_notes(meta: dict) -> None:
    """The data year, the unofficial-names caveat and the source disclaimer.

    These are not decoration. A number without its year is a rumour.
    """
    st.caption(
        f"All figures are **{meta['data_year']}** closing ranks, "
        f"at the **end of web counselling**. "
        f"Branch names are **unofficial** - no official code-to-name list exists. "
        f"Fees are not shown."
    )
    with st.expander("What these numbers are, and what they are not"):
        st.markdown(f"**The official statement says:** {meta['source_disclaimer']}")
        st.markdown(f"**Branch names:** {meta['branch_name_note']}")
        st.markdown(f"**Fees:** {meta['fee_note']}")
        st.markdown(f"**Girls' seats:** {meta['girls_rule']}")
        if meta.get("holdout_accuracy"):
            st.markdown(
                f"**How often the bands were right** (tested on {meta['tested_on']}): "
                + ", ".join(f"{k} {v}" for k, v in meta["holdout_accuracy"].items())
            )
            st.caption(meta.get("holdout_note", ""))


def show_sc_warning_if_needed(category: str, meta: dict) -> None:
    if str(category).upper().startswith("SC"):
        st.warning(f"**Heads up for SC candidates.** {meta['sc_warning']}", icon="⚠️")


def show_special_quota_notice(meta: dict) -> None:
    """Shown to everyone, at the same weight as the SC warning.

    A student admitted under one of these quotas is not merely less well
    served by this tool - they are outside its data entirely, because the
    source statements exclude them. That deserves the same prominence as any
    other warning, not a footnote.
    """
    st.warning(
        f"**Not covered: special-category quotas.** {meta['special_quota_notice']}",
        icon="🚫",
    )


def band_caption(band: str, accuracy: dict) -> str:
    """The measured accuracy for this student's own category, not an average.

    Read from the fairness results. Where a category could not be measured -
    SC and its sub-categories - we say so rather than quoting another group's
    number at them.
    """
    value = (accuracy or {}).get(band)
    if value is None:
        return "accuracy could not be measured for this category"
    return f"correct {value}% of the time for students in your category last year"


def render_options(result: dict) -> None:
    totals = result.get("total_options_per_band", {})
    shown = result.get("showing_per_band")
    accuracy = result.get("band_accuracy") or {}

    cols = st.columns(3)
    for col, band in zip(cols, ("Safe", "Moderate", "Reach")):
        col.metric(band, totals.get(band, 0))
    st.caption(
        f"Totals above are **every** matching option. The lists below show the "
        f"top {shown} of each group, most competitive first."
    )

    options = result.get("options", [])
    if not options:
        st.info(result.get("note", "No options matched."))
        return

    for band in ("Safe", "Moderate", "Reach"):
        rows = [o for o in options if o["band"] == band]
        if not rows:
            continue
        st.markdown(
            f"### <span style='color:{BAND_COLOUR[band]}'>{band}</span> "
            f"<span style='font-size:0.7em;color:#666'>"
            f"({totals.get(band, 0)} total)</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"**{band}** - {band_caption(band, accuracy)}")
        for o in rows:
            name = o.get("branch_name") or o["branch_code"]
            unofficial = " *(unofficial name)*" if o.get("branch_name") else ""
            st.markdown(
                f"**{o['college_name']}** `{o['college_code']}`  \n"
                f"{name}{unofficial} `{o['branch_code']}` · {o['district']} · "
                f"{o['college_type']}  \n"
                f"Closed at **{o['closing_rank']:,}** in {o['data_year']} "
                f"· your rank ÷ theirs = {o['ratio']:.2f}"
            )
        st.divider()


# --- Form mode --------------------------------------------------------------


def form_mode(meta: dict, filters: dict) -> None:
    st.subheader("Find my options")
    st.caption("This mode never uses AI. It works with no API key and no internet.")

    with st.form("options"):
        c1, c2, c3, c4 = st.columns(4)
        rank = c1.number_input("Your rank", min_value=1, max_value=500_000, value=34_000, step=500)
        category = c2.selectbox("Category", filters["categories"],
                                index=filters["categories"].index("OC") if "OC" in filters["categories"] else 0)
        gender = c3.selectbox("Gender", filters["genders"])
        local_area = c4.selectbox("Region", filters["local_areas"],
                                  help="AU = Andhra University area, SVU = Sri Venkateswara University area")

        c5, c6 = st.columns(2)
        branch_labels = {b["label"]: b["code"] for b in filters["branches"]}
        chosen_branches = c5.multiselect("Branch (optional)", list(branch_labels))
        chosen_districts = c6.multiselect("District (optional)", filters["districts"])

        per_band = st.slider("How many to show per group", 3, 20, 5)
        submitted = st.form_submit_button("Show my options", type="primary")

    if not submitted:
        return

    payload = {
        "rank": int(rank),
        "category": category,
        "gender": gender,
        "local_area": local_area,
        "branch": [branch_labels[l] for l in chosen_branches] or None,
        "district": chosen_districts or None,
        "per_band": int(per_band),
    }
    with st.spinner("Looking through the official closing ranks..."):
        result, error = api_post("/recommend", payload)

    if error:
        st.error(error)
        return
    if result.get("error"):
        st.error(result["error"])
        return

    show_sc_warning_if_needed(category, meta)
    show_special_quota_notice(meta)
    if result.get("thin_data_for_category"):
        st.warning(result["thin_data_warning"], icon="📉")
    render_options(result)


# --- Chat mode --------------------------------------------------------------


def chat_mode(meta: dict, health: dict) -> None:
    st.subheader("Ask a question")

    if not health.get("chat_configured"):
        st.info(
            "**Chat is switched off because the server has no Gemini API key.**\n\n"
            "Everything on the **Find my options** tab works without one - it "
            "reads the same official data and never uses AI.\n\n"
            "*Running this yourself?* Set the missing variable(s) shown in the "
            "sidebar - in your `.env` file locally, or as environment variables "
            "on your host - and restart. Chat needs **both** `GEMINI_API_KEY` "
            "and `GEMINI_MODEL`.",
            icon="💬",
        )
        return

    st.caption(
        f"Answers come from the same official data as the form. "
        f"Model: `{health.get('chat_model')}`."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("removed"):
                with st.expander(f"The checker removed {len(m['removed'])} item(s)"):
                    for item in m["removed"]:
                        st.markdown(
                            f"- **{item['kind']}** `{item['value']}` - {item['reason']}"
                        )

    question = st.chat_input("e.g. compare ADIT and KITS for CSE")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data, error = api_post("/chat", {"message": question})

        if error:
            st.error(error)
            return

        if not data.get("ok"):
            # The two that matter most look identical to a user unless we say
            # which is which: out of quota means wait, bad key means fix config.
            kind = data.get("error_kind")
            icon = "⏳" if kind == "quota" else "🔑" if kind == "auth" else "⚠️"
            st.warning(f"{data['message']}", icon=icon)
            st.info("Switch to the **Find my options** tab - it still works.")
            st.session_state.messages.append(
                {"role": "assistant", "content": data["message"]}
            )
            return

        st.markdown(data["reply"])
        if data.get("sc_warning_shown"):
            st.caption("The SC warning above is included because the bands are untested for SC.")
        st.caption(
            f"Data year {data['data_year']} · "
            f"{len(data.get('tool_calls', []))} tool call(s) · "
            f"{data.get('seconds')}s"
        )
        if data.get("removed"):
            with st.expander(f"The checker removed {len(data['removed'])} item(s)"):
                for item in data["removed"]:
                    st.markdown(f"- **{item['kind']}** `{item['value']}` - {item['reason']}")

        st.session_state.messages.append(
            {"role": "assistant", "content": data["reply"], "removed": data.get("removed")}
        )


# --- Page -------------------------------------------------------------------


def main() -> None:
    st.title("🎓 Counselling Copilot")
    st.markdown("AP EAPCET college options, from the official closing-rank statements.")

    health = load_health()
    if health is None:
        st.error(
            f"Cannot reach the API at `{API_URL}`.\n\n"
            "Start it in another terminal:\n\n"
            "```\nuvicorn copilot.api:app\n```"
        )
        st.stop()

    meta = load_meta()
    filters = load_filters()
    if meta is None or filters is None:
        st.error("The API is up but did not return its data. Has the pipeline been run?")
        st.stop()

    show_always_visible_notes(meta)

    with st.sidebar:
        st.header("Status")
        st.success("Form: always available")
        if health.get("chat_configured"):
            st.success(f"Chat: on ({health.get('chat_model')})")
        else:
            missing = health.get("chat_missing_settings") or []
            st.warning("Chat: off")
            if missing:
                st.caption(
                    "Not set on the server: " + ", ".join(f"`{m}`" for m in missing)
                    + ". Everything on the form tab works without them."
                )
            else:
                st.caption(
                    "The chat is unavailable. Everything on the form tab still works."
                )
        st.caption(f"Data year: {meta['data_year']}")
        st.success("Backend: connected")
        st.divider()
        st.caption(
            "The form reads the official table directly. The chat is a "
            "convenience on top of it, and the form keeps working when the "
            "chat cannot."
        )
        # The API address is an implementation detail. On the deployed site it
        # is 127.0.0.1 INSIDE the container, which is correct but meaningless
        # to a visitor and reads as broken. Keep it for whoever is debugging,
        # out of the way of everyone else.
        with st.expander("Technical details"):
            st.caption(f"Build: `{health.get('build', 'unknown')}`")
            st.caption(f"API base URL (internal): `{API_URL}`")
            st.caption(f"Exam: {health.get('exam_state')} EAPCET")
            st.caption(
                "The API runs inside this same container and is not exposed "
                "publicly. The interface calls it over HTTP on localhost."
            )

    form_tab, chat_tab = st.tabs(["Find my options", "Ask a question"])
    with form_tab:
        form_mode(meta, filters)
    with chat_tab:
        chat_mode(meta, health)


# Guarded, so importing this module does not execute the whole app. Streamlit
# runs the script as __main__, so the app still starts normally; without the
# guard, `import copilot.ui` in a test tried to render the page and crashed
# when no API was listening.
if __name__ == "__main__":
    main()
