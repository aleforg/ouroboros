"""Altair chart builders for the Ouroboros dashboard.

Altair is bundled with Streamlit (``>=1.37`` ships ``altair>=5``).
All functions return an ``altair.Chart`` (or layered chart) so callers
can pass it directly to ``st.altair_chart(..., use_container_width=True)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt


# ---------------------------------------------------------------------------
# ASR vs. iteration budget
# ---------------------------------------------------------------------------


def asr_vs_iter_chart(df: pd.DataFrame) -> "alt.LayerChart":
    """Interactive line chart of ASR vs iter_budget with 95% CI bands.

    Expects the DataFrame produced by ``metrics.asr_vs_iter()``:
    columns ``[iter_budget, category, asr, asr_ci_low, asr_ci_high, ...]``.
    """
    import altair as alt

    if df.empty:
        return alt.Chart(pd.DataFrame({"iter_budget": [], "asr": [], "category": []})).mark_line()

    # Move the synthetic "<all>" aggregate to a separate series for visibility
    df = df.copy()
    df["series"] = df["category"].where(df["category"] != "<all>", "All (aggregate)")

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.12)
        .encode(
            x=alt.X("iter_budget:Q", title="Iteration budget"),
            y=alt.Y("asr_ci_low:Q", title=""),
            y2=alt.Y2("asr_ci_high:Q"),
            color=alt.Color("series:N", legend=alt.Legend(title="Category")),
        )
    )

    line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("iter_budget:Q", title="Iteration budget"),
            y=alt.Y("asr:Q", title="ASR", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("series:N", legend=alt.Legend(title="Category")),
            strokeDash=alt.condition(
                alt.datum.category == "<all>",
                alt.value([6, 3]),
                alt.value([1, 0]),
            ),
            tooltip=[
                "iter_budget:Q",
                "series:N",
                alt.Tooltip("asr:Q", format=".3f"),
                alt.Tooltip("asr_ci_low:Q", format=".3f", title="CI low"),
                alt.Tooltip("asr_ci_high:Q", format=".3f", title="CI high"),
                "n_seeds:Q",
            ],
        )
    )

    return (band + line).properties(height=320, title="ASR vs. Iteration Budget").interactive()


# ---------------------------------------------------------------------------
# Cross-run ASR comparison
# ---------------------------------------------------------------------------


def cross_run_asr_chart(df: pd.DataFrame) -> "alt.Chart":
    """Bar chart for aggregate cross-run ASR with error bars (±std).

    Expects the ``per_category`` list-of-dicts from ``metrics.aggregate_runs()``,
    converted to a DataFrame with columns ``[category, n_runs, mean_asr, std_asr]``.
    """
    import altair as alt

    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_bar()

    bar = (
        alt.Chart(df)
        .mark_bar(color="#4a90e2")
        .encode(
            x=alt.X("category:N", title="Category", sort="-y"),
            y=alt.Y("mean_asr:Q", title="Mean ASR", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                "category:N",
                alt.Tooltip("mean_asr:Q", format=".3f"),
                alt.Tooltip("std_asr:Q", format=".3f", title="Std ASR"),
                "n_runs:Q",
            ],
        )
    )

    error = (
        alt.Chart(df)
        .mark_errorbar()
        .encode(
            x="category:N",
            y=alt.Y("mean_asr:Q", title=""),
            yError=alt.YError("std_asr:Q"),
        )
    )

    return (bar + error).properties(height=300, title="Cross-Run Mean ASR per Category")


# ---------------------------------------------------------------------------
# RAM gauge helper (returns plain metrics, not an Altair chart)
# ---------------------------------------------------------------------------


def ram_summary(ram_record: dict) -> dict[str, float]:
    """Extract rss_gb and available_gb from a ``ram.jsonl`` record for display."""
    if not ram_record:
        return {}
    # ram.jsonl records have nested 'phases' or top-level 'rss_gb'/'available_gb'
    # depending on when they were written.  Try both layouts.
    rss = ram_record.get("rss_gb") or ram_record.get("process_rss_gb", 0.0)
    avail = ram_record.get("available_gb") or ram_record.get("system_available_gb", 0.0)
    total = ram_record.get("total_gb") or ram_record.get("system_total_gb", 0.0)
    return {"rss_gb": float(rss), "available_gb": float(avail), "total_gb": float(total)}
