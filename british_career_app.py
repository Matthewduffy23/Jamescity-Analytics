"""
british_career_app.py

Multi-season British & Irish leagues career analytics app.
Loads BRITISHplayers_allseasons.csv (produced by download_british_seasons.py).

PAGES:
  1. Player Search    — filter/sort by position, league, age, career score
  2. Player Profile   — career history chart, season table, trajectory, projected score
  3. Shortlist        — saved players with export

HOW TO RUN:
    streamlit run british_career_app.py
"""

import io, re, math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="British Career Analytics", layout="wide", page_icon="📈")

CSV_NAME = "BRITISHplayers_allseasons.csv"

# Season ordering — most recent first
SEASON_ORDER = [
    "2025-26","2024-25","2023-24","2022-23","2021-22","2020-21","2019-20","2018-19",
    "2026","2025","2024","2023","2022","2021","2020",
]

# ── League strengths (from Scouting-Hub) ──────────────────────
LEAGUE_STRENGTHS = {
    "England 1.": 100.00, "England 2.": 75.10, "England 3.": 61.96,
    "England 4.": 50.78,  "England 5.": 33.33, "England 6.": 16.08,
    "Scotland 1.": 61.76, "Scotland 2.": 38.63, "Scotland 3.": 20.00,
    "Ireland 1.": 50.59,  "Ireland 2.": 10.00,
    "Northern Ireland 1.": 30.98, "Wales 1.": 26.67,
}
MAX_LS = 100.0

# ── Recency weights (most recent = index 0) ───────────────────
RECENCY_WEIGHTS = [1.00, 0.85, 0.72, 0.61, 0.52, 0.44, 0.37, 0.31]

# ── Minutes confidence ────────────────────────────────────────
def minutes_confidence(mins: float) -> float:
    if mins >= 2000: return 1.00
    if mins >= 1500: return 0.85
    if mins >= 1000: return 0.65
    if mins >=  500: return 0.40
    return 0.20

# ── Trajectory classifier ────────────────────────────────────
TRAJECTORY_LABELS = {
    "Rising":   ("#22c55e", "↑"),
    "Peaking":  ("#facc15", "→"),
    "Declining":("#ef4444", "↓"),
    "Breakout": ("#818cf8", "⚡"),
    "Unknown":  ("#6b7280", "?"),
}

# ── Role buckets (identical to Scouting-Hub) ─────────────────
ROLE_BUCKETS = {
    "CM": {
        "Deep Playmaker CM": {"metrics": {"Passes per 90": 1, "Accurate passes, %": 1, "Forward passes per 90": 2,
                                          "Accurate forward passes, %": 1.5, "Progressive passes per 90": 3,
                                          "Passes to final third per 90": 2.5, "Accurate long passes, %": 1}},
        "Advanced Playmaker CM": {"metrics": {"Deep completions per 90": 1.5, "Smart passes per 90": 2,
                                              "xA per 90": 4, "Passes to penalty area per 90": 2}},
        "Defensive Midfielder DM": {"metrics": {"Defensive duels per 90": 4, "Defensive duels won, %": 4,
                                                "PAdj Interceptions": 3, "Aerial duels per 90": 0.5, "Aerial duels won, %": 1}},
        "Goal Threat CM": {"metrics": {"Non-penalty goals per 90": 3, "xG per 90": 3, "Shots per 90": 1.5, "Touches in box per 90": 2}},
        "Ball Carrying CM": {"metrics": {"Dribbles per 90": 4, "Successful dribbles, %": 2, "Progressive runs per 90": 3, "Accelerations per 90": 3}},
    },
    "CB": {
        "Ball Playing CB": {"metrics": {"Passes per 90": 2, "Accurate passes, %": 2, "Forward passes per 90": 2,
                                        "Accurate forward passes, %": 2, "Progressive passes per 90": 2,
                                        "Progressive runs per 90": 1.5, "Dribbles per 90": 1.5,
                                        "Accurate long passes, %": 1, "Passes to final third per 90": 1.5}},
        "Wide CB": {"metrics": {"Defensive duels per 90": 1.5, "Defensive duels won, %": 2, "Dribbles per 90": 2,
                                "Forward passes per 90": 1, "Progressive passes per 90": 1, "Progressive runs per 90": 2}},
        "Box Defender": {"metrics": {"Aerial duels per 90": 1, "Aerial duels won, %": 3, "PAdj Interceptions": 2,
                                     "Shots blocked per 90": 1, "Defensive duels won, %": 4}},
    },
    "FB": {
        "Build Up FB": {"metrics": {"Passes per 90": 2, "Accurate passes, %": 1.5, "Forward passes per 90": 2,
                                    "Accurate forward passes, %": 2, "Progressive passes per 90": 2.5,
                                    "Progressive runs per 90": 2, "Dribbles per 90": 2,
                                    "Passes to final third per 90": 2, "xA per 90": 1}},
        "Attacking FB": {"metrics": {"Crosses per 90": 2, "Dribbles per 90": 3.5, "Accelerations per 90": 1,
                                     "Successful dribbles, %": 1, "Touches in box per 90": 2,
                                     "Progressive runs per 90": 3, "Passes to penalty area per 90": 2, "xA per 90": 3}},
        "Defensive FB": {"metrics": {"Aerial duels per 90": 1, "Aerial duels won, %": 1.5,
                                     "Defensive duels per 90": 2, "PAdj Interceptions": 3,
                                     "Shots blocked per 90": 1, "Defensive duels won, %": 3.5}},
    },
    "ATT": {
        "Playmaker": {"metrics": {"Passes per 90": 2, "xA per 90": 3, "Key passes per 90": 1,
                                  "Deep completions per 90": 1.5, "Smart passes per 90": 1.5,
                                  "Passes to penalty area per 90": 2}},
        "Goal Threat": {"metrics": {"xG per 90": 3, "Non-penalty goals per 90": 3, "Shots per 90": 2, "Touches in box per 90": 2}},
        "Ball Carrier": {"metrics": {"Dribbles per 90": 4, "Successful dribbles, %": 2,
                                     "Progressive runs per 90": 3, "Accelerations per 90": 3}},
    },
    "CF": {
        "Target Man CF": {"metrics": {"Aerial duels per 90": 3, "Aerial duels won, %": 5}},
        "Goal Threat CF": {"metrics": {"Non-penalty goals per 90": 3, "Shots per 90": 1.5, "xG per 90": 3,
                                       "Touches in box per 90": 1, "Shots on target, %": 0.5}},
        "Link Up CF": {"metrics": {"Passes per 90": 2, "Passes to penalty area per 90": 1.5,
                                   "Deep completions per 90": 1, "Smart passes per 90": 1.5,
                                   "Accurate passes, %": 1.5, "Key passes per 90": 1,
                                   "Dribbles per 90": 2, "Successful dribbles, %": 1,
                                   "Progressive runs per 90": 2, "xA per 90": 3}},
    },
}

BADGE_EXCLUDE_ROLES = {"target man cf"}

# ── Position token helpers (identical to Scouting-Hub) ───────
def _pos_token(p: str) -> str:
    s = str(p or "").strip().upper()
    toks = [t for t in re.split(r"[,/;]\s*|\s+", s) if t]
    return toks[0] if toks else ""

def _role_key_from_pos(tok: str) -> str:
    tok = str(tok or "").upper().strip()
    if tok.startswith("CF"):            return "CF"
    if tok.startswith(("CB","LCB","RCB")): return "CB"
    if tok.startswith(("RB","LB","RWB","LWB")): return "FB"
    if tok.startswith(("DMF","CMF","LCMF","RCMF","LDMF","RDMF")): return "CM"
    if tok in {"RW","RWF","LW","LWF","AMF","RAMF","LAMF"}: return "ATT"
    return ""

POS_GROUPS = {
    "Center Backs":    ["CB","LCB","RCB"],
    "Fullbacks":       ["RB","LB","RWB","LWB"],
    "Central Mids":    ["DMF","CMF","LCMF","RCMF","LDMF","RDMF"],
    "Attackers/Wingers":["RW","RWF","LW","LWF","AMF","RAMF","LAMF"],
    "Strikers":        ["CF"],
}

ALL_OUTFIELD = {tok for toks in POS_GROUPS.values() for tok in toks}

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    candidates = [
        Path(CSV_NAME),
        Path(__file__).resolve().parent / CSV_NAME,
        Path(__file__).resolve().parent.parent / CSV_NAME,
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return _clean(df)
    return pd.DataFrame()

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names
    df = df.rename(columns={"Team within selected timeframe": "Team"})

    num_cols = [
        "Minutes played","Matches played","Goals","Assists",
        "xG","xA","Age",
        "Passes per 90","Accurate passes, %","Forward passes per 90",
        "Accurate forward passes, %","Progressive passes per 90",
        "Passes to final third per 90","Accurate long passes, %",
        "Deep completions per 90","Smart passes per 90","xA per 90",
        "Passes to penalty area per 90","Key passes per 90",
        "Defensive duels per 90","Defensive duels won, %",
        "PAdj Interceptions","Aerial duels per 90","Aerial duels won, %",
        "Shots blocked per 90","Non-penalty goals per 90","xG per 90",
        "Shots per 90","Touches in box per 90","Dribbles per 90",
        "Successful dribbles, %","Progressive runs per 90","Accelerations per 90",
        "Crosses per 90","Accurate crosses, %","Shots on target, %",
        "Goal conversion, %",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Position token
    if "Position" in df.columns:
        df["_pos_tok"] = df["Position"].apply(_pos_token)
        df["_role_key"] = df["_pos_tok"].apply(_role_key_from_pos)

    # Season rank (0 = most recent)
    df["_season_rank"] = df["Season"].apply(
        lambda s: SEASON_ORDER.index(str(s)) if str(s) in SEASON_ORDER else 99
    )

    return df


# ══════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════
def _season_role_score(row: pd.Series, ref_df: pd.DataFrame) -> float:
    """
    Compute raw role score for a single season row vs its reference pool.
    Returns the best (non-excluded) role score as a 0-100 percentile-based value.
    """
    role_key = str(row.get("_role_key",""))
    buckets = ROLE_BUCKETS.get(role_key, {})
    if not buckets:
        return np.nan

    best = np.nan
    for role_name, spec in buckets.items():
        if role_name.lower() in BADGE_EXCLUDE_ROLES:
            continue
        met_w = (spec or {}).get("metrics", {})
        vals, wts = [], []
        for met, w in met_w.items():
            if met not in ref_df.columns:
                continue
            col_vals = pd.to_numeric(ref_df[met], errors="coerce").dropna()
            player_val = pd.to_numeric(row.get(met, np.nan), errors="coerce")
            if pd.isna(player_val) or col_vals.empty:
                continue
            pct = float((col_vals < player_val).sum() + 0.5 * (col_vals == player_val).sum()) / len(col_vals) * 100
            vals.append(pct)
            wts.append(float(w))
        if vals and sum(wts) > 0:
            score = float(np.average(vals, weights=wts))
            if pd.isna(best) or score > best:
                best = score
    return best


@st.cache_data(show_spinner=False)
def compute_all_season_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every row compute:
      - _raw_score       : role score vs same-league + same-position pool
      - _ls_adj_score    : league-strength adjusted score (β=0.40)
      - _weighted_score  : ls_adj_score × recency_weight × minutes_confidence
    """
    rows_out = []

    for (league, pos_tok), grp in df.groupby(["League", "_pos_tok"], sort=False):
        role_key = _role_key_from_pos(pos_tok)
        if not role_key:
            for _, row in grp.iterrows():
                rows_out.append({**row.to_dict(), "_raw_score": np.nan,
                                 "_ls_adj_score": np.nan, "_weighted_score": np.nan})
            continue

        # ref pool = same league + same role group
        allowed = {tok for toks in POS_GROUPS.values() for tok in toks
                   if _role_key_from_pos(tok) == role_key}
        ref = df[(df["League"] == league) & (df["_pos_tok"].isin(allowed))].copy()

        ls = float(LEAGUE_STRENGTHS.get(str(league), 30.0))
        ls_norm = ls / MAX_LS * 100.0  # put on 0-100 scale same as percentile

        for _, row in grp.iterrows():
            raw = _season_role_score(row, ref)
            if pd.notna(raw):
                ls_adj = 0.60 * raw + 0.40 * ls_norm
                rec_w  = RECENCY_WEIGHTS[min(int(row.get("_season_rank", 0)), len(RECENCY_WEIGHTS)-1)]
                min_c  = minutes_confidence(float(row.get("Minutes played", 0) or 0))
                weighted = ls_adj * rec_w * min_c
            else:
                ls_adj = weighted = np.nan

            rows_out.append({**row.to_dict(),
                             "_raw_score": raw,
                             "_ls_adj_score": ls_adj,
                             "_weighted_score": weighted})

    return pd.DataFrame(rows_out)


@st.cache_data(show_spinner=False)
def compute_career_scores(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-Wyscout-ID across all seasons into a career profile.
    Returns one row per player with career_score, trajectory, projection etc.
    """
    records = []

    for wid, grp in df_scored.groupby("Wyscout ID", sort=False):
        grp = grp.sort_values("_season_rank")  # most recent first
        valid = grp[grp["_weighted_score"].notna()].copy()
        if valid.empty:
            continue

        # Career score = sum(weighted) / sum(recency × minutes_conf) — normalise
        total_weight = sum(
            RECENCY_WEIGHTS[min(int(r["_season_rank"]), len(RECENCY_WEIGHTS)-1)]
            * minutes_confidence(float(r.get("Minutes played", 0) or 0))
            for _, r in valid.iterrows()
        )
        career_score = float(valid["_weighted_score"].sum() / total_weight) if total_weight > 0 else np.nan

        # Most recent row for player info
        latest = grp.iloc[0]

        # Trajectory — use ls_adj_score over last 3 seasons with data
        traj_rows = valid.head(3)
        trajectory = _classify_trajectory(traj_rows)

        # Projection
        projection = _project_score(valid)

        # Season history list for profile page
        history = []
        for _, r in grp.iterrows():
            history.append({
                "Season": r.get("Season",""),
                "League": r.get("League",""),
                "Team":   r.get("Team",""),
                "Age":    r.get("Age", np.nan),
                "Minutes played": r.get("Minutes played", np.nan),
                "Goals":  r.get("Goals", np.nan),
                "Assists":r.get("Assists", np.nan),
                "_ls_adj_score": r.get("_ls_adj_score", np.nan),
                "_raw_score":    r.get("_raw_score", np.nan),
            })

        records.append({
            "Wyscout ID":    wid,
            "Player":        latest.get("Player",""),
            "Team":          latest.get("Team",""),
            "League":        latest.get("League",""),
            "Position":      latest.get("Position",""),
            "_pos_tok":      latest.get("_pos_tok",""),
            "_role_key":     latest.get("_role_key",""),
            "Age":           latest.get("Age", np.nan),
            "career_score":  career_score,
            "trajectory":    trajectory,
            "projection":    projection,
            "seasons_data":  len(valid),
            "_history":      history,
        })

    out = pd.DataFrame(records)
    if not out.empty:
        out["career_score"] = pd.to_numeric(out["career_score"], errors="coerce")
    return out


def _classify_trajectory(traj_rows: pd.DataFrame) -> str:
    """Classify trajectory from up to 3 most-recent seasons (sorted recent→old)."""
    scores = traj_rows["_ls_adj_score"].dropna().tolist()
    mins   = traj_rows["Minutes played"].fillna(0).tolist()

    if len(scores) < 2:
        return "Unknown"

    # Breakout: low minutes in older seasons, high minutes recent
    if len(mins) >= 2 and mins[-1] < 600 and mins[0] >= 900 and scores[0] > 60:
        return "Breakout"

    # Compute trend using oldest → newest (reverse)
    scores_asc = list(reversed(scores))
    if len(scores_asc) >= 2:
        # Simple linear slope
        x = np.arange(len(scores_asc), dtype=float)
        slope = np.polyfit(x, scores_asc, 1)[0]
        recent = scores[0]  # most recent

        if slope > 3.0:
            return "Rising"
        elif slope < -3.0:
            return "Declining"
        else:
            if recent >= 65:
                return "Peaking"
            else:
                return "Peaking"  # flat and lower = still peaking/plateaued

    return "Unknown"


def _project_score(valid: pd.DataFrame, years_ahead: int = 2) -> dict:
    """
    Project score at current_age + years_ahead using linear regression on
    league-adjusted scores vs age. Returns dict with projected_score, peak_age, confidence.
    """
    sub = valid[["Age","_ls_adj_score"]].dropna()
    if len(sub) < 2:
        return {"projected_score": np.nan, "peak_age": np.nan, "confidence": "low"}

    ages   = sub["Age"].astype(float).values
    scores = sub["_ls_adj_score"].astype(float).values

    current_age = float(valid.iloc[0].get("Age", np.nan))
    if pd.isna(current_age):
        return {"projected_score": np.nan, "peak_age": np.nan, "confidence": "low"}

    # Fit quadratic if 3+ points, linear if 2
    if len(sub) >= 3:
        coeffs = np.polyfit(ages, scores, 2)
        proj_age = current_age + years_ahead
        projected = float(np.polyval(coeffs, proj_age))
        # Peak of quadratic: -b / 2a
        a, b, _ = coeffs
        peak_age = float(-b / (2*a)) if abs(a) > 1e-6 else np.nan
    else:
        coeffs = np.polyfit(ages, scores, 1)
        proj_age = current_age + years_ahead
        projected = float(np.polyval(coeffs, proj_age))
        peak_age = np.nan

    projected = max(0.0, min(100.0, projected))
    confidence = "high" if len(sub) >= 4 else "medium" if len(sub) >= 3 else "low"

    return {
        "projected_score": round(projected, 1),
        "peak_age": round(peak_age, 1) if pd.notna(peak_age) else None,
        "confidence": confidence,
        "proj_age": round(proj_age, 1),
    }


# ══════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════
def career_chart(history: list, player_name: str, trajectory: str) -> plt.Figure:
    """Draw career trajectory line chart with league labels."""
    valid = [(h["Season"], h["_ls_adj_score"], h["League"]) for h in history if pd.notna(h["_ls_adj_score"])]
    if not valid:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0d1117")
        ax.text(0.5, 0.5, "Insufficient data", color="#9ca3af", ha="center", va="center", fontsize=12)
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        return fig

    # Sort chronologically
    valid_sorted = sorted(valid, key=lambda x: SEASON_ORDER.index(x[0]) if x[0] in SEASON_ORDER else 99, reverse=True)
    seasons = [v[0] for v in valid_sorted]
    scores  = [v[1] for v in valid_sorted]
    leagues = [v[2] for v in valid_sorted]

    traj_color = TRAJECTORY_LABELS.get(trajectory, TRAJECTORY_LABELS["Unknown"])[0]

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#111827")

    ax.plot(range(len(scores)), scores, color=traj_color, linewidth=2.5, zorder=3)
    ax.scatter(range(len(scores)), scores, color=traj_color, s=60, zorder=4)

    # Shade under line
    ax.fill_between(range(len(scores)), scores, alpha=0.15, color=traj_color)

    # Annotate league changes
    prev_lg = None
    for i, (s, lg) in enumerate(zip(scores, leagues)):
        ax.text(i, s + 2.5, f"{s:.0f}", fontsize=7.5, color="#e5e7eb", ha="center", va="bottom", fontweight="bold")
        if lg != prev_lg:
            lg_short = lg.replace("England ","ENG ").replace("Scotland ","SCO ").replace("Ireland ","IRL ").replace("Northern Ireland ","NIR ").replace("Wales ","WAL ")
            ax.text(i, 2, lg_short, fontsize=6.5, color="#9ca3af", ha="center", va="bottom", rotation=0)
            prev_lg = lg

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, fontsize=8, color="#9ca3af")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Adj. Score", fontsize=9, color="#9ca3af")
    ax.tick_params(axis="y", colors="#9ca3af", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#374151")
    ax.yaxis.label.set_color("#9ca3af")
    ax.set_title(f"{player_name} — Career Trajectory", fontsize=11, color="#e5e7eb", pad=10)
    ax.grid(axis="y", color="#1f2937", linewidth=0.8, zorder=0)
    fig.tight_layout()
    return fig


def projection_chart(valid_history: list, projection: dict, player_name: str) -> plt.Figure:
    """Draw score vs age with projected point."""
    pts = [(h["Age"], h["_ls_adj_score"]) for h in valid_history if pd.notna(h["_ls_adj_score"]) and pd.notna(h["Age"])]
    if not pts:
        fig, ax = plt.subplots(figsize=(6,3), facecolor="#0d1117")
        ax.axis("off")
        return fig

    ages   = [p[0] for p in pts]
    scores = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0d1117")
    ax.set_facecolor("#111827")

    ax.scatter(ages, scores, color="#60a5fa", s=60, zorder=4, label="Historical")
    ax.plot(ages, scores, color="#60a5fa", linewidth=1.5, alpha=0.6, zorder=3)

    proj_score = projection.get("projected_score", np.nan)
    proj_age   = projection.get("proj_age", np.nan)

    if pd.notna(proj_score) and pd.notna(proj_age):
        ax.scatter([proj_age], [proj_score], color="#f59e0b", s=100, zorder=5,
                   marker="*", label=f"Projected (age {proj_age:.0f}): {proj_score:.0f}")
        ax.plot([ages[-1] if ages else proj_age, proj_age],
                [scores[-1] if scores else proj_score, proj_score],
                color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)

    # Peak age line
    peak_age = projection.get("peak_age")
    if peak_age and 15 < peak_age < 40:
        ax.axvline(peak_age, color="#a78bfa", linewidth=1.2, linestyle=":", alpha=0.7, label=f"Est. peak age {peak_age:.0f}")

    ax.set_xlabel("Age", fontsize=9, color="#9ca3af")
    ax.set_ylabel("Adj. Score", fontsize=9, color="#9ca3af")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#374151")
    ax.set_title(f"{player_name} — Score vs Age + Projection", fontsize=10, color="#e5e7eb", pad=8)
    ax.legend(fontsize=7.5, facecolor="#1f2937", labelcolor="#e5e7eb", framealpha=0.8)
    ax.grid(axis="both", color="#1f2937", linewidth=0.8, zorder=0)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════
def score_badge_color(score: float) -> str:
    if pd.isna(score): return "#4b5563"
    if score >= 70:    return "#22c55e"
    if score >= 55:    return "#facc15"
    if score >= 40:    return "#f97316"
    return "#ef4444"

def traj_chip(trajectory: str) -> str:
    color, icon = TRAJECTORY_LABELS.get(trajectory, ("#6b7280","?"))
    return f'<span style="background:{color};color:#000;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700">{icon} {trajectory}</span>'

def render_player_card(row: pd.Series, on_shortlist: bool, shortlist_key: str):
    """Render a compact player tile."""
    score = float(row.get("career_score", np.nan))
    traj  = str(row.get("trajectory","Unknown"))
    proj  = row.get("projection", {}) or {}
    color = score_badge_color(score)

    with st.container():
        col_score, col_info, col_action = st.columns([1, 4, 1])

        with col_score:
            st.markdown(
                f'<div style="background:{color};border-radius:12px;padding:10px 6px;text-align:center;">'
                f'<div style="font-size:22px;font-weight:900;color:#000">{score:.0f}</div>'
                f'<div style="font-size:9px;color:#000;opacity:0.7">CAREER</div>'
                f'</div>', unsafe_allow_html=True
            )

        with col_info:
            proj_str = ""
            ps = proj.get("projected_score")
            pa = proj.get("proj_age")
            if ps and pa:
                proj_str = f" · <span style='color:#f59e0b'>▶ {ps:.0f} @ {pa:.0f}</span>"

            conf_str = f" · {proj.get('confidence','?')} conf." if proj.get("confidence") else ""

            st.markdown(
                f"**{row['Player']}**  "
                f"<span style='color:#9ca3af;font-size:12px'>{row.get('Position','')}</span><br>"
                f"<span style='color:#d1d5db;font-size:12px'>{row.get('Team','')} · {row.get('League','')} · Age {row.get('Age','?')}</span><br>"
                f"{traj_chip(traj)}{proj_str}{conf_str} · "
                f"<span style='color:#6b7280;font-size:11px'>{int(row.get('seasons_data',0))} seasons</span>",
                unsafe_allow_html=True
            )

        with col_action:
            btn_label = "★ Remove" if on_shortlist else "☆ Save"
            if st.button(btn_label, key=f"sl_{shortlist_key}_{row['Wyscout ID']}", use_container_width=True):
                sl = st.session_state.setdefault("shortlist", {})
                wid = str(row["Wyscout ID"])
                if on_shortlist:
                    sl.pop(wid, None)
                else:
                    sl[wid] = row.to_dict()
                st.rerun()

        st.divider()


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
st.session_state.setdefault("shortlist", {})
st.session_state.setdefault("_scores_computed", False)
st.session_state.setdefault("selected_player_id", None)

# ── Sidebar nav ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 British Career Analytics")
    page = st.radio("", ["🔍 Player Search", "👤 Player Profile", "⭐ Shortlist"], label_visibility="collapsed")
    st.divider()
    sl_count = len(st.session_state.get("shortlist", {}))
    st.caption(f"Shortlist: {sl_count} player{'s' if sl_count != 1 else ''}")

# ── Load data ─────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df_raw = load_data()

if df_raw.empty:
    st.error(f"Could not find **{CSV_NAME}**. Place it in the same folder as this script and restart.")
    st.stop()

# ── Compute scores (cached) ───────────────────────────────────
with st.spinner("Computing season scores..."):
    df_scored = compute_all_season_scores(df_raw)

with st.spinner("Building career profiles..."):
    df_career = compute_career_scores(df_scored)

if df_career.empty:
    st.error("No scoreable players found. Check your CSV has position and metric columns.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PLAYER SEARCH
# ══════════════════════════════════════════════════════════════
if page == "🔍 Player Search":
    st.title("🔍 Player Search")

    # ── Filters ──────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([2,2,2,1,1])

    with f1:
        pos_filter = st.multiselect(
            "Position group",
            list(POS_GROUPS.keys()),
            default=list(POS_GROUPS.keys()),
            key="search_pos"
        )

    with f2:
        all_leagues = sorted(df_career["League"].dropna().unique().tolist())
        league_filter = st.multiselect("League (current)", all_leagues, default=all_leagues, key="search_league")

    with f3:
        traj_filter = st.multiselect(
            "Trajectory",
            ["Rising","Peaking","Declining","Breakout","Unknown"],
            default=["Rising","Peaking","Declining","Breakout","Unknown"],
            key="search_traj"
        )

    with f4:
        min_age, max_age = st.slider("Age", 15, 40, (16, 32), key="search_age")

    with f5:
        min_seasons = st.slider("Min seasons", 1, 8, 1, key="search_seasons")

    name_search = st.text_input("Search player name", "", key="search_name")

    sort_col = st.selectbox("Sort by", ["career_score","Age","seasons_data"], index=0, key="search_sort")

    # ── Apply filters ─────────────────────────────────────────
    allowed_toks = {tok for grp in pos_filter for tok in POS_GROUPS.get(grp, [])}
    dfc = df_career.copy()

    if allowed_toks:
        dfc = dfc[dfc["_pos_tok"].isin(allowed_toks)]
    if league_filter:
        dfc = dfc[dfc["League"].isin(league_filter)]
    if traj_filter:
        dfc = dfc[dfc["trajectory"].isin(traj_filter)]

    dfc = dfc[(dfc["Age"] >= min_age) & (dfc["Age"] <= max_age)]
    dfc = dfc[dfc["seasons_data"] >= min_seasons]

    if name_search.strip():
        dfc = dfc[dfc["Player"].str.contains(name_search.strip(), case=False, na=False)]

    dfc = dfc.sort_values(sort_col, ascending=(sort_col == "Age"), na_position="last")

    st.caption(f"**{len(dfc)}** players · sorted by {sort_col}")

    # ── Results ───────────────────────────────────────────────
    shortlist = st.session_state.get("shortlist", {})
    for i, (_, row) in enumerate(dfc.head(100).iterrows()):
        on_sl = str(row["Wyscout ID"]) in shortlist
        render_player_card(row, on_sl, shortlist_key=f"search_{i}")

    if len(dfc) > 100:
        st.info(f"Showing top 100 of {len(dfc)} players. Use filters to narrow down.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PLAYER PROFILE
# ══════════════════════════════════════════════════════════════
elif page == "👤 Player Profile":
    st.title("👤 Player Profile")

    # Player picker
    player_names = sorted(df_career["Player"].dropna().unique().tolist())
    default_idx = 0

    # If coming from search shortlist
    pre_id = st.session_state.get("selected_player_id")
    if pre_id:
        pre_row = df_career[df_career["Wyscout ID"] == pre_id]
        if not pre_row.empty:
            pre_name = pre_row.iloc[0]["Player"]
            if pre_name in player_names:
                default_idx = player_names.index(pre_name)

    picked = st.selectbox("Select player", player_names, index=default_idx, key="profile_pick")
    prow = df_career[df_career["Player"] == picked].iloc[0]

    wid        = prow["Wyscout ID"]
    trajectory = str(prow["trajectory"])
    projection = prow["projection"] or {}
    history    = prow["_history"] or []
    career_sc  = float(prow.get("career_score", np.nan))
    color      = score_badge_color(career_sc)

    # ── Header ───────────────────────────────────────────────
    h1, h2, h3 = st.columns([1.2, 3.5, 1.5])

    with h1:
        st.markdown(
            f'<div style="background:{color};border-radius:16px;padding:18px 10px;text-align:center;margin-top:8px">'
            f'<div style="font-size:36px;font-weight:900;color:#000">{career_sc:.0f}</div>'
            f'<div style="font-size:11px;color:#000;opacity:0.75">CAREER SCORE</div>'
            f'</div>', unsafe_allow_html=True
        )

    with h2:
        traj_color, traj_icon = TRAJECTORY_LABELS.get(trajectory, ("#6b7280","?"))
        st.markdown(f"## {prow['Player']}")
        st.markdown(
            f"**{prow.get('Position','')}** · {prow.get('Team','')} · {prow.get('League','')} · Age **{prow.get('Age','?')}**<br>"
            f"{traj_chip(trajectory)} · **{int(prow.get('seasons_data',0))}** seasons of data",
            unsafe_allow_html=True
        )

        # Projection summary
        ps = projection.get("projected_score")
        pa = projection.get("proj_age")
        pk = projection.get("peak_age")
        conf = projection.get("confidence","?")
        if ps and pa:
            delta = ps - career_sc
            delta_str = f"(+{delta:.0f})" if delta >= 0 else f"({delta:.0f})"
            st.markdown(
                f"**Projected score:** "
                f"<span style='color:#f59e0b;font-size:18px;font-weight:700'>{ps:.0f}</span> "
                f"<span style='color:#9ca3af'>@ age {pa:.0f} {delta_str} · {conf} confidence"
                + (f" · Est. peak age {pk:.0f}" if pk else "") + "</span>",
                unsafe_allow_html=True
            )

    with h3:
        shortlist = st.session_state.get("shortlist", {})
        on_sl = str(wid) in shortlist
        if st.button("★ Remove from shortlist" if on_sl else "☆ Add to shortlist",
                     key="profile_sl_btn", use_container_width=True):
            if on_sl:
                shortlist.pop(str(wid), None)
            else:
                shortlist[str(wid)] = prow.to_dict()
            st.session_state["shortlist"] = shortlist
            st.rerun()

    st.divider()

    # ── Charts ────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        fig1 = career_chart(history, prow["Player"], trajectory)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with c2:
        fig2 = projection_chart(history, projection, prow["Player"])
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Season-by-season table ────────────────────────────────
    st.subheader("Season History")

    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        hist_df = hist_df.sort_values(
            "Season",
            key=lambda s: s.map(lambda x: SEASON_ORDER.index(x) if x in SEASON_ORDER else 99)
        )

        display_cols = ["Season","League","Team","Age","Minutes played","Goals","Assists","_ls_adj_score","_raw_score"]
        display_cols = [c for c in display_cols if c in hist_df.columns]
        hist_display = hist_df[display_cols].rename(columns={
            "_ls_adj_score": "Adj. Score",
            "_raw_score":    "Raw Score",
        })

        for num_col in ["Adj. Score","Raw Score","Age","Minutes played","Goals","Assists"]:
            if num_col in hist_display.columns:
                hist_display[num_col] = pd.to_numeric(hist_display[num_col], errors="coerce").round(1)

        st.dataframe(
            hist_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Adj. Score": st.column_config.ProgressColumn("Adj. Score", min_value=0, max_value=100, format="%.0f"),
                "Raw Score":  st.column_config.ProgressColumn("Raw Score",  min_value=0, max_value=100, format="%.0f"),
            }
        )

    # ── Raw season stats ──────────────────────────────────────
    with st.expander("Full season stats"):
        raw_rows = df_scored[df_scored["Wyscout ID"] == wid].sort_values(
            "_season_rank"
        ).drop(columns=["_pos_tok","_role_key","_season_rank","_raw_score","_ls_adj_score","_weighted_score"], errors="ignore")

        st.dataframe(raw_rows, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — SHORTLIST
# ══════════════════════════════════════════════════════════════
elif page == "⭐ Shortlist":
    st.title("⭐ Shortlist")

    shortlist = st.session_state.get("shortlist", {})

    if not shortlist:
        st.info("No players saved yet. Use Player Search or Player Profile to add players.")
    else:
        sl_ids  = list(shortlist.keys())
        sl_rows = df_career[df_career["Wyscout ID"].astype(str).isin(sl_ids)].copy()
        sl_rows = sl_rows.sort_values("career_score", ascending=False, na_position="last")

        st.caption(f"**{len(sl_rows)}** players on shortlist")

        for i, (_, row) in enumerate(sl_rows.iterrows()):
            render_player_card(row, on_shortlist=True, shortlist_key=f"sl_{i}")

        st.divider()

        # Export
        export_cols = ["Player","Team","League","Position","Age","career_score","trajectory","seasons_data"]
        export_cols = [c for c in export_cols if c in sl_rows.columns]
        export_df = sl_rows[export_cols].copy()
        export_df = export_df.rename(columns={"career_score":"Career Score","trajectory":"Trajectory","seasons_data":"Seasons"})

        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Export shortlist CSV",
            data=csv_bytes,
            file_name="shortlist.csv",
            mime="text/csv",
        )

        if st.button("🗑 Clear shortlist", type="secondary"):
            st.session_state["shortlist"] = {}
            st.rerun()
