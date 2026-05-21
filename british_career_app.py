"""
british_career_app.py  —  Jamescity Analytics · British Career Intelligence
Loads BRITISHplayers_allseasons.csv  (produced by download_british_seasons.py)

PAGES
  1. Player Search   — pro-layout cards, search/filter, rating & potential badges
  2. Player Profile  — career history chart, season metrics dropdown, projection
  3. Shortlist       — saved players + CSV export

RUN:  streamlit run british_career_app.py
"""

import io, re, math, base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import requests

# ── Optional helper modules (copied from Scouting-Hub) ───────
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FOTMOB_TEAM_URLS
except Exception:
    _FOTMOB_TEAM_URLS = {}

try:
    from league_logo_urls import get_league_logo_url as _get_league_logo_url
except Exception:
    _get_league_logo_url = lambda lg: ""

try:
    from photo_utils import get_player_photo_url as _get_player_photo_url
except Exception:
    _get_player_photo_url = lambda player, team: ""

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Jamescity Analytics · British Career Intelligence",
    layout="wide",
    page_icon="📈",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS  —  Montserrat dark theme matching other apps
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif!important;}
.stApp{background:#0a0f1c!important;}
section[data-testid="stSidebar"]{background:#060a14!important;border-right:1px solid #0d1220!important;}
section[data-testid="stSidebar"] *{color:#fff!important;}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important;}
.stSelectbox>div>div{background:#0d1424!important;border:1px solid #1e2d4a!important;}
div[data-baseweb="select"] *{background:#0d1424!important;color:#fff!important;}
div[data-baseweb="popover"] *{background:#0d1424!important;color:#fff!important;}
.stTextInput>div>div>input{background:#0d1424!important;border:1px solid #1e2d4a!important;color:#fff!important;}
.stButton>button{background:#ffffff!important;color:#000000!important;font-weight:700!important;
  border:none!important;font-family:'Montserrat',sans-serif!important;border-radius:2px!important;}
.stDownloadButton>button{background:#ffffff!important;color:#000000!important;font-weight:700!important;
  border:none!important;font-family:'Montserrat',sans-serif!important;border-radius:2px!important;}
label{color:#6b7280!important;font-size:9px!important;letter-spacing:.12em!important;text-transform:uppercase!important;}
h1,h2,h3{color:#fff!important;font-family:'Montserrat',sans-serif!important;}
footer{display:none!important;}
div[data-testid="stDataFrame"] *{font-family:'Montserrat',sans-serif!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
CSV_NAME = "BRITISHplayers_allseasons.csv"

SEASON_ORDER = [
    "2025-26","2024-25","2023-24","2022-23","2021-22","2020-21","2019-20","2018-19",
    "2026","2025","2024","2023","2022","2021","2020",
]

LEAGUE_STRENGTHS = {
    "England 1.": 100.00, "England 2.": 75.10,  "England 3.": 61.96,
    "England 4.": 50.78,  "England 5.": 33.33,  "England 6.": 16.08,
    "Scotland 1.": 61.76, "Scotland 2.": 38.63, "Scotland 3.": 20.00,
    "Ireland 1.":  50.59, "Ireland 2.":  10.00,
    "Northern Ireland 1.": 30.98, "Wales 1.": 26.67,
}
MAX_LS = 100.0
BETA   = 0.40

RECENCY_WEIGHTS = [1.00, 0.85, 0.72, 0.61, 0.52, 0.44, 0.37, 0.31]

PROJECTED_LEVEL_MAP = [
    (85, "Premier League"),
    (72, "Championship"),
    (60, "League One"),
    (48, "League Two"),
    (36, "National League"),
    (0,  "Below NL"),
]

TRAJECTORY_LABELS = {
    "Rising":    ("#22c55e",  "↑"),
    "Peaking":   ("#facc15",  "→"),
    "Declining": ("#ef4444",  "↓"),
    "Breakout":  ("#818cf8",  "⚡"),
    "Unknown":   ("#6b7280",  "?"),
}

POS_COLORS = {
    "CF":"#6EA8FF","LWF":"#6EA8FF","RWF":"#6EA8FF","LW":"#6EA8FF","RW":"#6EA8FF",
    "LAMF":"#6EA8FF","RAMF":"#6EA8FF","AMF":"#7FE28A",
    "LCMF":"#5FD37A","RCMF":"#5FD37A","CMF":"#5FD37A",
    "DMF":"#31B56B","LDMF":"#31B56B","RDMF":"#31B56B",
    "LWB":"#FFD34D","RWB":"#FFD34D","LB":"#FF9A3C","RB":"#FF9A3C",
    "CB":"#D1763A","LCB":"#D1763A","RCB":"#D1763A",
    "GK":"#B8A1FF",
}

LEAGUE_SHORT = {
    "England 1.":"PL","England 2.":"CHAMP","England 3.":"L1","England 4.":"L2",
    "England 5.":"NL","England 6.":"NL6",
    "Scotland 1.":"SPL","Scotland 2.":"SCH","Scotland 3.":"SL1",
    "Ireland 1.":"LOI","Ireland 2.":"FD","Northern Ireland 1.":"NIL","Wales 1.":"WPL",
}

ROLE_BUCKETS = {
    "CM": {
        "Deep Playmaker":    {"metrics":{"Passes per 90":1,"Accurate passes, %":1,"Forward passes per 90":2,"Accurate forward passes, %":1.5,"Progressive passes per 90":3,"Passes to final third per 90":2.5,"Accurate long passes, %":1}},
        "Advanced Playmaker":{"metrics":{"Deep completions per 90":1.5,"Smart passes per 90":2,"xA per 90":4,"Passes to penalty area per 90":2}},
        "Defensive CM":      {"metrics":{"Defensive duels per 90":4,"Defensive duels won, %":4,"PAdj Interceptions":3,"Aerial duels per 90":0.5,"Aerial duels won, %":1}},
        "Goal Threat CM":    {"metrics":{"Non-penalty goals per 90":3,"xG per 90":3,"Shots per 90":1.5,"Touches in box per 90":2}},
        "Ball Carrying CM":  {"metrics":{"Dribbles per 90":4,"Successful dribbles, %":2,"Progressive runs per 90":3,"Accelerations per 90":3}},
    },
    "CB": {
        "Ball Playing CB": {"metrics":{"Passes per 90":2,"Accurate passes, %":2,"Forward passes per 90":2,"Accurate forward passes, %":2,"Progressive passes per 90":2,"Progressive runs per 90":1.5,"Dribbles per 90":1.5,"Accurate long passes, %":1,"Passes to final third per 90":1.5}},
        "Wide CB":         {"metrics":{"Defensive duels per 90":1.5,"Defensive duels won, %":2,"Dribbles per 90":2,"Forward passes per 90":1,"Progressive passes per 90":1,"Progressive runs per 90":2}},
        "Box Defender":    {"metrics":{"Aerial duels per 90":1,"Aerial duels won, %":3,"PAdj Interceptions":2,"Shots blocked per 90":1,"Defensive duels won, %":4}},
    },
    "FB": {
        "Build Up FB":   {"metrics":{"Passes per 90":2,"Accurate passes, %":1.5,"Forward passes per 90":2,"Accurate forward passes, %":2,"Progressive passes per 90":2.5,"Progressive runs per 90":2,"Dribbles per 90":2,"Passes to final third per 90":2,"xA per 90":1}},
        "Attacking FB":  {"metrics":{"Crosses per 90":2,"Dribbles per 90":3.5,"Accelerations per 90":1,"Successful dribbles, %":1,"Touches in box per 90":2,"Progressive runs per 90":3,"Passes to penalty area per 90":2,"xA per 90":3}},
        "Defensive FB":  {"metrics":{"Aerial duels per 90":1,"Aerial duels won, %":1.5,"Defensive duels per 90":2,"PAdj Interceptions":3,"Shots blocked per 90":1,"Defensive duels won, %":3.5}},
    },
    "ATT": {
        "Playmaker":    {"metrics":{"Passes per 90":2,"xA per 90":3,"Key passes per 90":1,"Deep completions per 90":1.5,"Smart passes per 90":1.5,"Passes to penalty area per 90":2}},
        "Goal Threat":  {"metrics":{"xG per 90":3,"Non-penalty goals per 90":3,"Shots per 90":2,"Touches in box per 90":2}},
        "Ball Carrier": {"metrics":{"Dribbles per 90":4,"Successful dribbles, %":2,"Progressive runs per 90":3,"Accelerations per 90":3}},
    },
    "CF": {
        "Goal Threat CF": {"metrics":{"Non-penalty goals per 90":3,"Shots per 90":1.5,"xG per 90":3,"Touches in box per 90":1,"Shots on target, %":0.5}},
        "Link Up CF":     {"metrics":{"Passes per 90":2,"Passes to penalty area per 90":1.5,"Deep completions per 90":1,"Smart passes per 90":1.5,"Accurate passes, %":1.5,"Key passes per 90":1,"Dribbles per 90":2,"Successful dribbles, %":1,"Progressive runs per 90":2,"xA per 90":3}},
        "Target Man CF":  {"metrics":{"Aerial duels per 90":3,"Aerial duels won, %":5}},
    },
}
BADGE_EXCLUDE = {"target man cf"}

POS_GROUPS = {
    "Center Backs":          ["CB","LCB","RCB"],
    "Fullbacks":             ["RB","LB","RWB","LWB"],
    "Central Mids":          ["DMF","CMF","LCMF","RCMF","LDMF","RDMF"],
    "Attackers / Wingers":   ["RW","RWF","LW","LWF","AMF","RAMF","LAMF"],
    "Strikers":              ["CF"],
}
ALL_OUTFIELD = {tok for toks in POS_GROUPS.values() for tok in toks}

# ── Country flag helper ───────────────────────────────────────
_TWEMOJI_SPECIAL = {
    "eng":"1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct":"1f3f4-e0067-e0062-e0073-e0063-e0074-e007f",
    "wls":"1f3f4-e0067-e0062-e0077-e006c-e0073-e007f",
}
_CC_MAP = {
    "england":"eng","scotland":"sct","wales":"wls","northern ireland":"gb",
    "ireland":"ie","republic of ireland":"ie",
    "brazil":"br","argentina":"ar","france":"fr","germany":"de","spain":"es",
    "italy":"it","portugal":"pt","netherlands":"nl","belgium":"be",
}
def _flag_img(country: str) -> str:
    if not country: return ""
    n = str(country).strip().lower()
    cc = _CC_MAP.get(n, "")
    if not cc:
        return ""
    if cc in _TWEMOJI_SPECIAL:
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{_TWEMOJI_SPECIAL[cc]}.svg"
    elif len(cc) == 2:
        base = 0x1F1E6
        code = f"{base+(ord(cc[0].upper())-65):x}-{base+(ord(cc[1].upper())-65):x}"
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
    else:
        return ""
    return f'<img src="{src}" style="height:13px;vertical-align:middle;margin-right:3px;">'


# ══════════════════════════════════════════════════════════════
# BADGE / LOGO HELPERS  (identical to app__9_.py)
# ══════════════════════════════════════════════════════════════
def _fotmob_crest_url(team: str) -> str:
    raw = (_FOTMOB_TEAM_URLS.get(team) or "").strip()
    if not raw:
        return ""
    if raw.lower().endswith((".png",".jpg",".jpeg",".webp",".svg")):
        return raw
    m = re.search(r"/teams/(\d+)/", raw)
    if m:
        return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png"
    return ""

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_b64(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        mime = r.headers.get("Content-Type","image/png").split(";")[0].strip()
        return f"data:{mime};base64,{base64.b64encode(r.content).decode()}"
    except Exception:
        return ""

def _team_badge_html(team: str, size: int = 28) -> str:
    """Inline badge img or empty string."""
    url = _fotmob_crest_url(team)
    if not url:
        return ""
    b64 = _fetch_b64(url)
    if not b64:
        return ""
    return (f'<img src="{b64}" style="width:{size}px;height:{size}px;'
            f'object-fit:contain;vertical-align:middle;margin-right:6px;border-radius:3px;">')

def _league_logo_html(league: str, size: int = 18) -> str:
    """Inline league logo img or empty string."""
    url = _get_league_logo_url(league)
    if not url:
        return ""
    b64 = _fetch_b64(url)
    if not b64:
        return ""
    return (f'<img src="{b64}" style="width:{size}px;height:{size}px;'
            f'object-fit:contain;vertical-align:middle;margin-right:4px;border-radius:2px;">')

# ══════════════════════════════════════════════════════════════
# POSITION HELPERS  (identical to Quick Search / ScoutBoard Pro)
# ══════════════════════════════════════════════════════════════
def _pos_token(p: str) -> str:
    s = str(p or "").strip().upper()
    toks = [t for t in re.split(r"[,/;]\s*|\s+", s) if t]
    return toks[0] if toks else ""

def _role_key(tok: str) -> str:
    tok = str(tok or "").upper().strip()
    if tok.startswith("CF"):                        return "CF"
    if tok.startswith(("CB","LCB","RCB")):          return "CB"
    if tok.startswith(("RB","LB","RWB","LWB")):     return "FB"
    if tok.startswith(("DMF","CMF","LCMF","RCMF","LDMF","RDMF")): return "CM"
    if tok in {"RW","RWF","LW","LWF","AMF","RAMF","LAMF"}:        return "ATT"
    return ""

# ══════════════════════════════════════════════════════════════
# RATING COLOUR  (matching other apps)
# ══════════════════════════════════════════════════════════════
def rating_color(v: float) -> str:
    try: v = float(v)
    except: return "#1a2035"
    if v >= 85: return "#2E6114"
    if v >= 75: return "#5C9E2E"
    if v >= 66: return "#7FBC41"
    if v >= 54: return "#A7D763"
    if v >= 44: return "#F6D645"
    if v >= 25: return "#D77A2E"
    return "#C63733"

def fmt2(v) -> str:
    try: return f"{max(0, min(99, int(float(v)))):02d}"
    except: return "--"

def minutes_confidence(mins: float) -> float:
    if mins >= 2000: return 1.00
    if mins >= 1500: return 0.85
    if mins >= 1000: return 0.65
    if mins >=  500: return 0.40
    return 0.20

def projected_level(score: float) -> str:
    for threshold, label in PROJECTED_LEVEL_MAP:
        if score >= threshold:
            return label
    return "Below NL"

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    for p in [Path(CSV_NAME), Path(__file__).resolve().parent / CSV_NAME]:
        if p.exists():
            df = pd.read_csv(p)
            return _clean(df)
    return pd.DataFrame()

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Team within selected timeframe": "Team"})
    num_cols = [
        "Minutes played","Matches played","Goals","Assists","xG","xA","Age",
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
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Position" in df.columns:
        df["_pos_tok"]  = df["Position"].apply(_pos_token)
        df["_role_key"] = df["_pos_tok"].apply(_role_key)
    df["_season_rank"] = df["Season"].apply(
        lambda s: SEASON_ORDER.index(str(s)) if str(s) in SEASON_ORDER else 99
    )
    return df

# ══════════════════════════════════════════════════════════════
# VECTORISED SCORING ENGINE
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised: computes percentile-based role score per row vs same-league
    same-position-group reference pool. Adds _raw_score, _ls_adj_score, _weighted_score.
    """
    df = df.copy()
    df["_raw_score"]      = np.nan
    df["_ls_adj_score"]   = np.nan
    df["_weighted_score"] = np.nan

    for (league, role_k), grp_idx in df.groupby(["League", "_role_key"]).groups.items():
        if not role_k:
            continue
        buckets = ROLE_BUCKETS.get(role_k, {})
        if not buckets:
            continue

        # reference pool: same league, same role group
        pool_mask = (df["League"] == league) & (df["_role_key"] == role_k)
        ref = df[pool_mask]

        ls       = float(LEAGUE_STRENGTHS.get(str(league), 30.0))
        ls_norm  = ls / MAX_LS * 100.0

        for role_name, spec in buckets.items():
            if role_name.lower() in BADGE_EXCLUDE:
                continue
            met_w = spec.get("metrics", {})
            avail = {m: w for m, w in met_w.items() if m in df.columns}
            if not avail:
                continue

            # Vectorised percentile: for each metric compute pct vs ref pool
            pct_cols = []
            weights  = []
            for met, w in avail.items():
                ref_vals = pd.to_numeric(ref[met], errors="coerce").dropna().values
                if len(ref_vals) == 0:
                    continue
                player_vals = pd.to_numeric(df.loc[grp_idx, met], errors="coerce")
                pct = player_vals.apply(
                    lambda v, rv=ref_vals: float((rv < v).sum() + 0.5*(rv == v).sum()) / len(rv) * 100
                    if pd.notna(v) else np.nan
                )
                pct_cols.append(pct)
                weights.append(w)

            if not pct_cols:
                continue

            pct_matrix = pd.concat(pct_cols, axis=1)
            w_arr      = np.array(weights, dtype=float)
            role_score = (pct_matrix.mul(w_arr).sum(axis=1) /
                          pct_matrix.notna().mul(w_arr).sum(axis=1))

            # Keep best role score per player
            existing = df.loc[grp_idx, "_raw_score"]
            better   = role_score > existing.fillna(-1)
            df.loc[grp_idx[better], "_raw_score"] = role_score[better]

        # Compute ls_adj and weighted for this group
        raw = df.loc[grp_idx, "_raw_score"]
        valid = raw.notna()
        if valid.any():
            ls_adj = (1 - BETA) * raw + BETA * ls_norm
            df.loc[grp_idx[valid], "_ls_adj_score"] = ls_adj[valid]

            ranks = df.loc[grp_idx, "_season_rank"]
            mins  = df.loc[grp_idx, "Minutes played"].fillna(0)
            rec_w = ranks.apply(lambda r: RECENCY_WEIGHTS[min(int(r), len(RECENCY_WEIGHTS)-1)])
            min_c = mins.apply(minutes_confidence)
            weighted = ls_adj * rec_w * min_c
            df.loc[grp_idx[valid], "_weighted_score"] = weighted[valid]

    return df

@st.cache_data(show_spinner=False)
def build_career_profiles(df_scored: pd.DataFrame) -> pd.DataFrame:
    records = []
    for wid, grp in df_scored.groupby("Wyscout ID", sort=False):
        grp   = grp.sort_values("_season_rank")
        valid = grp[grp["_weighted_score"].notna()]
        if valid.empty:
            continue

        # Career score
        total_w = sum(
            RECENCY_WEIGHTS[min(int(r["_season_rank"]), len(RECENCY_WEIGHTS)-1)]
            * minutes_confidence(float(r.get("Minutes played", 0) or 0))
            for _, r in valid.iterrows()
        )
        career_score = float(valid["_weighted_score"].sum() / total_w) if total_w > 0 else np.nan

        # Potential = best single-season ls_adj score (peak performance)
        potential = float(valid["_ls_adj_score"].max())

        latest = grp.iloc[0]

        trajectory = _classify_trajectory(valid.head(3))
        projection = _project_score(valid)

        history = []
        for _, r in grp.iterrows():
            history.append({
                "Season":         r.get("Season", ""),
                "League":         r.get("League", ""),
                "Team":           r.get("Team", ""),
                "Age":            r.get("Age", np.nan),
                "Minutes played": r.get("Minutes played", np.nan),
                "Goals":          r.get("Goals", np.nan),
                "Assists":        r.get("Assists", np.nan),
                "_ls_adj_score":  r.get("_ls_adj_score", np.nan),
                "_raw_score":     r.get("_raw_score", np.nan),
                # store all metric columns too for dropdown
                **{c: r.get(c, np.nan) for c in df_scored.columns
                   if c not in {"Season","League","Team","Age","Minutes played",
                                "Goals","Assists","_ls_adj_score","_raw_score",
                                "Wyscout ID","Player","Position","_pos_tok",
                                "_role_key","_season_rank","_raw_score",
                                "_ls_adj_score","_weighted_score",
                                "Market value","Contract expires","On loan",
                                "Birth country","Passport countries","Foot","Height"}
                   and not str(c).startswith("_")}
            })

        records.append({
            "Wyscout ID":   wid,
            "Player":       latest.get("Player", ""),
            "Team":         latest.get("Team", ""),
            "League":       latest.get("League", ""),
            "Position":     latest.get("Position", ""),
            "_pos_tok":     latest.get("_pos_tok", ""),
            "_role_key":    latest.get("_role_key", ""),
            "Age":          latest.get("Age", np.nan),
            "Birth country":latest.get("Birth country", ""),
            "Foot":         latest.get("Foot", ""),
            "Height":       latest.get("Height", np.nan),
            "career_score": career_score,
            "potential":    potential,
            "trajectory":   trajectory,
            "projection":   projection,
            "seasons_data": len(valid),
            "_history":     history,
        })

    out = pd.DataFrame(records)
    if not out.empty:
        out["career_score"] = pd.to_numeric(out["career_score"], errors="coerce")
        out["potential"]    = pd.to_numeric(out["potential"], errors="coerce")
    return out


def _classify_trajectory(traj_rows: pd.DataFrame) -> str:
    scores = traj_rows["_ls_adj_score"].dropna().tolist()
    mins   = traj_rows["Minutes played"].fillna(0).tolist()
    if len(scores) < 2:
        return "Unknown"
    if len(mins) >= 2 and mins[-1] < 600 and mins[0] >= 900 and scores[0] > 60:
        return "Breakout"
    scores_asc = list(reversed(scores))
    x     = np.arange(len(scores_asc), dtype=float)
    slope = np.polyfit(x, scores_asc, 1)[0]
    if slope > 3.0:  return "Rising"
    if slope < -3.0: return "Declining"
    return "Peaking"


def _project_score(valid: pd.DataFrame) -> dict:
    sub = valid[["Age","_ls_adj_score"]].dropna()
    if len(sub) < 2:
        return {"projected_score": np.nan, "peak_age": None, "confidence": "low", "proj_age": np.nan}
    ages   = sub["Age"].astype(float).values
    scores = sub["_ls_adj_score"].astype(float).values
    current_age = float(valid.iloc[0].get("Age", np.nan))
    if pd.isna(current_age):
        return {"projected_score": np.nan, "peak_age": None, "confidence": "low", "proj_age": np.nan}
    proj_age = current_age + 2
    if len(sub) >= 3:
        coeffs = np.polyfit(ages, scores, 2)
        projected = float(np.polyval(coeffs, proj_age))
        a, b, _ = coeffs
        peak_age = float(-b / (2*a)) if abs(a) > 1e-6 else None
        confidence = "high" if len(sub) >= 4 else "medium"
    else:
        coeffs = np.polyfit(ages, scores, 1)
        projected = float(np.polyval(coeffs, proj_age))
        peak_age  = None
        confidence = "low"
    projected = max(0.0, min(100.0, projected))
    return {
        "projected_score": round(projected, 1),
        "peak_age":  round(peak_age, 1) if peak_age and 15 < peak_age < 42 else None,
        "confidence": confidence,
        "proj_age":   round(proj_age, 1),
    }


# ══════════════════════════════════════════════════════════════
# PRO LAYOUT CARD HTML
# ══════════════════════════════════════════════════════════════
def _score_chip_html(label: str, val, size: int = 26) -> str:
    bg = rating_color(val) if pd.notna(val) else "#1a2035"
    fg = "#000" if pd.notna(val) and float(val) >= 44 else "#fff"
    v_str = fmt2(val) if pd.notna(val) else "--"
    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px;">'
        f'<div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:.10em;text-transform:uppercase;">{label}</div>'
        f'<div style="background:{bg};color:{fg};font-size:{size}px;font-weight:900;'
        f'padding:5px 14px;border-radius:6px;min-width:52px;text-align:center;'
        f'font-family:Montserrat,sans-serif;">{v_str}</div>'
        f'</div>'
    )

def _traj_chip_html(trajectory: str) -> str:
    color, icon = TRAJECTORY_LABELS.get(trajectory, ("#6b7280","?"))
    return (f'<span style="background:{color};color:#000;padding:2px 9px;border-radius:99px;'
            f'font-size:10px;font-weight:800;font-family:Montserrat,sans-serif;">'
            f'{icon} {trajectory}</span>')

def _pos_chip_html(pos_tok: str) -> str:
    bg = POS_COLORS.get(pos_tok.upper(), "#2d3550")
    return (f'<span style="background:{bg};color:#000;padding:1px 7px;border-radius:4px;'
            f'font-size:10px;font-weight:800;font-family:Montserrat,sans-serif;">{pos_tok}</span>')

def _league_short(lg: str) -> str:
    return LEAGUE_SHORT.get(str(lg), str(lg).replace("England ","ENG ").replace("Scotland ","SCO "))

def player_card_html(row: pd.Series) -> str:
    cs    = float(row.get("career_score", np.nan))
    pot   = float(row.get("potential", np.nan))
    traj  = str(row.get("trajectory", "Unknown"))
    proj  = row.get("projection", {}) or {}
    pos   = str(row.get("_pos_tok",""))
    lg    = str(row.get("League",""))
    flag  = _flag_img(str(row.get("Birth country","")))
    badge = _team_badge_html(str(row.get("Team","")), size=22)
    ll    = _league_logo_html(str(row.get("League","")), size=16)
    proj_lvl = projected_level(cs) if pd.notna(cs) else "—"
    proj_score = proj.get("projected_score")
    proj_age   = proj.get("proj_age")
    foot  = str(row.get("Foot","") or "")
    foot_html = f'<span style="color:#6b7280;font-size:10px;">{foot[:1].upper()}{"F" if foot else ""}</span>' if foot and foot.lower() not in {"nan","none",""} else ""

    proj_str = ""
    if proj_score and proj_age:
        proj_str = (f'<span style="color:#f59e0b;font-size:10px;font-weight:700;">'
                    f'▶ {proj_score:.0f} @ {proj_age:.0f}</span>')

    return f"""
<div style="background:#0f1628;border:1px solid #1a2540;border-radius:12px;
            padding:14px 18px;margin-bottom:10px;display:flex;align-items:center;gap:16px;
            font-family:Montserrat,sans-serif;">

  <!-- SCORE CHIPS -->
  <div style="display:flex;flex-direction:column;gap:6px;align-items:center;flex-shrink:0;min-width:110px;">
    {_score_chip_html("Rating", cs, 28)}
    {_score_chip_html("Potential", pot, 22)}
  </div>

  <!-- PLAYER INFO -->
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
      <span style="font-size:16px;font-weight:900;color:#fff;white-space:nowrap;">{row['Player']}</span>
      {_pos_chip_html(pos)}
      {_traj_chip_html(traj)}
      {foot_html}
    </div>
    <div style="font-size:11px;color:#9ca3af;margin-bottom:5px;">
      {badge}{flag}{row.get('Team','')}
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      {ll}<span style="color:#cbd5e1;font-weight:600;">{_league_short(lg)}</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      Age <span style="color:#fff;font-weight:700;">{int(row['Age']) if pd.notna(row.get('Age')) else '?'}</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      <span style="color:#6b7280;">{int(row.get('seasons_data',0))} seasons</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <span style="font-size:10px;color:#9ca3af;">Projected level:
        <span style="color:#e5e7eb;font-weight:700;">{proj_lvl}</span>
      </span>
      {proj_str}
    </div>
  </div>

</div>"""


# ══════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════
COLOR_SCALE = ["#be2a3e","#e25f48","#f88f4d","#f4d166","#90b960","#4b9b5f","#22763f"]
CMAP = LinearSegmentedColormap.from_list("jca", COLOR_SCALE)

def career_chart(history: list, player_name: str, trajectory: str) -> plt.Figure:
    valid = [(h["Season"], h["_ls_adj_score"], h["League"])
             for h in history if pd.notna(h.get("_ls_adj_score"))]
    if not valid:
        fig, ax = plt.subplots(figsize=(7,3), facecolor="#0a0f1c")
        ax.text(0.5,0.5,"Insufficient data",color="#9ca3af",ha="center",va="center",fontsize=11)
        ax.set_facecolor("#0a0f1c"); ax.axis("off")
        return fig

    valid_s = sorted(valid, key=lambda x: SEASON_ORDER.index(x[0]) if x[0] in SEASON_ORDER else 99, reverse=True)
    seasons = [v[0] for v in valid_s]
    scores  = [v[1] for v in valid_s]
    leagues = [v[2] for v in valid_s]

    tc, _ = TRAJECTORY_LABELS.get(trajectory, ("#60a5fa","→"))

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0a0f1c")
    ax.set_facecolor("#111827")

    xs = range(len(scores))
    ax.fill_between(xs, scores, alpha=0.12, color=tc)
    ax.plot(xs, scores, color=tc, linewidth=2.5, zorder=3)
    ax.scatter(xs, scores, color=tc, s=70, zorder=4)

    prev_lg = None
    for i, (s, lg) in enumerate(zip(scores, leagues)):
        ax.text(i, s + 2.8, f"{s:.0f}", fontsize=8, color="#e5e7eb",
                ha="center", va="bottom", fontweight="bold",
                fontfamily="Montserrat")
        lg_s = _league_short(lg)
        if lg != prev_lg:
            ax.text(i, 1.5, lg_s, fontsize=6.5, color="#9ca3af",
                    ha="center", va="bottom", fontfamily="Montserrat")
            prev_lg = lg

    # Reference lines
    for lvl, lbl in [(85,"PL"), (72,"CHAMP"), (60,"L1"), (48,"L2"), (36,"NL")]:
        ax.axhline(lvl, color="#1f2937", linewidth=0.8, linestyle="--", zorder=0)
        ax.text(len(scores)-0.5, lvl+0.8, lbl, fontsize=6, color="#374151",
                ha="right", va="bottom", fontfamily="Montserrat")

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, fontsize=8, color="#9ca3af", fontfamily="Montserrat")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Adj. Score", fontsize=9, color="#9ca3af")
    ax.tick_params(axis="y", colors="#9ca3af", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#374151")
    ax.grid(axis="y", color="#1f2937", linewidth=0.6, zorder=0)
    ax.set_title(f"{player_name}  —  Career Trajectory",
                 fontsize=11, color="#e5e7eb", pad=10, fontfamily="Montserrat", fontweight=700)
    fig.tight_layout()
    return fig


def projection_chart(history: list, projection: dict, player_name: str) -> plt.Figure:
    pts = [(h["Age"], h["_ls_adj_score"]) for h in history
           if pd.notna(h.get("_ls_adj_score")) and pd.notna(h.get("Age"))]
    if not pts:
        fig, ax = plt.subplots(figsize=(6,3), facecolor="#0a0f1c"); ax.axis("off"); return fig

    ages   = sorted(set(p[0] for p in pts))
    # average if multiple seasons at same age
    age_score = {}
    for a, s in pts:
        age_score.setdefault(a, []).append(s)
    ages_u  = sorted(age_score.keys())
    scores_u = [np.mean(age_score[a]) for a in ages_u]

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0a0f1c")
    ax.set_facecolor("#111827")

    ax.scatter(ages_u, scores_u, color="#60a5fa", s=65, zorder=4, label="Historical")
    ax.plot(ages_u, scores_u, color="#60a5fa", linewidth=1.8, alpha=0.6, zorder=3)

    ps  = projection.get("projected_score")
    pa  = projection.get("proj_age")
    pk  = projection.get("peak_age")

    if ps and pa and pd.notna(ps) and pd.notna(pa):
        ax.scatter([pa],[ps], color="#f59e0b", s=120, zorder=5, marker="*",
                   label=f"Projected (age {pa:.0f}): {ps:.0f}")
        ax.plot([ages_u[-1], pa],[scores_u[-1], ps],
                color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3)

    if pk and 15 < pk < 42:
        ax.axvline(pk, color="#a78bfa", linewidth=1.2, linestyle=":",
                   alpha=0.7, label=f"Est. peak age {pk:.0f}")

    # Level bands
    for lvl, lbl in [(85,"PL"), (72,"Champ"), (60,"L1"), (48,"L2")]:
        ax.axhline(lvl, color="#1f2937", linewidth=0.7, linestyle="--", zorder=0)
        ax.text(max(ages_u+([] if not pa else [pa]))+0.2, lvl+0.5, lbl,
                fontsize=6.5, color="#374151", va="bottom", fontfamily="Montserrat")

    ax.set_xlabel("Age", fontsize=9, color="#9ca3af")
    ax.set_ylabel("Adj. Score", fontsize=9, color="#9ca3af")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#374151")
    ax.set_title(f"{player_name}  —  Score vs Age + Projection",
                 fontsize=10, color="#e5e7eb", pad=8, fontfamily="Montserrat", fontweight=700)
    ax.legend(fontsize=7.5, facecolor="#1f2937", labelcolor="#e5e7eb", framealpha=0.8)
    ax.grid(axis="both", color="#1f2937", linewidth=0.6, zorder=0)
    fig.tight_layout()
    return fig


def regression_chart(df_career: pd.DataFrame, player_row: pd.Series) -> plt.Figure:
    """
    Predictive / regression chart: plots all players (role group) score vs age,
    fits a population curve, overlays this player's trajectory + projection.
    Shows where the player sits vs historical development paths.
    """
    role_k = str(player_row.get("_role_key",""))
    if not role_k:
        fig, ax = plt.subplots(figsize=(8,4), facecolor="#0a0f1c")
        ax.text(0.5,0.5,"No position data",color="#9ca3af",ha="center",va="center")
        ax.axis("off"); return fig

    # Population: all players in same role group with 3+ seasons
    pop = df_career[
        (df_career["_role_key"] == role_k) &
        (df_career["seasons_data"] >= 2) &
        df_career["career_score"].notna() &
        df_career["Age"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0a0f1c")
    ax.set_facecolor("#111827")

    # Population scatter
    if not pop.empty:
        ax.scatter(pop["Age"], pop["career_score"], color="#1f2937", s=20,
                   alpha=0.7, zorder=1, label="All players (same role)")
        # Population trend line
        try:
            ages_p  = pop["Age"].astype(float).values
            scores_p = pop["career_score"].astype(float).values
            valid_m = ~(np.isnan(ages_p)|np.isnan(scores_p))
            if valid_m.sum() >= 5:
                coeffs = np.polyfit(ages_p[valid_m], scores_p[valid_m], 2)
                x_fit  = np.linspace(ages_p[valid_m].min(), ages_p[valid_m].max(), 100)
                y_fit  = np.polyval(coeffs, x_fit)
                ax.plot(x_fit, y_fit, color="#374151", linewidth=1.5,
                        linestyle="--", zorder=2, label="Population curve", alpha=0.6)
        except Exception:
            pass

    # This player
    history = player_row.get("_history", [])
    proj    = player_row.get("projection", {}) or {}
    pts = [(h["Age"], h["_ls_adj_score"]) for h in history
           if pd.notna(h.get("_ls_adj_score")) and pd.notna(h.get("Age"))]
    if pts:
        ages_pl  = [p[0] for p in pts]
        scores_pl= [p[1] for p in pts]
        ax.scatter(ages_pl, scores_pl, color="#f59e0b", s=80, zorder=5, label=player_row["Player"])
        ax.plot(ages_pl, scores_pl, color="#f59e0b", linewidth=2.0, zorder=4)

        ps = proj.get("projected_score"); pa = proj.get("proj_age")
        if ps and pa and pd.notna(ps):
            ax.scatter([pa],[ps], color="#ef4444", s=120, zorder=6, marker="*",
                       label=f"Projected: {ps:.0f} @ {pa:.0f}")
            ax.plot([ages_pl[-1], pa],[scores_pl[-1], ps],
                    color="#ef4444", linewidth=1.5, linestyle="--", alpha=0.8, zorder=5)

    # Level lines
    for lvl, lbl in [(85,"Premier League"),(72,"Championship"),(60,"League One"),(48,"League Two")]:
        ax.axhline(lvl, color="#1f2937", linewidth=0.8, linestyle=":", zorder=0)
        ax.text(15.2, lvl+0.6, lbl, fontsize=7, color="#4b5563",
                va="bottom", fontfamily="Montserrat")

    ax.set_xlabel("Age", fontsize=9, color="#9ca3af")
    ax.set_ylabel("Adj. Score", fontsize=9, color="#9ca3af")
    ax.set_xlim(14, 40); ax.set_ylim(0, 105)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#374151")
    ax.set_title(f"Predictive Model  —  {player_row['Player']} vs {role_k} population",
                 fontsize=10, color="#e5e7eb", pad=8, fontfamily="Montserrat", fontweight=700)
    ax.legend(fontsize=7.5, facecolor="#1f2937", labelcolor="#e5e7eb", framealpha=0.8)
    ax.grid(axis="both", color="#1f2937", linewidth=0.5, zorder=0)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# METRIC DISPLAY COLUMNS (for season breakdown dropdown)
# ══════════════════════════════════════════════════════════════
METRIC_GROUPS = {
    "Attacking": [
        "Goals","Assists","xG","xA","Non-penalty goals per 90","xG per 90",
        "Shots per 90","Shots on target, %","Touches in box per 90",
        "Crosses per 90","Accurate crosses, %",
    ],
    "Passing": [
        "Passes per 90","Accurate passes, %","Forward passes per 90",
        "Accurate forward passes, %","Progressive passes per 90",
        "Passes to final third per 90","Accurate long passes, %",
        "Deep completions per 90","Smart passes per 90","Key passes per 90",
        "xA per 90","Passes to penalty area per 90",
    ],
    "Carrying": [
        "Dribbles per 90","Successful dribbles, %",
        "Progressive runs per 90","Accelerations per 90",
    ],
    "Defending": [
        "Defensive duels per 90","Defensive duels won, %",
        "PAdj Interceptions","Aerial duels per 90","Aerial duels won, %",
        "Shots blocked per 90",
    ],
}


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
st.session_state.setdefault("shortlist", {})
st.session_state.setdefault("profile_player", None)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="font-family:Montserrat,sans-serif;font-size:18px;font-weight:900;'
        'color:#fff;letter-spacing:.04em;margin-bottom:4px;">JAMESCITY</div>'
        '<div style="font-size:10px;font-weight:700;letter-spacing:.18em;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:16px;">CAREER INTELLIGENCE</div>',
        unsafe_allow_html=True
    )
    page = st.radio("", ["🔍 Search", "👤 Profile", "⭐ Shortlist"], label_visibility="collapsed")
    st.divider()
    sl_count = len(st.session_state.get("shortlist", {}))
    st.caption(f"Shortlist: {sl_count} player{'s' if sl_count != 1 else ''}")

# ══════════════════════════════════════════════════════════════
# LOAD + COMPUTE
# ══════════════════════════════════════════════════════════════
with st.spinner("Loading CSV..."):
    df_raw = load_data()

if df_raw.empty:
    st.error(f"Cannot find **{CSV_NAME}**. Place it in the same folder as this script.")
    st.stop()

with st.spinner("Computing scores..."):
    df_scored  = compute_all_scores(df_raw)

with st.spinner("Building career profiles..."):
    df_career  = build_career_profiles(df_scored)

if df_career.empty:
    st.error("No scoreable players found — check CSV has position and metric columns.")
    st.stop()


# ══════════════════════════════════════════════════════════════
# ── PAGE 1: SEARCH ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
if page == "🔍 Search":
    st.markdown(
        '<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:16px;">PLAYER SEARCH</div>',
        unsafe_allow_html=True
    )

    # ── Filters ──────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2,2,2])
    with fc1:
        pos_filter = st.multiselect(
            "Position group",
            list(POS_GROUPS.keys()),
            default=list(POS_GROUPS.keys()),
        )
    with fc2:
        all_leagues = sorted(df_career["League"].dropna().unique())
        league_filter = st.multiselect("Current league", all_leagues, default=all_leagues)
    with fc3:
        traj_filter = st.multiselect(
            "Trajectory",
            ["Rising","Peaking","Declining","Breakout","Unknown"],
            default=["Rising","Peaking","Declining","Breakout","Unknown"],
        )

    fc4, fc5, fc6 = st.columns([2,2,2])
    with fc4:
        age_range = st.slider("Age", 15, 40, (16, 32))
    with fc5:
        min_seasons = st.slider("Min seasons of data", 1, 8, 1)
    with fc6:
        min_rating = st.slider("Min rating", 0, 90, 0)

    name_q   = st.text_input("Search name", "", placeholder="Player name...")
    sort_by  = st.selectbox("Sort by", ["career_score","potential","Age","seasons_data"], index=0)

    # ── Apply filters ─────────────────────────────────────────
    allowed_toks = {tok for grp in pos_filter for tok in POS_GROUPS.get(grp, [])}
    dfc = df_career.copy()
    if allowed_toks:
        dfc = dfc[dfc["_pos_tok"].isin(allowed_toks)]
    if league_filter:
        dfc = dfc[dfc["League"].isin(league_filter)]
    if traj_filter:
        dfc = dfc[dfc["trajectory"].isin(traj_filter)]
    dfc = dfc[(dfc["Age"] >= age_range[0]) & (dfc["Age"] <= age_range[1])]
    dfc = dfc[dfc["seasons_data"] >= min_seasons]
    dfc = dfc[dfc["career_score"].fillna(0) >= min_rating]
    if name_q.strip():
        dfc = dfc[dfc["Player"].str.contains(name_q.strip(), case=False, na=False)]
    dfc = dfc.sort_values(sort_by, ascending=(sort_by == "Age"), na_position="last")

    st.markdown(
        f'<div style="font-size:10px;color:#6b7280;font-weight:600;'
        f'letter-spacing:.08em;margin-bottom:12px;">'
        f'{len(dfc)} PLAYERS FOUND</div>',
        unsafe_allow_html=True
    )

    shortlist = st.session_state.get("shortlist", {})
    for i, (_, row) in enumerate(dfc.head(100).iterrows()):
        html = player_card_html(row)
        st.markdown(html, unsafe_allow_html=True)

        btn_c1, btn_c2, btn_c3 = st.columns([2, 1, 1])
        with btn_c2:
            wid = str(row["Wyscout ID"])
            on_sl = wid in shortlist
            if st.button("★ Remove" if on_sl else "☆ Save", key=f"sl_s_{i}_{wid}"):
                if on_sl: shortlist.pop(wid, None)
                else:
                    shortlist[wid] = row.to_dict()
                st.session_state["shortlist"] = shortlist
                st.rerun()
        with btn_c3:
            if st.button("👤 Profile", key=f"prof_s_{i}_{wid}"):
                st.session_state["profile_player"] = str(row["Wyscout ID"])
                st.rerun()

    if len(dfc) > 100:
        st.info(f"Showing top 100 of {len(dfc)}. Use filters to narrow results.")


# ══════════════════════════════════════════════════════════════
# ── PAGE 2: PROFILE ──────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
elif page == "👤 Profile":
    st.markdown(
        '<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:16px;">PLAYER PROFILE</div>',
        unsafe_allow_html=True
    )

    player_names = sorted(df_career["Player"].dropna().unique())
    pre_id = st.session_state.get("profile_player")
    default_idx = 0
    if pre_id:
        pre_rows = df_career[df_career["Wyscout ID"].astype(str) == str(pre_id)]
        if not pre_rows.empty:
            pname = pre_rows.iloc[0]["Player"]
            if pname in player_names:
                default_idx = player_names.index(pname)

    picked = st.selectbox("Select player", player_names, index=default_idx)
    prow = df_career[df_career["Player"] == picked].iloc[0]

    wid        = prow["Wyscout ID"]
    trajectory = str(prow["trajectory"])
    projection = prow["projection"] or {}
    history    = prow["_history"] or []
    cs         = float(prow.get("career_score", np.nan))
    pot        = float(prow.get("potential", np.nan))
    pos        = str(prow.get("_pos_tok",""))

    # ── Header card ──────────────────────────────────────────
    cs_col  = rating_color(cs)
    cs_fg   = "#000" if pd.notna(cs) and cs >= 44 else "#fff"
    pot_col = rating_color(pot)
    pot_fg  = "#000" if pd.notna(pot) and pot >= 44 else "#fff"
    flag    = _flag_img(str(prow.get("Birth country","")))
    badge   = _team_badge_html(str(prow.get("Team","")), size=32)
    ll      = _league_logo_html(str(prow.get("League","")), size=20)
    traj_color, traj_icon = TRAJECTORY_LABELS.get(trajectory, ("#6b7280","?"))
    proj_lvl = projected_level(cs) if pd.notna(cs) else "—"
    ps = projection.get("projected_score"); pa = projection.get("proj_age")
    pk = projection.get("peak_age"); conf = projection.get("confidence","?")

    st.markdown(f"""
<div style="background:#0f1628;border:1px solid #1a2540;border-radius:16px;
            padding:22px 28px;margin-bottom:20px;display:flex;align-items:flex-start;
            gap:24px;flex-wrap:wrap;font-family:Montserrat,sans-serif;">
  <!-- SCORE BADGES -->
  <div style="display:flex;flex-direction:column;gap:8px;align-items:center;flex-shrink:0;">
    <div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:.10em;text-transform:uppercase;">RATING</div>
    <div style="background:{cs_col};color:{cs_fg};font-size:42px;font-weight:900;
                padding:10px 20px;border-radius:10px;min-width:80px;text-align:center;">{fmt2(cs)}</div>
    <div style="font-size:9px;color:#9ca3af;font-weight:700;letter-spacing:.10em;text-transform:uppercase;margin-top:4px;">POTENTIAL</div>
    <div style="background:{pot_col};color:{pot_fg};font-size:28px;font-weight:900;
                padding:6px 16px;border-radius:8px;min-width:70px;text-align:center;">{fmt2(pot)}</div>
  </div>
  <!-- INFO -->
  <div style="flex:1;min-width:200px;">
    <div style="font-size:28px;font-weight:900;color:#fff;letter-spacing:.02em;line-height:1.1;margin-bottom:8px;">
      {prow['Player'].upper()}
    </div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
      {_pos_chip_html(pos)}
      <span style="background:{traj_color};color:#000;padding:2px 9px;border-radius:99px;font-size:10px;font-weight:800;">{traj_icon} {trajectory}</span>
      <span style="color:#9ca3af;font-size:11px;">{int(prow.get('seasons_data',0))} seasons data</span>
    </div>
    <div style="font-size:12px;color:#9ca3af;margin-bottom:10px;">
      {badge}{flag}{prow.get('Team','')}
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      {ll}<span style="color:#cbd5e1;font-weight:600;">{_league_short(str(prow.get('League','')))} ({prow.get('League','')})</span>
      <span style="color:#374151;">&nbsp;·&nbsp;</span>
      Age <span style="color:#fff;font-weight:700;">{int(prow['Age']) if pd.notna(prow.get('Age')) else '?'}</span>
    </div>
    <div style="display:flex;gap:20px;flex-wrap:wrap;">
      <div>
        <div style="font-size:9px;color:#6b7280;font-weight:700;letter-spacing:.1em;text-transform:uppercase;">Projected level</div>
        <div style="font-size:15px;color:#e5e7eb;font-weight:800;">{proj_lvl}</div>
      </div>
      {'<div><div style="font-size:9px;color:#6b7280;font-weight:700;letter-spacing:.1em;text-transform:uppercase;">2yr Projection</div><div style="font-size:15px;color:#f59e0b;font-weight:800;">'+str(ps)+' @ age '+str(pa)+'</div></div>' if ps and pa else ''}
      {'<div><div style="font-size:9px;color:#6b7280;font-weight:700;letter-spacing:.1em;text-transform:uppercase;">Est. peak age</div><div style="font-size:15px;color:#a78bfa;font-weight:800;">'+str(pk)+'</div></div>' if pk else ''}
      <div>
        <div style="font-size:9px;color:#6b7280;font-weight:700;letter-spacing:.1em;text-transform:uppercase;">Confidence</div>
        <div style="font-size:13px;color:#9ca3af;font-weight:700;">{conf}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Shortlist button
    shortlist = st.session_state.get("shortlist", {})
    on_sl = str(wid) in shortlist
    if st.button("★ Remove from shortlist" if on_sl else "☆ Add to shortlist", key="prof_sl"):
        if on_sl: shortlist.pop(str(wid), None)
        else: shortlist[str(wid)] = prow.to_dict()
        st.session_state["shortlist"] = shortlist
        st.rerun()

    st.divider()

    # ── Charts ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Career History", "🎯 Projection", "🔬 Predictive Model"])

    with tab1:
        fig1 = career_chart(history, prow["Player"], trajectory)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with tab2:
        fig2 = projection_chart(history, projection, prow["Player"])
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with tab3:
        fig3 = regression_chart(df_career, prow)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.caption(
            f"Population = all {str(prow.get('_role_key',''))} players with 2+ seasons. "
            f"Player shown in gold. Star = 2-year projection."
        )

    # ── Season-by-season table ────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="font-size:10px;font-weight:900;letter-spacing:.16em;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:10px;">SEASON HISTORY</div>',
        unsafe_allow_html=True
    )

    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        hist_df = hist_df.sort_values(
            "Season",
            key=lambda s: s.map(lambda x: SEASON_ORDER.index(x) if x in SEASON_ORDER else 99)
        )
        base_cols = ["Season","League","Team","Age","Minutes played","Goals","Assists","_ls_adj_score","_raw_score"]
        base_cols = [c for c in base_cols if c in hist_df.columns]
        disp = hist_df[base_cols].rename(columns={"_ls_adj_score":"Adj Score","_raw_score":"Raw Score"})
        for nc in ["Adj Score","Raw Score","Age","Minutes played","Goals","Assists"]:
            if nc in disp.columns:
                disp[nc] = pd.to_numeric(disp[nc], errors="coerce").round(1)

        st.dataframe(
            disp, use_container_width=True, hide_index=True,
            column_config={
                "Adj Score": st.column_config.ProgressColumn("Adj Score", min_value=0, max_value=100, format="%.0f"),
                "Raw Score": st.column_config.ProgressColumn("Raw Score", min_value=0, max_value=100, format="%.0f"),
            }
        )

    # ── Season metrics dropdown ───────────────────────────────
    st.divider()
    st.markdown(
        '<div style="font-size:10px;font-weight:900;letter-spacing:.16em;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:10px;">SEASON METRICS</div>',
        unsafe_allow_html=True
    )

    avail_seasons = [h["Season"] for h in history if pd.notna(h.get("_ls_adj_score"))]
    if avail_seasons:
        sel_season = st.selectbox("Season", avail_seasons, key="prof_season")
        sel_h = next((h for h in history if h["Season"] == sel_season), None)
        if sel_h:
            met_group = st.selectbox("Metric group", list(METRIC_GROUPS.keys()), key="prof_mgrp")
            met_cols  = [m for m in METRIC_GROUPS[met_group] if m in sel_h and pd.notna(sel_h.get(m))]
            if met_cols:
                # Compare vs same-season same-league pool
                pool = df_scored[
                    (df_scored["Season"]   == sel_season) &
                    (df_scored["League"]   == sel_h["League"]) &
                    (df_scored["_role_key"] == str(prow.get("_role_key","")))
                ]
                rows_out = []
                for m in met_cols:
                    val = sel_h.get(m, np.nan)
                    if pd.isna(val): continue
                    pool_vals = pd.to_numeric(pool[m], errors="coerce").dropna() if m in pool.columns else pd.Series(dtype=float)
                    if not pool_vals.empty:
                        pct = float((pool_vals < val).sum() + 0.5*(pool_vals == val).sum()) / len(pool_vals) * 100
                    else:
                        pct = np.nan
                    rows_out.append({"Metric": m, "Value": round(float(val),2), "Percentile": round(pct,0) if pd.notna(pct) else None})

                if rows_out:
                    mdf = pd.DataFrame(rows_out)
                    st.dataframe(
                        mdf, use_container_width=True, hide_index=True,
                        column_config={
                            "Percentile": st.column_config.ProgressColumn(
                                "Percentile (vs league+pos)", min_value=0, max_value=100, format="%.0f"
                            )
                        }
                    )
            else:
                st.info("No metric data for this season/group.")
    else:
        st.info("No scored season data available for metric breakdown.")


# ══════════════════════════════════════════════════════════════
# ── PAGE 3: SHORTLIST ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
elif page == "⭐ Shortlist":
    st.markdown(
        '<div style="font-size:11px;font-weight:900;letter-spacing:.18em;color:#6b7280;'
        'text-transform:uppercase;margin-bottom:16px;">SHORTLIST</div>',
        unsafe_allow_html=True
    )

    shortlist = st.session_state.get("shortlist", {})

    if not shortlist:
        st.info("No players saved. Use Player Search or Profile to save players.")
    else:
        sl_ids  = list(shortlist.keys())
        sl_rows = df_career[df_career["Wyscout ID"].astype(str).isin(sl_ids)].copy()
        sl_rows = sl_rows.sort_values("career_score", ascending=False, na_position="last")

        st.markdown(
            f'<div style="font-size:10px;color:#6b7280;font-weight:600;letter-spacing:.08em;margin-bottom:12px;">'
            f'{len(sl_rows)} PLAYERS</div>', unsafe_allow_html=True
        )

        for i, (_, row) in enumerate(sl_rows.iterrows()):
            st.markdown(player_card_html(row), unsafe_allow_html=True)
            b1, b2, b3 = st.columns([3,1,1])
            with b2:
                if st.button("★ Remove", key=f"sl_r_{i}"):
                    shortlist.pop(str(row["Wyscout ID"]), None)
                    st.session_state["shortlist"] = shortlist
                    st.rerun()
            with b3:
                if st.button("👤 Profile", key=f"sl_p_{i}"):
                    st.session_state["profile_player"] = str(row["Wyscout ID"])
                    st.rerun()

        st.divider()

        export_cols = ["Player","Team","League","Position","Age","career_score","potential","trajectory","seasons_data"]
        export_cols = [c for c in export_cols if c in sl_rows.columns]
        edf = sl_rows[export_cols].rename(columns={
            "career_score":"Rating","potential":"Potential",
            "trajectory":"Trajectory","seasons_data":"Seasons"
        })
        st.download_button(
            "⬇ Export shortlist CSV",
            data=edf.to_csv(index=False).encode(),
            file_name="shortlist_british.csv",
            mime="text/csv",
        )
        if st.button("🗑 Clear shortlist", type="secondary"):
            st.session_state["shortlist"] = {}
            st.rerun()
