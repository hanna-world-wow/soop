from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AD Performance Report", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main .block-container {
    background: #F7F9FC !important;
    color: #111827 !important;
    font-family: "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", sans-serif !important;
}
[data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 2.0rem !important;
    padding-bottom: 2.6rem !important;
}
.report-header { background:#fff; border:1px solid #E5EAF2; border-radius:14px; padding:24px 28px; margin-bottom:20px; }
.report-header h1 { margin:0; color:#165DFF; font-size:24px; font-weight:700; }
.report-header p { margin:8px 0 0 0; color:#4B5563; font-size:13px; }
.sec-title { font-size:13px; font-weight:700; color:#1D4ED8; margin:24px 0 12px 0; letter-spacing:0.3px; }
.kpi-wrap {
    background:#fff; border:1px solid #E5EAF2; border-radius:12px;
    padding:14px 14px; min-height:128px; max-height:128px; height:128px;
    min-width:0; overflow:hidden; display:flex; flex-direction:column;
    justify-content:space-between;
}
.kpi-label { font-size:12px; color:#6B7280; font-weight:600; margin-bottom:6px; }
.kpi-val { font-size:28px; color:#0F172A; font-weight:700; line-height:1.1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi-delta-pos { color:#059669; font-size:12px; margin-top:6px; }
.kpi-delta-neg { color:#DC2626; font-size:12px; margin-top:6px; }
.kpi-delta-neu { color:#6B7280; font-size:12px; margin-top:6px; }
.insight { background:#EFF6FF; border:1px solid #BFDBFE; border-left:4px solid #1D4ED8; border-radius:8px; padding:10px 12px; margin:6px 0; font-size:13px; }
.insight-warn { background:#FFFBEB; border:1px solid #FCD34D; border-left:4px solid #D97706; border-radius:8px; padding:10px 12px; margin:6px 0; font-size:13px; }
.stTabs [data-baseweb="tab-list"] { background:#fff; border:1px solid #E5EAF2; border-radius:10px; padding:4px; gap:6px; }
.stTabs [data-baseweb="tab"] { font-size:13px; border-radius:8px; padding:8px 12px; }
</style>
""",
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "캠페인명", "일자", "노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수",
    "CTR", "CTR(전체)", "VTR", "광고비", "eCPM", "CPC", "CPV", "연도", "월", "광고상품명_정리",
    "구분", "프로모션별", "광고주",
]
NUMERIC_COLUMNS = [
    "노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수",
    "CTR", "CTR(전체)", "VTR", "광고비", "eCPM", "CPC", "CPV", "연도", "월",
]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
COLORS = ["#2563EB", "#1E40AF", "#16A34A", "#D97706", "#7C3AED", "#0891B2", "#DB2777"]

INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "금융·보험": [], "통신·IT": ["소니", "폴라로이드", "ASL로지텍콜라보"], "유통·이커머스": ["네이버스토어", "핫딜", "꽃다발"],
    "자동차": [],
    "식음료": ["과자", "킷캣", "킷켓", "암소갈비", "동원참치", "에브리워터", "네꼬닭", "사과당x여우티", "오밀당x중앙해장"],
    "엔터·미디어": [], "공공·기관": [], "여행·숙박": [],
    "패션·뷰티": ["캘빈클라인", "아미", "아페쎄", "스투시", "듀이셀", "바세린", "헤넬"],
    "의료·헬스": ["광동멀티비타민", "데이팩", "정원삼", "천의삼"],
    "특집": ["설기획전", "커부해", "설선물준비했설", "숲다이어리"],
    "굿즈": ["마플샵", "포토북", "감스트굿즈", "민교교록앵콜", "lck이벤트"],
    "기타": [],
}


@st.cache_data(show_spinner=False)
def load_csv(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


@st.cache_data(show_spinner=False)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_product_name(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    if "_" in s:
        return s
    s = re.sub(r"\([^)]*\)$", "", s)
    return s.strip()


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    out["일자"] = pd.to_datetime(out["일자"], errors="coerce")
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["요일"] = out["일자"].dt.weekday.map(DAY_MAP)
    out["연도"] = out["연도"].fillna(out["일자"].dt.year)
    out["월"] = out["월"].fillna(out["일자"].dt.month)
    out["연월"] = pd.to_datetime(dict(year=out["연도"].astype("Int64"), month=out["월"].astype("Int64"), day=1), errors="coerce")
    out["광고상품명_원본"] = out["광고상품명_정리"].astype("string")
    out["광고상품명_정리"] = out["광고상품명_정리"].astype("string").fillna("미분류").apply(normalize_product_name)
    out["프로모션명"] = out["프로모션별"].astype("string").fillna(out["캠페인명"].astype("string").fillna("미분류"))
    out["광고주"] = out["광고주"].astype("string").fillna("미분류")
    for c in ["노출수", "클릭수", "총클릭수", "동영상조회수", "컴패니언배너클릭수"]:
        out[c] = out[c].fillna(0)
    out["총클릭수"] = out["총클릭수"].where(out["총클릭수"] > 0, out["클릭수"])

    def classify(v: str) -> str:
        u = re.sub(r"\s+", "", str(v)).upper()
        for ind, kws in INDUSTRY_KEYWORDS.items():
            if any(re.sub(r"\s+", "", k).upper() in u for k in kws):
                return ind
        return "기타"

    out["업종"] = out["광고주"].apply(classify)
    return out


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def aggregate(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False, as_index=False)[["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수"]].sum(min_count=1).fillna(0)
    g["CTR"] = safe_div(g["클릭수"], g["노출수"])
    g["CTR_total"] = safe_div(g["총클릭수"], g["노출수"])
    g["VTR"] = safe_div(g["동영상조회수"], g["노출수"])
    total = g["노출수"].sum()
    g["노출비중"] = g["노출수"] / total if total > 0 else 0
    return g


def compute_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out[(out["노출수"] > 0) & out["CTR_total"].notna()]
    if len(valid) < 2 or valid["CTR_total"].std(ddof=0) == 0:
        out["eff_score"] = np.nan
        out["eff_grade"] = "N/A"
        return out
    mu, sd = valid["CTR_total"].mean(), valid["CTR_total"].std(ddof=0)
    out["eff_score"] = (out["CTR_total"] - mu) / sd
    out["eff_grade"] = np.where(out["eff_score"] >= 1, "S", np.where(out["eff_score"] >= 0.3, "A", np.where(out["eff_score"] >= -0.3, "B", "C")))
    out.loc[out["eff_score"].isna(), "eff_grade"] = "N/A"
    return out


def fmt_pct(v) -> str:
    return "N/A" if pd.isna(v) else f"{v * 100:.2f}%"


def fmt_table_number(v) -> str:
    return "N/A" if pd.isna(v) else f"{float(v):,.0f}"


def fmt_kor_unit(v) -> str:
    if pd.isna(v):
        return "N/A"
    n = float(v)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 100_000_000:
        return f"{sign}{n/100_000_000:.1f}억"
    if n >= 10_000:
        return f"{sign}{n/10_000:.1f}만"
    if n >= 1_000:
        return f"{sign}{n/1_000:.1f}천"
    return f"{sign}{n:,.0f}"


def fmt_kpi_compact(v) -> str:
    return fmt_kor_unit(v)


def ellipsis(v: str, n: int = 16) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def axis_ticks_from_series(s: pd.Series, bins: int = 5) -> Tuple[List[float], List[str]]:
    if s.empty:
        return [0], ["0"]
    vmax = float(np.nanmax(np.abs(s.values))) if s.notna().any() else 0
    if vmax == 0:
        return [0], ["0"]
    vals = np.linspace(0, vmax, bins)
    return vals.tolist(), [fmt_kor_unit(v) for v in vals]


def nice_axis_max(vmax: float, pad: float = 0.15) -> float:
    if pd.isna(vmax) or vmax <= 0:
        return 1.0
    target = vmax * (1 + pad)
    base = 10 ** np.floor(np.log10(target))
    ratio = target / base
    if ratio <= 1.5:
        nice = 1.5 * base
    elif ratio <= 2:
        nice = 2 * base
    elif ratio <= 2.5:
        nice = 2.5 * base
    elif ratio <= 5:
        nice = 5 * base
    else:
        nice = 10 * base
    return float(nice)


def make_kor_ticks(vmax: float, bins: int = 5) -> Tuple[List[float], List[str]]:
    top = nice_axis_max(vmax)
    vals = np.linspace(0, top, bins)
    return vals.tolist(), [fmt_kor_unit(v) for v in vals]


def make_pct_ticks(vmax: float, bins: int = 5) -> Tuple[List[float], List[str]]:
    top = nice_axis_max(vmax, pad=0.2)
    vals = np.linspace(0, top, bins)
    return vals.tolist(), [fmt_pct(v) for v in vals]


def apply_meta_theme(fig: go.Figure, title: str = "", h: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title=dict(text=title, font=dict(size=15, family="Pretendard, Noto Sans KR, Apple SD Gothic Neo, sans-serif")),
        font=dict(family="Pretendard, Noto Sans KR, Apple SD Gothic Neo, sans-serif", size=12, color="#111827"),
        margin=dict(l=28, r=28, t=72, b=140),
        legend=dict(orientation="h", y=-0.32, yanchor="top", x=0, xanchor="left", itemwidth=80),
        height=h,
    )
    return fig


def apply_month_axis(fig: go.Figure, months: List[str], angle: int = -28):
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=months,
        tickmode="array",
        tickvals=months,
        ticktext=months,
        tickangle=angle,
        automargin=True,
    )


def summarize_period(df: pd.DataFrame) -> Dict[str, float]:
    imp = df["노출수"].sum()
    clk = df["클릭수"].sum()
    tclk = df["총클릭수"].sum()
    vw = df["동영상조회수"].sum()
    return {
        "노출수": imp,
        "클릭수": clk,
        "총클릭수": tclk,
        "CTR_total": (tclk / imp) if imp > 0 else np.nan,
        "동영상조회수": vw,
        "VTR": (vw / imp) if imp > 0 else np.nan,
        "광고상품수": df.loc[df["노출수"] > 0, "광고상품명_정리"].nunique(),
        "캠페인수": df.loc[df["노출수"] > 0, "캠페인명"].nunique(),
    }


def period_product_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, min_imp: int) -> pd.DataFrame:
    a = aggregate(df_a, ["광고상품명_정리"])[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR"]]
    b = aggregate(df_b, ["광고상품명_정리"])[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR"]]
    a = a.rename(columns={"노출수": "A_노출수", "총클릭수": "A_총클릭수", "CTR_total": "A_CTR", "VTR": "A_VTR"})
    b = b.rename(columns={"노출수": "B_노출수", "총클릭수": "B_총클릭수", "CTR_total": "B_CTR", "VTR": "B_VTR"})
    m = a.merge(b, on="광고상품명_정리", how="outer").fillna(0)
    m["노출 증감"] = m["B_노출수"] - m["A_노출수"]
    m["클릭 증감"] = m["B_총클릭수"] - m["A_총클릭수"]
    m["CTR 변화(%p)"] = (m["B_CTR"] - m["A_CTR"]) * 100
    m["VTR 변화(%p)"] = (m["B_VTR"] - m["A_VTR"]) * 100
    m["해석상태"] = np.where((m["A_노출수"] < min_imp) | (m["B_노출수"] < min_imp), "표본부족", "분석가능")
    return m.sort_values("노출 증감", ascending=False)


def promotion_product_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, min_imp: int) -> pd.DataFrame:
    a = aggregate(df_a, ["광고상품명_정리"])[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR"]]
    b = aggregate(df_b, ["광고상품명_정리"])[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR"]]
    a = a.rename(columns={"노출수": "A_노출수", "총클릭수": "A_총클릭수", "CTR_total": "A_CTR", "VTR": "A_VTR"})
    b = b.rename(columns={"노출수": "B_노출수", "총클릭수": "B_총클릭수", "CTR_total": "B_CTR", "VTR": "B_VTR"})
    m = a.merge(b, on="광고상품명_정리", how="outer").fillna(0)
    m["노출 증감"] = m["B_노출수"] - m["A_노출수"]
    m["총클릭 증감"] = m["B_총클릭수"] - m["A_총클릭수"]
    m["CTR 변화(%p)"] = (m["B_CTR"] - m["A_CTR"]) * 100
    m["VTR 변화(%p)"] = (m["B_VTR"] - m["A_VTR"]) * 100
    m["해석상태"] = np.where((m["A_노출수"] < min_imp) | (m["B_노출수"] < min_imp), "표본부족", "분석가능")
    return m.sort_values("노출 증감", ascending=False)


def delta_sentence_count(a: float, b: float) -> str:
    d = b - a
    if abs(d) < 1e-9:
        return "A와 B가 동일"
    return f"B가 A보다 {fmt_kor_unit(abs(d))} {'큼' if d > 0 else '적음'}"


def delta_sentence_pp(a: float, b: float) -> str:
    if pd.isna(a) or pd.isna(b):
        return "비교 불가"
    d = (b - a) * 100
    if abs(d) < 1e-12:
        return "A와 B가 동일"
    return f"B가 A보다 {abs(d):.2f}%p {'높음' if d > 0 else '낮음'}"


def compute_concentration_metrics(prod_df: pd.DataFrame) -> Dict[str, float]:
    s = prod_df.sort_values("노출수", ascending=False)["노출수"].astype(float)
    total = s.sum()
    if total <= 0:
        return {"top3": 0, "top5": 0, "top10": 0, "hhi": 0, "hhi_10000": 0}
    share = s / total
    hhi = float((share ** 2).sum())
    return {
        "top3": float(share.head(3).sum()),
        "top5": float(share.head(5).sum()),
        "top10": float(share.head(10).sum()),
        "hhi": hhi,
        "hhi_10000": hhi * 10000,
    }


def sec(title: str):
    st.markdown(f"<div class='sec-title'>{title}</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", delta_good=None):
    cls = "kpi-delta-neu" if delta_good is None else ("kpi-delta-pos" if delta_good else "kpi-delta-neg")
    st.markdown(f"<div class='kpi-wrap'><div class='kpi-label'>{label}</div><div class='kpi-val'>{value}</div><div class='{cls}'>{delta}</div></div>", unsafe_allow_html=True)


def insight(text: str, warn: bool = False):
    st.markdown(f"<div class='{'insight-warn' if warn else 'insight'}'>{text}</div>", unsafe_allow_html=True)


@dataclass
class Filters:
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    years: List[int]
    months: List[int]
    weekdays: List[str]
    categories: List[str]
    industries: List[str]
    products: List[str]
    campaigns: List[str]


@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()
    if f.date_min is not None:
        out = out[out["일자"] >= f.date_min]
    if f.date_max is not None:
        out = out[out["일자"] <= f.date_max]
    if f.years:
        out = out[out["연도"].astype("Int64").isin(f.years)]
    if f.months:
        out = out[out["월"].astype("Int64").isin(f.months)]
    if f.weekdays:
        out = out[out["요일"].isin(f.weekdays)]
    if f.categories:
        out = out[out["구분"].isin(f.categories)]
    if f.industries:
        out = out[out["업종"].isin(f.industries)]
    if f.products:
        out = out[out["광고상품명_정리"].isin(f.products)]
    if f.campaigns:
        out = out[out["캠페인명"].isin(f.campaigns)]
    return out


def period_delta(df: pd.DataFrame) -> Dict:
    m = sorted(df["연월"].dropna().unique())
    if len(m) < 2:
        return {}
    cur = df[df["연월"] == m[-1]]
    prev = df[df["연월"] == m[-2]]
    res = {}
    for c in ["노출수", "총클릭수", "동영상조회수"]:
        cc, pp = cur[c].sum(), prev[c].sum()
        res[c] = None if pp == 0 else (cc - pp) / pp
    cctr = (cur["총클릭수"].sum() / cur["노출수"].sum()) if cur["노출수"].sum() > 0 else np.nan
    pctr = (prev["총클릭수"].sum() / prev["노출수"].sum()) if prev["노출수"].sum() > 0 else np.nan
    res["CTR_total"] = None if pd.isna(pctr) or pctr == 0 else (cctr - pctr) / pctr
    res["CTR_pp"] = None if pd.isna(pctr) else (cctr - pctr)
    return res


def build_volume_click_ctr_chart(dfv: pd.DataFrame, x: str, title: str) -> go.Figure:
    ctr_series = dfv["CTR_total"].fillna(0)
    ctr_max = float(ctr_series.max()) if not ctr_series.empty else 0.0
    ctr_upper = max(0.01, nice_axis_max(ctr_max, pad=0.25))

    fig = go.Figure()
    fig.add_bar(
        x=dfv[x], y=dfv["노출수"], name="노출수", marker_color="#2563EB", yaxis="y",
        offsetgroup="imp", legendgroup="imp",
        hovertemplate="노출수: %{customdata}<extra></extra>", customdata=[fmt_kor_unit(v) for v in dfv["노출수"]]
    )
    fig.add_bar(
        x=dfv[x], y=dfv["총클릭수"], name="총클릭수", marker_color="#16A34A", yaxis="y2",
        offsetgroup="clk", legendgroup="clk",
        hovertemplate="총클릭수: %{customdata}<extra></extra>", customdata=[fmt_kor_unit(v) for v in dfv["총클릭수"]]
    )
    fig.add_scatter(
        x=dfv[x], y=dfv["CTR_total"], name="CTR_total", mode="lines+markers+text", line=dict(color="#D97706", width=2), marker=dict(size=7),
        text=[fmt_pct(v) for v in dfv["CTR_total"]], textposition="top center", hovertemplate="CTR_total: %{text}<extra></extra>",
        cliponaxis=False,
        yaxis="y3"
    )
    apply_meta_theme(fig, title, 360)
    y1v, y1t = make_kor_ticks(float(dfv["노출수"].max()) if not dfv.empty else 0)
    y2v, y2t = make_kor_ticks(float(dfv["총클릭수"].max()) if not dfv.empty else 0)
    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-28, automargin=True),
        yaxis=dict(title="노출수", tickmode="array", tickvals=y1v, ticktext=y1t, gridcolor="#EEF2F7", range=[0, y1v[-1]]),
        yaxis2=dict(title="총클릭수", overlaying="y", side="right", tickmode="array", tickvals=y2v, ticktext=y2t, showgrid=False, range=[0, y2v[-1]]),
        yaxis3=dict(
            overlaying="y",
            side="right",
            range=[0, ctr_upper],
            showticklabels=False,
            visible=False,
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


def main():
    with st.sidebar:
        upload = st.file_uploader("CSV 업로드", type=["csv"])
        path = st.text_input("또는 파일 경로")

    if upload is None and not path:
        st.markdown("<div class='report-header'><h1>📊 AD Performance Report</h1><p>사이드바에서 CSV를 업로드하세요.</p></div>", unsafe_allow_html=True)
        st.stop()

    raw = load_csv(upload.getvalue()) if upload else load_csv_path(path)
    df_raw = preprocess(raw)

    with st.sidebar:
        mn, mx = df_raw["일자"].min(), df_raw["일자"].max()
        rg = st.date_input("날짜 범위", value=(mn.date(), mx.date()) if pd.notna(mn) and pd.notna(mx) else ())
        years = sorted(df_raw["연도"].dropna().astype(int).unique().tolist())
        months = sorted(df_raw["월"].dropna().astype(int).unique().tolist())
        weekdays = [d for d in DAY_ORDER if d in set(df_raw["요일"].dropna().unique())]
        cats = sorted(df_raw["구분"].dropna().astype(str).unique().tolist())
        inds = sorted(df_raw["업종"].dropna().astype(str).unique().tolist())
        prods = sorted(df_raw["광고상품명_정리"].dropna().astype(str).unique().tolist())
        camps = sorted(df_raw["캠페인명"].dropna().astype(str).unique().tolist())
        fy = st.multiselect("연도", years, years)
        fm = st.multiselect("월", months, months)
        fw = st.multiselect("요일", weekdays, weekdays)
        fc = st.multiselect("구분", cats, cats)
        fi = st.multiselect("업종", inds, inds)
        fp = st.multiselect("광고상품", prods, prods)
        q = st.text_input("캠페인 검색")
        camps2 = [c for c in camps if q.lower() in c.lower()]
        fcam = st.multiselect("캠페인", camps2, camps2)
        min_imp = st.number_input("최소 노출 기준(min_imp)", min_value=0, value=10000, step=1000)

    d0 = pd.to_datetime(rg[0]) if rg and len(rg) > 0 else None
    d1 = pd.to_datetime(rg[1]) if rg and len(rg) > 1 else None
    f = Filters(d0, d1, fy, fm, fw, fc, fi, fp, fcam)
    df = apply_filters(df_raw, f)

    st.markdown(f"<div class='report-header'><h1>📊 AD Performance Report</h1><p>기간 {df['일자'].min().date()} ~ {df['일자'].max().date()} · 행 {len(df):,} · 상품 {df['광고상품명_정리'].nunique()} · 프로모션 {df['프로모션명'].nunique()}</p></div>", unsafe_allow_html=True)

    sec("전체 KPI 요약")
    total_imp = df["노출수"].sum()
    total_clk = df["클릭수"].sum()
    total_tclk = df["총클릭수"].sum()
    total_vw = df["동영상조회수"].sum()
    ctr_t = total_tclk / total_imp if total_imp else np.nan
    vtr = total_vw / total_imp if total_imp else np.nan
    dd = period_delta(df)

    cols = st.columns(6)
    for i, (lbl, val, dk) in enumerate([
        ("노출수", fmt_kpi_compact(total_imp), "노출수"),
        ("총클릭수", fmt_kpi_compact(total_tclk), "총클릭수"),
        ("클릭수", fmt_kpi_compact(total_clk), None),
        ("CTR(전체)", fmt_pct(ctr_t), "CTR_total"),
        ("동영상조회수", fmt_kpi_compact(total_vw), "동영상조회수"),
        ("VTR", fmt_pct(vtr), None),
    ]):
        with cols[i]:
            d = dd.get(dk) if dk else None
            s = "" if d is None else (f"▲ {abs(d)*100:.2f}%" if d > 0 else f"▼ {abs(d)*100:.2f}%")
            kpi_card(lbl, val, s, None if d is None else d > 0)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📦 상품 분석", "🔀 상품 × 프로모션", "📈 시계열", "🔁 프로모션 비교", "📅 요일·구분", "🏭 업종", "📋 Appendix"
    ])

    with tab1:
        agg = compute_efficiency(aggregate(df, ["광고상품명_정리"]))
        agg["표본부족"] = np.where(agg["노출수"] < min_imp, "표본부족", "분석가능")
        disp = agg[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR", "표본부족", "eff_grade"]].copy()
        disp.columns = ["상품", "노출수", "총클릭수", "CTR_total", "VTR", "해석상태", "효율등급"]
        disp["노출수"] = disp["노출수"].map(fmt_table_number)
        disp["총클릭수"] = disp["총클릭수"].map(fmt_table_number)
        disp["CTR_total"] = disp["CTR_total"].map(fmt_pct)
        disp["VTR"] = disp["VTR"].map(fmt_pct)
        sec("상품 성과 테이블")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        plot_df = agg.sort_values("노출수", ascending=False).head(15).copy()
        plot_df["상품축"] = plot_df["광고상품명_정리"].map(lambda x: ellipsis(x, 14))
        sec("TOP 15 상품 — 노출수 · 총클릭수 · CTR_total")
        st.plotly_chart(build_volume_click_ctr_chart(plot_df, "상품축", "볼륨 중심 비교 (CTR은 데이터레이블)"), use_container_width=True)

        st.write("")
        sec("버블 포트폴리오 매트릭스")
        show_low_sample = st.checkbox("표본부족 포함", value=False, key="bubble_low_sample")
        bubble_df = agg.copy() if show_low_sample else agg[agg["노출수"] >= min_imp].copy()
        if bubble_df.empty:
            st.warning("표시할 상품이 없습니다. (min_imp 또는 필터 확인)")
        else:
            med_x = bubble_df["노출수"].median()
            med_y = bubble_df["CTR_total"].median()
            x_max = max(float(bubble_df["노출수"].max()), 1)
            y_max = max(float(bubble_df["CTR_total"].max()), 0.01)
            y_min = max(0.0, float(bubble_df["CTR_total"].min()) * 0.9)

            fig_bubble = go.Figure()
            # 사분면 배경
            fig_bubble.add_shape(type="rect", x0=0, x1=med_x, y0=y_min, y1=med_y, fillcolor="rgba(239,68,68,0.05)", line=dict(width=0), layer="below")
            fig_bubble.add_shape(type="rect", x0=med_x, x1=x_max * 1.05, y0=y_min, y1=med_y, fillcolor="rgba(251,146,60,0.05)", line=dict(width=0), layer="below")
            fig_bubble.add_shape(type="rect", x0=0, x1=med_x, y0=med_y, y1=y_max * 1.15, fillcolor="rgba(59,130,246,0.06)", line=dict(width=0), layer="below")
            fig_bubble.add_shape(type="rect", x0=med_x, x1=x_max * 1.05, y0=med_y, y1=y_max * 1.15, fillcolor="rgba(16,185,129,0.07)", line=dict(width=0), layer="below")
            fig_bubble.add_vline(x=med_x, line=dict(color="#64748B", dash="dot"))
            fig_bubble.add_hline(y=med_y, line=dict(color="#64748B", dash="dot"))

            bubble_text = [ellipsis(v, 12) if i < 10 else "" for i, v in enumerate(bubble_df.sort_values("총클릭수", ascending=False)["광고상품명_정리"])]
            plot_b = bubble_df.sort_values("총클릭수", ascending=False).copy()
            fig_bubble.add_trace(go.Scatter(
                x=plot_b["노출수"], y=plot_b["CTR_total"], mode="markers+text", text=bubble_text,
                textposition="top center", name="광고상품",
                customdata=np.array([[n, fmt_kor_unit(i), fmt_kor_unit(c), fmt_pct(ct), fmt_pct(v)] for n, i, c, ct, v in zip(plot_b["광고상품명_정리"], plot_b["노출수"], plot_b["총클릭수"], plot_b["CTR_total"], plot_b["VTR"])], dtype=object),
                hovertemplate="상품명: %{customdata[0]}<br>노출수: %{customdata[1]}<br>총클릭수: %{customdata[2]}<br>CTR_total: %{customdata[3]}<br>VTR: %{customdata[4]}<extra></extra>",
                marker=dict(size=np.clip(np.sqrt(plot_b["총클릭수"].fillna(0)) * 2.5, 12, 55), color="#2563EB", opacity=0.55, line=dict(color="#1D4ED8", width=1)),
            ))
            fig_bubble.add_annotation(x=med_x * 0.45, y=med_y + (y_max - med_y) * 0.72, text="테스트 확대", showarrow=False, font=dict(size=11, color="#1E3A8A"))
            fig_bubble.add_annotation(x=med_x * 1.35, y=med_y + (y_max - med_y) * 0.72, text="확대 후보 ★", showarrow=False, font=dict(size=11, color="#065F46"))
            fig_bubble.add_annotation(x=med_x * 1.35, y=max(y_min, med_y * 0.35), text="개선 우선", showarrow=False, font=dict(size=11, color="#9A3412"))
            fig_bubble.add_annotation(x=med_x * 0.45, y=max(y_min, med_y * 0.35), text="집행 축소 검토", showarrow=False, font=dict(size=11, color="#991B1B"))

            apply_meta_theme(fig_bubble, "상품 포지셔닝 매트릭스 (X:노출, Y:CTR_total, 버블:총클릭수)", 460)
            xvals, xtext = axis_ticks_from_series(plot_b["노출수"])
            fig_bubble.update_layout(showlegend=False)
            fig_bubble.update_xaxes(title="노출수", tickmode="array", tickvals=xvals, ticktext=xtext, automargin=True)
            fig_bubble.update_yaxes(title="CTR_total", tickformat=".2%", range=[y_min, y_max * 1.15], automargin=True)
            st.plotly_chart(fig_bubble, use_container_width=True)

        st.write("")
        sec("노출 집중도 분석")
        conc_src = agg[agg["노출수"] > 0].copy()
        conc = compute_concentration_metrics(conc_src)
        c_kpi = st.columns(4)
        with c_kpi[0]:
            kpi_card("Top3 누적 비중", fmt_pct(conc["top3"]))
        with c_kpi[1]:
            kpi_card("Top5 누적 비중", fmt_pct(conc["top5"]))
        with c_kpi[2]:
            kpi_card("Top10 누적 비중", fmt_pct(conc["top10"]))
        with c_kpi[3]:
            kpi_card("HHI (0~10000)", f"{conc['hhi_10000']:.0f}")

        top_imp = conc_src.sort_values("노출수", ascending=False).head(12).copy()
        top_imp["점유율"] = top_imp["노출수"] / top_imp["노출수"].sum() if top_imp["노출수"].sum() > 0 else 0
        top_imp["누적점유율"] = top_imp["점유율"].cumsum()
        top_imp["상품축"] = top_imp["광고상품명_정리"].map(lambda x: ellipsis(x, 14))

        fig_conc = go.Figure()
        fig_conc.add_bar(
            x=top_imp["상품축"], y=top_imp["점유율"], name="노출 점유율",
            marker_color="#2563EB",
            customdata=np.array([[n, fmt_kor_unit(i), fmt_pct(s)] for n, i, s in zip(top_imp["광고상품명_정리"], top_imp["노출수"], top_imp["점유율"])], dtype=object),
            hovertemplate="상품명: %{customdata[0]}<br>노출수: %{customdata[1]}<br>점유율: %{customdata[2]}<extra></extra>",
        )
        fig_conc.add_scatter(
            x=top_imp["상품축"], y=top_imp["누적점유율"], name="누적 점유율", mode="lines+markers+text",
            line=dict(color="#D97706", width=2), marker=dict(size=6),
            text=[fmt_pct(v) for v in top_imp["누적점유율"]], textposition="top center",
            yaxis="y2", hovertemplate="누적 점유율: %{text}<extra></extra>",
        )
        apply_meta_theme(fig_conc, "상위 상품 노출 점유율 및 누적 비중", 380)
        fig_conc.update_layout(
            barmode="group",
            xaxis=dict(tickangle=-24, automargin=True),
            yaxis=dict(title="점유율", tickformat=".0%"),
            yaxis2=dict(title="누적 점유율", overlaying="y", side="right", tickformat=".0%", showgrid=False),
        )
        st.plotly_chart(fig_conc, use_container_width=True)

        if conc["top3"] >= 0.6 or conc["hhi_10000"] >= 1800:
            insight("개선 우선: 노출 쏠림이 높아 특정 상품 의존도가 큽니다. 대체 상품 테스트/확대가 필요합니다.", warn=True)
        else:
            insight("확대 후보: 노출 분산이 비교적 안정적입니다. 고CTR 저노출 상품의 노출 확대 테스트를 권장합니다.")

        med_imp, med_ctr = agg["노출수"].median(), agg["CTR_total"].median()
        qh = agg[(agg["노출수"] >= med_imp) & (agg["CTR_total"] >= med_ctr) & (agg["노출수"] >= min_imp)]
        ql = agg[(agg["노출수"] >= med_imp) & (agg["CTR_total"] < med_ctr) & (agg["노출수"] >= min_imp)]
        lh = agg[(agg["노출수"] < med_imp) & (agg["CTR_total"] >= med_ctr) & (agg["노출수"] >= min_imp)]
        ll = agg[(agg["노출수"] < med_imp) & (agg["CTR_total"] < med_ctr) & (agg["노출수"] >= min_imp)]
        sec("운영 인사이트")
        insight(f"확대 후보(고노출·고CTR): {', '.join(qh['광고상품명_정리'].head(3).tolist()) or '없음'}")
        insight(f"개선 우선(고노출·저CTR): {', '.join(ql['광고상품명_정리'].head(3).tolist()) or '없음'}", warn=True)
        insight(f"재테스트 후보(저노출·고CTR): {', '.join(lh['광고상품명_정리'].head(3).tolist()) or '없음'}")
        insight(f"집행 축소 검토(저노출·저CTR): {', '.join(ll['광고상품명_정리'].head(3).tolist()) or '없음'}", warn=True)

    with tab2:
        allp = sorted(df["광고상품명_정리"].unique().tolist())
        sel = st.selectbox("상품 선택", allp)
        d2 = aggregate(df[df["광고상품명_정리"] == sel], ["프로모션명", "연월"])
        pp = aggregate(df[df["광고상품명_정리"] == sel], ["프로모션명"]).sort_values("노출수", ascending=False)
        pp["프로모션축"] = pp["프로모션명"].map(lambda x: ellipsis(x, 16))

        sec("프로모션별 노출 · 클릭 · CTR")
        st.plotly_chart(build_volume_click_ctr_chart(pp.head(20), "프로모션축", f"{sel} 프로모션 성과"), use_container_width=True)

        sec("상품 — 프로모션별 월별 노출·CTR 추이 (단순화)")
        top_promos = pp.head(6)["프로모션명"].tolist()
        md = d2[d2["프로모션명"].isin(top_promos)].copy().sort_values("연월")
        md["연월_str"] = md["연월"].dt.strftime("%Y-%m")
        figm = go.Figure()
        for i, prm in enumerate(top_promos):
            sub = md[md["프로모션명"] == prm]
            figm.add_bar(
                x=sub["연월_str"],
                y=sub["노출수"],
                name=ellipsis(prm, 16),
                marker_color=COLORS[i % len(COLORS)],
                customdata=np.array([
                    [fmt_kor_unit(v1), fmt_pct(v2)] for v1, v2 in zip(sub["노출수"], sub["CTR_total"])
                ], dtype=object),
                hovertemplate="월: %{x}<br>노출수: %{customdata[0]}<br>CTR_total: %{customdata[1]}<extra></extra>",
            )
        apply_meta_theme(figm, "월별 노출 비교(CTR은 hover 보조)", 360)
        yv, yt = axis_ticks_from_series(md["노출수"]) if not md.empty else ([0], ["0"])
        month_labels = md["연월_str"].dropna().drop_duplicates().tolist()
        figm.update_layout(
            barmode="group",
            yaxis=dict(tickmode="array", tickvals=yv, ticktext=yt, title="노출수", automargin=True),
        )
        apply_month_axis(figm, month_labels, angle=-20)
        st.plotly_chart(figm, use_container_width=True)

        st.caption("※ 아래 차트는 선택 상품이 아닌 전체 데이터 기준(상위 프로모션) 월간 비교입니다.")
        sec("월별 프로모션 전체 성과 (분리 보기)")
        top_n_all = st.slider("전체 프로모션 상위 N(노출수 기준)", 3, 8, 5, key="t2_allpromo_n")
        all_pm = aggregate(df, ["연월", "프로모션명"]).sort_values("연월")
        top_promos_all = (
            all_pm.groupby("프로모션명", as_index=False)["노출수"].sum().sort_values("노출수", ascending=False).head(top_n_all)["프로모션명"].tolist()
        )
        all_pm = all_pm[all_pm["프로모션명"].isin(top_promos_all)].copy()
        all_pm["연월_str"] = all_pm["연월"].dt.strftime("%b %Y")
        month_all = all_pm["연월_str"].dropna().drop_duplicates().tolist()

        c_all_1, c_all_2 = st.columns(2)
        with c_all_1:
            sec("1) 월별 프로모션 전체 노출수")
            fig_imp = go.Figure()
            for i, prm in enumerate(top_promos_all):
                sub = all_pm[all_pm["프로모션명"] == prm]
                fig_imp.add_bar(
                    x=sub["연월_str"], y=sub["노출수"], name=ellipsis(prm, 14),
                    offsetgroup=f"imp_{i}", marker_color=COLORS[i % len(COLORS)],
                    customdata=[fmt_kor_unit(v) for v in sub["노출수"]],
                    hovertemplate="월: %{x}<br>노출수: %{customdata}<extra></extra>",
                )
            apply_meta_theme(fig_imp, "전체 데이터 기준 월별 프로모션 노출수", 400)
            yv_imp, yt_imp = make_kor_ticks(float(all_pm["노출수"].max()) if not all_pm.empty else 0)
            fig_imp.update_layout(barmode="group", yaxis=dict(tickmode="array", tickvals=yv_imp, ticktext=yt_imp, range=[0, yv_imp[-1]]))
            apply_month_axis(fig_imp, month_all, angle=-18)
            st.plotly_chart(fig_imp, use_container_width=True)

        with c_all_2:
            sec("2) 월별 프로모션 전체 총클릭수")
            fig_clk = go.Figure()
            for i, prm in enumerate(top_promos_all):
                sub = all_pm[all_pm["프로모션명"] == prm]
                fig_clk.add_bar(
                    x=sub["연월_str"], y=sub["총클릭수"], name=ellipsis(prm, 14),
                    offsetgroup=f"clk_{i}", marker_color=COLORS[i % len(COLORS)],
                    customdata=[fmt_kor_unit(v) for v in sub["총클릭수"]],
                    hovertemplate="월: %{x}<br>총클릭수: %{customdata}<extra></extra>",
                )
            apply_meta_theme(fig_clk, "전체 데이터 기준 월별 프로모션 총클릭수", 400)
            yv_clk, yt_clk = make_kor_ticks(float(all_pm["총클릭수"].max()) if not all_pm.empty else 0)
            fig_clk.update_layout(barmode="group", yaxis=dict(tickmode="array", tickvals=yv_clk, ticktext=yt_clk, range=[0, yv_clk[-1]]))
            apply_month_axis(fig_clk, month_all, angle=-18)
            st.plotly_chart(fig_clk, use_container_width=True)

        sec("3) 월별 프로모션 전체 CTR_total")
        fig_ctr = go.Figure()
        for i, prm in enumerate(top_promos_all):
            sub = all_pm[all_pm["프로모션명"] == prm]
            fig_ctr.add_scatter(
                x=sub["연월_str"], y=sub["CTR_total"], name=ellipsis(prm, 14), mode="lines+markers",
                line=dict(width=2, color=COLORS[i % len(COLORS)]), marker=dict(size=6),
                text=[fmt_pct(v) for v in sub["CTR_total"]],
                hovertemplate="월: %{x}<br>CTR_total: %{text}<extra></extra>",
            )
        apply_meta_theme(fig_ctr, "전체 데이터 기준 월별 프로모션 CTR_total", 380)
        yv_ctr, yt_ctr = make_pct_ticks(float(all_pm["CTR_total"].max()) if not all_pm.empty else 0.01)
        fig_ctr.update_layout(yaxis=dict(tickmode="array", tickvals=yv_ctr, ticktext=yt_ctr, range=[0, yv_ctr[-1]]))
        apply_month_axis(fig_ctr, month_all, angle=-18)
        st.plotly_chart(fig_ctr, use_container_width=True)

    with tab3:
        unit = st.radio("시간 단위", ["월별", "일별"], horizontal=True)
        t = "연월" if unit == "월별" else "일자"
        td = aggregate(df, [t]).sort_values(t)
        td["x"] = td[t].dt.strftime("%b %Y" if unit == "월별" else "%Y-%m-%d")
        sec("전체 추이 — 노출수(좌) / 총클릭수(우) + CTR 라벨")
        fig_td = build_volume_click_ctr_chart(td, "x", "기간별 핵심 추이")
        if unit == "월별":
            apply_month_axis(fig_td, td["x"].dropna().drop_duplicates().tolist(), angle=-20)
        st.plotly_chart(fig_td, use_container_width=True)

        imp_pos = df[df["노출수"] > 0]
        sec("기간별 노출 발생 광고상품 수")
        prod_cnt = imp_pos.groupby(t)["광고상품명_정리"].nunique().reset_index(name="광고상품수")
        prod_cnt["x"] = prod_cnt[t].dt.strftime("%Y-%m" if unit == "월별" else "%Y-%m-%d")
        fig_p = go.Figure(go.Bar(x=prod_cnt["x"], y=prod_cnt["광고상품수"], marker_color="#1E40AF", text=prod_cnt["광고상품수"]))
        apply_meta_theme(fig_p, "기간별 노출 발생 광고상품 수", 280)
        fig_p.update_traces(textposition="outside", hovertemplate="기간: %{x}<br>광고상품 수: %{y:,}<extra></extra>")
        yv_p, yt_p = make_kor_ticks(float(prod_cnt["광고상품수"].max()) if not prod_cnt.empty else 0)
        fig_p.update_yaxes(tickmode="array", tickvals=yv_p, ticktext=yt_p, range=[0, yv_p[-1]])
        st.plotly_chart(fig_p, use_container_width=True)

        sec("기간별 노출 발생 캠페인 수")
        # 집계 기준: period별 캠페인명 nunique, 단 노출수 > 0 행만 포함
        # 과거 값이 작게 나왔던 이유: 전체 데이터에서 중복 제거를 먼저 하거나 노출 없는 행 포함 시 왜곡 가능
        camp_cnt = imp_pos.groupby(t)["캠페인명"].nunique().reset_index(name="캠페인수")
        camp_cnt["x"] = camp_cnt[t].dt.strftime("%Y-%m" if unit == "월별" else "%Y-%m-%d")
        fig_c = go.Figure(go.Bar(x=camp_cnt["x"], y=camp_cnt["캠페인수"], marker_color="#7C3AED", text=camp_cnt["캠페인수"]))
        apply_meta_theme(fig_c, "기간별 노출 발생 캠페인 수", 280)
        fig_c.update_traces(textposition="outside")
        yv_c, yt_c = make_kor_ticks(float(camp_cnt["캠페인수"].max()) if not camp_cnt.empty else 0)
        fig_c.update_yaxes(tickmode="array", tickvals=yv_c, ticktext=yt_c, range=[0, yv_c[-1]])
        st.plotly_chart(fig_c, use_container_width=True)

    with tab4:
        sec("프로모션 A/B 비교")
        p_list = sorted(df["프로모션명"].dropna().astype(str).unique().tolist())
        cpa, cpb = st.columns(2)
        with cpa:
            promo_a = st.selectbox("프로모션 A", p_list, index=0 if p_list else None, key="cmp_promo_a")
        with cpb:
            promo_b = st.selectbox("프로모션 B", p_list, index=1 if len(p_list) > 1 else 0, key="cmp_promo_b")

        if promo_a and promo_b:
            dfa = df[df["프로모션명"] == promo_a]
            dfb = df[df["프로모션명"] == promo_b]
            sa, sb = summarize_period(dfa), summarize_period(dfb)

            sec("프로모션 KPI 비교")
            mcols = st.columns(4)
            klist = ["노출수", "클릭수", "총클릭수", "CTR_total", "동영상조회수", "VTR", "광고상품수", "캠페인수"]
            for i, k in enumerate(klist):
                with mcols[i % 4]:
                    if k in {"CTR_total", "VTR"}:
                        delta_txt = delta_sentence_pp(sa[k], sb[k])
                        kpi_card(k, f"A {fmt_pct(sa[k])} | B {fmt_pct(sb[k])}", delta_txt)
                    else:
                        delta_txt = delta_sentence_count(sa[k], sb[k])
                        kpi_card(k, f"A {fmt_kpi_compact(sa[k])} | B {fmt_kpi_compact(sb[k])}", delta_txt)

            sec("상품별 변화량 (프로모션 A vs B)")
            cmp = promotion_product_compare(dfa, dfb, min_imp)
            topn = st.slider("표시 상품 수", 10, 100, 30, key="cmp_topn")
            cmp = cmp.head(topn)
            cmp_d = cmp.copy()
            for c in ["A_노출수", "B_노출수", "A_총클릭수", "B_총클릭수", "노출 증감", "총클릭 증감"]:
                cmp_d[c] = cmp_d[c].map(fmt_table_number)
            for c in ["A_CTR", "B_CTR", "A_VTR", "B_VTR"]:
                cmp_d[c] = cmp_d[c].map(fmt_pct)
            cmp_d["CTR 변화(%p)"] = cmp["CTR 변화(%p)"].map(lambda v: f"{v:+.2f}%p")
            cmp_d["VTR 변화(%p)"] = cmp["VTR 변화(%p)"].map(lambda v: f"{v:+.2f}%p")
            st.dataframe(
                cmp_d[["광고상품명_정리", "A_노출수", "B_노출수", "A_총클릭수", "B_총클릭수", "A_CTR", "B_CTR", "CTR 변화(%p)", "A_VTR", "B_VTR", "VTR 변화(%p)", "해석상태"]],
                use_container_width=True,
                hide_index=True,
            )

            inc = cmp.sort_values("노출 증감", ascending=False).head(3)["광고상품명_정리"].tolist()
            dec = cmp.sort_values("노출 증감", ascending=True).head(3)["광고상품명_정리"].tolist()
            ctr_up = cmp.sort_values("CTR 변화(%p)", ascending=False).head(3)["광고상품명_정리"].tolist()
            insight(f"확대 후보: {', '.join(inc) if inc else '없음'}")
            insight(f"집행 축소 검토: {', '.join(dec) if dec else '없음'}", warn=True)
            insight(f"재테스트 후보(CTR 개선): {', '.join(ctr_up) if ctr_up else '없음'}")

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            day = aggregate(df, ["요일"]).copy()
            day["요일"] = pd.Categorical(day["요일"], categories=DAY_ORDER, ordered=True)
            day = day.sort_values("요일")
            sec("요일별 노출 · 클릭 · CTR")
            st.plotly_chart(build_volume_click_ctr_chart(day, "요일", "요일별 성과"), use_container_width=True)
        with c2:
            cat = aggregate(df, ["구분"]).sort_values("노출수", ascending=False).copy()
            cat["구분축"] = cat["구분"].map(lambda x: ellipsis(x, 12))
            sec("구분별 노출 · 클릭 · CTR")
            st.plotly_chart(build_volume_click_ctr_chart(cat, "구분축", "구분별 성과"), use_container_width=True)

        sec("상품 × 요일 히트맵 + TOP 조합")
        h = aggregate(df, ["광고상품명_정리", "요일"])
        pv = h.pivot(index="광고상품명_정리", columns="요일", values="CTR_total").reindex(columns=DAY_ORDER)
        fig_h = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=[ellipsis(v, 14) for v in pv.index], colorscale="Blues", hovertemplate="요일: %{x}<br>상품: %{y}<br>CTR_total: %{z:.2%}<extra></extra>"))
        apply_meta_theme(fig_h, "CTR 히트맵", max(260, len(pv) * 26 + 100))
        st.plotly_chart(fig_h, use_container_width=True)
        top_combo = h.sort_values(["CTR_total", "노출수"], ascending=[False, False]).head(10)[["광고상품명_정리", "요일", "노출수", "총클릭수", "CTR_total"]]
        top_combo["노출수"] = top_combo["노출수"].map(fmt_table_number)
        top_combo["총클릭수"] = top_combo["총클릭수"].map(fmt_table_number)
        top_combo["CTR_total"] = top_combo["CTR_total"].map(fmt_pct)
        st.dataframe(top_combo, hide_index=True, use_container_width=True)

    with tab6:
        ind = compute_efficiency(aggregate(df, ["업종"]))
        ind["업종축"] = ind["업종"].map(lambda x: ellipsis(x, 10))
        sec("업종별 노출 · 클릭 · CTR")
        st.plotly_chart(build_volume_click_ctr_chart(ind.sort_values("노출수", ascending=False), "업종축", "업종별 성과"), use_container_width=True)

        sec("업종별 광고주 수 + 광고주 목록")
        adv = df.groupby("업종")["광고주"].agg(lambda s: sorted(set(map(str, s)))).reset_index(name="광고주목록")
        adv["광고주 수"] = adv["광고주목록"].map(len)
        adv["광고주 목록"] = adv["광고주목록"].map(lambda x: ", ".join(x[:12]) + (" …" if len(x) > 12 else ""))
        out = adv[["업종", "광고주 수", "광고주 목록"]].sort_values("광고주 수", ascending=False)
        st.dataframe(out, use_container_width=True, hide_index=True)

    with tab7:
        sec("Appendix ① 기본 트래픽 지표")
        st.markdown(
            """
- **노출수**: 광고가 사용자에게 표시된 횟수(볼륨 지표).
- **클릭수**: 기본 클릭 집계.
- **총클릭수**: 클릭수 + 컴패니언/부가 클릭을 포함한 실반응 클릭.
- **동영상조회수**: 영상 재생/조회 기반 반응량.
"""
        )

        sec("Appendix ② 비율 지표")
        st.markdown(
            """
- **CTR** = 클릭수 ÷ 노출수.
- **CTR(전체)** = 총클릭수 ÷ 노출수. (본 대시보드 핵심 효율 지표)
- **VTR** = 동영상조회수 ÷ 노출수.
- **동반클릭/컴패니언 개념**: 기본 클릭 외 보조 반응 클릭 포함 시 총반응을 더 잘 반영.
"""
        )

        sec("Appendix ③ 파생 지표 및 해석")
        st.markdown(
            """
- **노출비중**: 특정 행(상품/프로모션/업종)의 노출수 / 전체 노출수.
- **효율점수(Z-score)**: 동일 집계 그룹 내 CTR_total의 상대 위치.
- **효율등급(S/A/B/C)**: Z-score 구간화.
- **표본부족(min_imp)**: 최소 노출 기준 미달 시 효율 해석 제한.
"""
        )

        sec("Appendix ④ 집계/전처리 규칙")
        st.markdown(
            """
- 일자 파싱 후 연/월/연월 컬럼 생성.
- 광고상품명 정규화:
  - 대소문자/공백 차이 최소화
  - 문자열 끝 괄호 suffix 제거 후 통합
  - `_` 포함 suffix 상품은 별도 상품 유지
- 업종 분류: 광고주명 키워드 매칭(공백/대소문자 정규화 후 포함 매칭).
"""
        )

        sec("Appendix ⑤ 비교 지표 해석")
        st.markdown(
            """
- CTR/VTR 변화는 **상대증감률(%)보다 %p(퍼센트포인트)** 우선 해석.
- 프로모션 비교에서 문장형 차이 표기 사용:
  - 예) `B가 A보다 0.18%p 높음`, `B가 A보다 25.4만 큼`
"""
        )

        sec("Appendix ⑥ 노출 집중도(Concentration)")
        st.markdown(
            """
- **Top3/Top5/Top10 누적 노출 비중**: 상위 상품 의존도 점검.
- **HHI** = 각 상품 점유율 제곱합.
  - 본 대시보드는 **0~10000 스케일** 병행 제공(`HHI*10000`).
  - 값이 높을수록 소수 상품 집중도가 큼.
"""
        )

        sec("Appendix ⑦ 비용 지표 사용 주의")
        st.markdown(
            """
- 광고비/CPC/CPV/eCPM은 내부 하우스 광고 데이터 특성상 결측/품질 이슈 가능성이 있어,
  본문 핵심 판단은 노출/총클릭/CTR_total/VTR 중심으로 권장합니다.
"""
        )


if __name__ == "__main__":
    main()
