"""
AD Performance Benchmark Dashboard — Professional Edition
실행 방법:
  pip install streamlit pandas numpy plotly
  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# 0. 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AD Performance Benchmark",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 0-1. 글로벌 CSS (프리미엄 다크 리포트 스타일)
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── 구글 폰트 ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── 전체 배경 ─────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── 메인 헤더 배너 ────────────────────── */
    .report-header {
        background: linear-gradient(135deg, #0d1117 0%, #1a2744 40%, #0d2137 100%);
        border: 1px solid #1f6feb;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .report-header::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(31,111,235,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .report-header h1 {
        font-size: 26px; font-weight: 700;
        color: #f0f6fc !important;
        letter-spacing: -0.5px; margin: 0 0 4px 0;
    }
    .report-header p { color: #8b949e !important; font-size: 13px; margin: 0; }

    /* ── 섹션 타이틀 ────────────────────────── */
    .section-title {
        font-size: 13px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.5px;
        color: #58a6ff;
        border-left: 3px solid #1f6feb;
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }

    /* ── KPI 카드 ───────────────────────────── */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #1f6feb; }
    .kpi-label {
        font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px;
        color: #8b949e; margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px; font-weight: 700;
        font-family: 'DM Mono', monospace;
        color: #f0f6fc; line-height: 1;
    }
    .kpi-delta-pos { font-size: 12px; color: #3fb950; margin-top: 4px; }
    .kpi-delta-neg { font-size: 12px; color: #f85149; margin-top: 4px; }
    .kpi-delta-neu { font-size: 12px; color: #8b949e; margin-top: 4px; }
    .kpi-accent-blue  { border-top: 3px solid #1f6feb !important; }
    .kpi-accent-green { border-top: 3px solid #3fb950 !important; }
    .kpi-accent-gold  { border-top: 3px solid #d29922 !important; }
    .kpi-accent-red   { border-top: 3px solid #f85149 !important; }
    .kpi-accent-purple{ border-top: 3px solid #8957e5 !important; }
    .kpi-accent-cyan  { border-top: 3px solid #39c5cf !important; }
    .kpi-accent-pink  { border-top: 3px solid #e85d9c !important; }

    /* ── 배지 ───────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 11px; font-weight: 600;
    }
    .badge-blue   { background: #1f3a6e; color: #58a6ff; }
    .badge-orange { background: #3d2b00; color: #d29922; }
    .badge-gray   { background: #21262d; color: #8b949e; }

    /* ── 테이블 스타일 ──────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* ── 탭 ─────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: #161b22;
        border-bottom: 1px solid #30363d;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        font-size: 13px; font-weight: 500;
        padding: 12px 20px;
        border-radius: 0;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #1f6feb !important;
    }

    /* ── 입력 위젯 ──────────────────────────── */
    .stSelectbox > div, .stMultiSelect > div {
        background: #161b22 !important;
        border-color: #30363d !important;
        color: #c9d1d9 !important;
    }
    .stSlider > div { color: #c9d1d9 !important; }

    /* ── 구분선 ─────────────────────────────── */
    hr { border-color: #30363d !important; }

    /* ── 스크롤바 ───────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* ── Plotly 배경 통일 ───────────────────── */
    .js-plotly-plot { border-radius: 8px; overflow: hidden; }

    /* ── 사이드바 타이틀 ────────────────────── */
    .sidebar-title {
        font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        color: #58a6ff !important;
        margin-bottom: 12px;
    }

    /* ── 인사이트 카드 ──────────────────────── */
    .insight-card {
        background: #0d2137;
        border: 1px solid #1f6feb;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #c9d1d9;
    }
    .insight-icon { font-size: 16px; margin-right: 6px; }

    /* ── 효율 등급 배지 ─────────────────────── */
    .grade-S { background:#1a3a1a; color:#3fb950; padding:2px 8px; border-radius:4px; font-weight:700; font-size:12px; }
    .grade-A { background:#0d2137; color:#58a6ff; padding:2px 8px; border-radius:4px; font-weight:700; font-size:12px; }
    .grade-B { background:#2d2600; color:#d29922; padding:2px 8px; border-radius:4px; font-weight:700; font-size:12px; }
    .grade-C { background:#2d1b1b; color:#f85149; padding:2px 8px; border-radius:4px; font-weight:700; font-size:12px; }

    /* ── 숨기기: Streamlit 기본 metric 컨테이너 ── */
    [data-testid="metric-container"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 1. 상수 및 헬퍼
# ──────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "캠페인명", "일자", "노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수",
    "CTR", "CTR(전체)", "VTR", "광고비", "eCPM", "CPC", "CPV", "연도", "월", "광고상품명_정리", "구분",
    "프로모션별", "광고주",
]
NUMERIC_COLUMNS = [
    "노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수", "CTR", "CTR(전체)",
    "VTR", "광고비", "eCPM", "CPC", "CPV", "연도", "월",
]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}

# Plotly 공통 레이아웃
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(family="DM Sans", color="#c9d1d9", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(size=11)),
    colorway=["#1f6feb", "#3fb950", "#d29922", "#f85149", "#8957e5", "#39c5cf", "#e85d9c"],
)

COLOR_SCALE = [
    [0.0, "#0d1117"],
    [0.3, "#0d2137"],
    [0.6, "#1a3a6e"],
    [1.0, "#1f6feb"],
]

SECTION_NAMES = ["KPI 요약", "광고주 분석", "상품 분석", "캠페인 효율", "시계열", "요일 분석", "구분 비교", "프로모션"]


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def fmt_num(v, fmt=","):
    if pd.isna(v):
        return "N/A"
    if fmt == "pct":
        return f"{v:.2%}"
    if fmt == "won":
        return f"₩{v:,.0f}"
    if fmt == "dec2":
        return f"{v:.2f}"
    return f"{v:{fmt}}"


def efficiency_grade(score: float) -> str:
    if score >= 1.0:
        return "S"
    elif score >= 0.3:
        return "A"
    elif score >= -0.3:
        return "B"
    else:
        return "C"


# ──────────────────────────────────────────────
# 2. 데이터 로드 & 전처리
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["일자"] = pd.to_datetime(out["일자"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["요일"] = out["일자"].dt.weekday.map(DAY_MAP)
    out["연도"] = out["연도"].fillna(out["일자"].dt.year)
    out["월"] = out["월"].fillna(out["일자"].dt.month)
    out["연월"] = pd.to_datetime(
        dict(year=out["연도"].astype("Int64"), month=out["월"].astype("Int64"), day=1), errors="coerce"
    )
    for col in ["캠페인명", "광고상품명_정리", "구분"]:
        out[col] = out[col].astype("string").fillna("미분류")
    out["프로모션명"] = (
        out["프로모션별"].astype("string").fillna("미분류")
        if "프로모션별" in out.columns
        else out["캠페인명"].astype("string").fillna("미분류")
    )
    out["광고주"] = out["광고주"].astype("string").fillna("미분류") if "광고주" in out.columns else "미분류"
    out["총클릭수"] = out["총클릭수"].fillna(out["클릭수"]).fillna(0)
    out["클릭수"] = out["클릭수"].fillna(0)
    out["노출수"] = out["노출수"].fillna(0)
    out["동영상조회수"] = out["동영상조회수"].fillna(0)
    out["광고비"] = out["광고비"].fillna(0)
    return out


# ──────────────────────────────────────────────
# 3. 집계 함수
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False, as_index=False)[
            ["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "광고비", "동영상조회수"]
        ]
        .sum(min_count=1)
        .fillna(0)
    )
    agg["CTR"] = safe_div(agg["클릭수"], agg["노출수"])
    agg["CTR_total"] = safe_div(agg["총클릭수"], agg["노출수"])
    agg["CPC"] = safe_div(agg["광고비"], agg["클릭수"])
    agg["CPC_total"] = safe_div(agg["광고비"], agg["총클릭수"])
    agg["eCPM"] = safe_div(agg["광고비"], agg["노출수"]) * 1000
    agg["VTR"] = safe_div(agg["동영상조회수"], agg["노출수"])
    agg["CPV"] = safe_div(agg["광고비"], agg["동영상조회수"])
    # 광고비 비중
    total_spend = agg["광고비"].sum()
    agg["광고비비중"] = agg["광고비"] / total_spend if total_spend > 0 else 0
    return agg


def compute_efficiency_score(df: pd.DataFrame) -> pd.DataFrame:
    """CTR 높을수록, CPC 낮을수록 좋은 복합 효율 점수 (z-score 기반)"""
    out = df.copy()
    valid = out[(out["CTR_total"].notna()) & (out["CPC_total"].notna()) & (out["광고비"] > 0)].copy()
    if len(valid) < 2:
        out["efficiency_score"] = np.nan
        out["efficiency_grade"] = "N/A"
        return out

    ctr_std = valid["CTR_total"].std(ddof=0)
    cpc_std = valid["CPC_total"].std(ddof=0)
    ctr_z = (valid["CTR_total"] - valid["CTR_total"].mean()) / (ctr_std if ctr_std > 0 else 1)
    cpc_z = (valid["CPC_total"] - valid["CPC_total"].mean()) / (cpc_std if cpc_std > 0 else 1)
    valid["efficiency_score"] = (ctr_z * 0.6 - cpc_z * 0.4).round(3)
    valid["efficiency_grade"] = valid["efficiency_score"].apply(efficiency_grade)

    out = out.merge(valid[["efficiency_score", "efficiency_grade"]], left_index=True, right_index=True, how="left")
    out["efficiency_grade"] = out["efficiency_grade"].fillna("N/A")
    return out


# ──────────────────────────────────────────────
# 4. 필터 관련
# ──────────────────────────────────────────────
@dataclass
class Filters:
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    years: List[int]
    months: List[int]
    weekdays: List[str]
    categories: List[str]
    products: List[str]
    advertisers: List[str]
    campaigns: List[str]


def make_filters(date_range, years, months, weekdays, categories, products, advertisers, campaigns) -> Filters:
    d0 = pd.to_datetime(date_range[0]) if date_range and len(date_range) > 0 else None
    d1 = pd.to_datetime(date_range[1]) if date_range and len(date_range) > 1 else None
    return Filters(d0, d1, years, months, weekdays, categories, products, advertisers, campaigns)


@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()
    if f.date_min is not None:
        out = out[out["일자"] >= f.date_min]
    if f.date_max is not None:
        out = out[out["일자"] <= f.date_max]
    if f.years:
        out = out[out["연도"].isin(f.years)]
    if f.months:
        out = out[out["월"].isin(f.months)]
    if f.weekdays:
        out = out[out["요일"].isin(f.weekdays)]
    if f.categories:
        out = out[out["구분"].isin(f.categories)]
    if f.products:
        out = out[out["광고상품명_정리"].isin(f.products)]
    if f.advertisers:
        out = out[out["광고주"].isin(f.advertisers)]
    if f.campaigns:
        out = out[out["캠페인명"].isin(f.campaigns)]
    return out


# ──────────────────────────────────────────────
# 5. UI 컴포넌트
# ──────────────────────────────────────────────
def section_header(title: str, badge_text: str = "", badge_type: str = "gray"):
    badge_html = (
        f'<span class="badge badge-{badge_type}" style="margin-left:10px;">{badge_text}</span>'
        if badge_text
        else ""
    )
    st.markdown(f'<div class="section-title">{title}{badge_html}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", delta_good: Optional[bool] = None, accent: str = "blue"):
    delta_cls = "kpi-delta-neu"
    if delta_good is True:
        delta_cls = "kpi-delta-pos"
    elif delta_good is False:
        delta_cls = "kpi-delta-neg"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card kpi-accent-{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(icon: str, text: str):
    st.markdown(
        f'<div class="insight-card"><span class="insight-icon">{icon}</span>{text}</div>',
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=height)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")
    return fig


def style_dataframe(df: pd.DataFrame):
    """스타일링된 데이터프레임 렌더링"""
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ──────────────────────────────────────────────
# 6. 전월 비교 헬퍼
# ──────────────────────────────────────────────
def period_comparison(df: pd.DataFrame):
    """최근 2개월 데이터 비교 → delta dict 반환"""
    months_sorted = sorted(df["연월"].dropna().unique())
    if len(months_sorted) < 2:
        return {}
    cur_m, prev_m = months_sorted[-1], months_sorted[-2]
    cur = df[df["연월"] == cur_m]
    prev = df[df["연월"] == prev_m]

    def pct_change(a, b):
        return (a - b) / b if b and b != 0 else None

    def total_kpi(d):
        imp = d["노출수"].sum()
        clk = d["총클릭수"].sum()
        spd = d["광고비"].sum()
        vw = d["동영상조회수"].sum()
        return {
            "노출수": imp,
            "총클릭수": clk,
            "광고비": spd,
            "CTR_total": clk / imp if imp else np.nan,
            "CPC_total": spd / clk if clk else np.nan,
            "eCPM": spd / imp * 1000 if imp else np.nan,
            "VTR": vw / imp if imp else np.nan,
        }

    c, p = total_kpi(cur), total_kpi(prev)
    deltas = {}
    for k in c:
        ch = pct_change(c[k], p[k])
        deltas[k] = (c[k], ch)
    return deltas


def delta_str(ch: Optional[float], pct_format: bool = True) -> tuple[str, Optional[bool]]:
    if ch is None or pd.isna(ch):
        return ("전월 비교 없음", None)
    sign = "▲" if ch > 0 else "▼"
    val = f"{abs(ch):.1%}" if pct_format else f"{abs(ch):.2f}"
    return (f"{sign} {val} vs 전월", ch > 0)


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────
def main():
    # ── 사이드바 ────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📡 Data Source</div>', unsafe_allow_html=True)
        upload = st.file_uploader("CSV 업로드", type=["csv"])
        path = st.text_input("또는 파일 경로 입력")

    if upload is None and not path:
        st.markdown(
            """
            <div class="report-header">
                <h1>📡 AD Performance Benchmark Dashboard</h1>
                <p>광고 운영 데이터를 CSV로 업로드하면 광고주 · 상품 · 캠페인별 성과를 자동으로 분석합니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("👈 사이드바에서 CSV 데이터를 업로드해주세요.")
        st.stop()

    try:
        with st.spinner("데이터 로딩 중..."):
            raw = load_csv(upload.getvalue()) if upload else load_csv_path(path)
    except Exception as e:
        st.error(f"로드 실패: {e}")
        st.stop()

    df_raw = preprocess(raw)

    # ── 전역 필터 사이드바 ───────────────────────
    with st.sidebar:
        st.divider()
        st.markdown('<div class="sidebar-title">🔽 Global Filters</div>', unsafe_allow_html=True)

        min_d = df_raw["일자"].min()
        max_d = df_raw["일자"].max()
        g_date = st.date_input(
            "날짜 범위",
            value=(min_d.date(), max_d.date()) if pd.notna(min_d) and pd.notna(max_d) else (),
        )
        years = sorted(df_raw["연도"].dropna().astype(int).unique().tolist())
        months = sorted(df_raw["월"].dropna().astype(int).unique().tolist())
        weekdays = [d for d in DAY_ORDER if d in set(df_raw["요일"].dropna().unique())]
        categories = sorted(df_raw["구분"].dropna().unique().tolist())
        products = sorted(df_raw["광고상품명_정리"].dropna().unique().tolist())
        advertisers = sorted(df_raw["광고주"].dropna().unique().tolist())
        campaigns = sorted(df_raw["캠페인명"].dropna().unique().tolist())

        g_years = st.multiselect("연도", years, default=years)
        g_months = st.multiselect("월", months, default=months)
        g_weekdays = st.multiselect("요일", weekdays, default=weekdays)
        g_advertisers = st.multiselect("광고주", advertisers, default=advertisers)
        g_categories = st.multiselect("구분", categories, default=categories)
        g_products = st.multiselect("광고상품명", products, default=products)
        search = st.text_input("캠페인 검색", placeholder="키워드 입력...")
        camp_opts = [c for c in campaigns if search.lower() in c.lower()]
        g_campaigns = st.multiselect("캠페인명", camp_opts, default=camp_opts)

        st.divider()
        st.markdown('<div class="sidebar-title">📊 지표 선택</div>', unsafe_allow_html=True)
        primary_metric = st.selectbox(
            "주요 지표",
            ["CTR_total", "CPC_total", "eCPM", "VTR", "CPV", "광고비", "노출수", "총클릭수"],
        )

    gf = make_filters(g_date, g_years, g_months, g_weekdays, g_categories, g_products, g_advertisers, g_campaigns)
    df = apply_filters(df_raw, gf)

    # ── 리포트 헤더 ──────────────────────────────
    date_label = ""
    if pd.notna(df["일자"].min()) and pd.notna(df["일자"].max()):
        date_label = f"{df['일자'].min().date()} ~ {df['일자'].max().date()}"
    st.markdown(
        f"""
        <div class="report-header">
            <h1>📡 AD Performance Benchmark Report</h1>
            <p>
                분석 기간: <strong style="color:#58a6ff">{date_label}</strong> &nbsp;|&nbsp;
                원천 데이터: <strong style="color:#58a6ff">{len(df_raw):,}건</strong> &nbsp;|&nbsp;
                필터 적용: <strong style="color:#3fb950">{len(df):,}건</strong> &nbsp;|&nbsp;
                광고주: <strong style="color:#d29922">{df['광고주'].nunique()}개</strong> &nbsp;|&nbsp;
                상품: <strong style="color:#8957e5">{df['광고상품명_정리'].nunique()}개</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 전월 비교 ────────────────────────────────
    deltas = period_comparison(df)

    # ── KPI 카드 ─────────────────────────────────
    section_header("KPI 요약", "전 기간 합산", "blue")

    total_imp = df["노출수"].sum()
    total_clk = df["클릭수"].sum()
    total_tclk = df["총클릭수"].sum()
    total_spd = df["광고비"].sum()
    total_vw = df["동영상조회수"].sum()
    ctr_t = total_tclk / total_imp if total_imp else np.nan
    cpc_t = total_spd / total_tclk if total_tclk else np.nan
    ecpm = total_spd / total_imp * 1000 if total_imp else np.nan
    vtr = total_vw / total_imp if total_imp else np.nan
    cpv = total_spd / total_vw if total_vw else np.nan

    cols = st.columns(7)
    kpi_defs = [
        ("노출수", fmt_num(total_imp), "blue", "노출수"),
        ("총클릭수", fmt_num(total_tclk), "cyan", "총클릭수"),
        ("CTR(전체)", fmt_num(ctr_t, "pct"), "green", "CTR_total"),
        ("광고비", fmt_num(total_spd, "won"), "gold", "광고비"),
        ("CPC(전체)", fmt_num(cpc_t, "won"), "red", "CPC_total"),
        ("eCPM", fmt_num(ecpm, "won"), "purple", "eCPM"),
        ("VTR", fmt_num(vtr, "pct"), "pink", "VTR"),
    ]
    accents = ["blue", "cyan", "green", "gold", "red", "purple", "pink"]
    for i, (col, (lbl, val, _, dk)) in enumerate(zip(cols, kpi_defs)):
        with col:
            dv, dg = ("", None)
            if dk in deltas:
                dv, dg = delta_str(deltas[dk][1])
                # CPC 방향 반전 (낮을수록 좋음)
                if dk in ("CPC_total", "eCPM", "CPV") and dg is not None:
                    dg = not dg
            kpi_card(lbl, val, dv, dg, accents[i])

    st.write("")

    # ── 탭 레이아웃 ──────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 광고주 분석",
        "📦 상품 심층분석",
        "📈 시계열 트렌드",
        "📅 요일·구분 분석",
        "🏷️ 프로모션 분석",
    ])

    # ════════════════════════════════════════════
    # TAB 1: 광고주별 성과
    # ════════════════════════════════════════════
    with tab1:
        section_header("광고주별 성과 요약")
        adv_agg = aggregate(df, ["광고주"])
        adv_agg = compute_efficiency_score(adv_agg)
        adv_agg = adv_agg[adv_agg["노출수"] > 0].sort_values("광고비", ascending=False)

        # 광고주 성과 테이블
        disp = adv_agg[["광고주", "노출수", "총클릭수", "광고비", "CTR_total", "CPC_total", "eCPM", "VTR", "광고비비중", "efficiency_grade"]].copy()
        disp.columns = ["광고주", "노출수", "총클릭수", "광고비(₩)", "CTR(%)", "CPC(₩)", "eCPM(₩)", "VTR(%)", "광고비비중(%)", "효율등급"]
        disp["노출수"] = disp["노출수"].apply(lambda x: f"{x:,.0f}")
        disp["총클릭수"] = disp["총클릭수"].apply(lambda x: f"{x:,.0f}")
        disp["광고비(₩)"] = disp["광고비(₩)"].apply(lambda x: f"₩{x:,.0f}")
        disp["CTR(%)"] = disp["CTR(%)"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        disp["CPC(₩)"] = disp["CPC(₩)"].apply(lambda x: f"₩{x:,.0f}" if pd.notna(x) else "N/A")
        disp["eCPM(₩)"] = disp["eCPM(₩)"].apply(lambda x: f"₩{x:,.0f}" if pd.notna(x) else "N/A")
        disp["VTR(%)"] = disp["VTR(%)"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        disp["광고비비중(%)"] = disp["광고비비중(%)"].apply(lambda x: f"{x:.1%}")
        style_dataframe(disp)

        st.write("")
        c1, c2 = st.columns(2)

        # 광고주별 광고비 vs CTR 버블차트
        with c1:
            section_header("효율 포지셔닝 (CTR vs CPC)", "버블=광고비")
            bubble_df = adv_agg[(adv_agg["CTR_total"].notna()) & (adv_agg["CPC_total"].notna()) & (adv_agg["광고비"] > 0)]
            if not bubble_df.empty:
                fig = px.scatter(
                    bubble_df,
                    x="CPC_total", y="CTR_total",
                    size="광고비", color="광고주",
                    hover_name="광고주",
                    hover_data={"노출수": ":,.0f", "광고비": ":,.0f", "CTR_total": ":.3%", "CPC_total": ":,.0f"},
                    size_max=50,
                )
                # 평균 기준선
                avg_ctr = bubble_df["CTR_total"].mean()
                avg_cpc = bubble_df["CPC_total"].mean()
                fig.add_hline(y=avg_ctr, line_dash="dot", line_color="#8b949e", annotation_text="평균 CTR", annotation_position="right")
                fig.add_vline(x=avg_cpc, line_dash="dot", line_color="#8b949e", annotation_text="평균 CPC", annotation_position="top")
                apply_plotly_theme(fig, height=360)
                fig.update_xaxes(title="CPC(₩) — 낮을수록 효율적")
                fig.update_yaxes(title="CTR — 높을수록 효율적", tickformat=".2%")
                st.plotly_chart(fig, use_container_width=True)

        # 광고비 비중 파이
        with c2:
            section_header("광고비 점유율")
            fig2 = go.Figure(go.Pie(
                labels=adv_agg["광고주"],
                values=adv_agg["광고비"],
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(family="DM Sans", size=11),
                marker=dict(colors=["#1f6feb", "#3fb950", "#d29922", "#f85149", "#8957e5", "#39c5cf", "#e85d9c"]),
            ))
            apply_plotly_theme(fig2, height=360)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # 광고주 × 상품 히트맵
        st.write("")
        section_header("광고주 × 상품 효율 히트맵", f"지표: {primary_metric}")
        adv_prod = aggregate(df, ["광고주", "광고상품명_정리"])
        if not adv_prod.empty and primary_metric in adv_prod.columns:
            pivot = adv_prod.pivot_table(index="광고주", columns="광고상품명_정리", values=primary_metric, aggfunc="mean")
            if not pivot.empty:
                fig3 = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=list(pivot.columns),
                    y=list(pivot.index),
                    colorscale=COLOR_SCALE,
                    hoverongaps=False,
                    hovertemplate="%{y} × %{x}<br>값: %{z:.4f}<extra></extra>",
                ))
                apply_plotly_theme(fig3, height=max(200, len(pivot) * 40 + 80))
                fig3.update_xaxes(tickangle=-30)
                st.plotly_chart(fig3, use_container_width=True)

        # 광고주별 효율 등급 분포
        st.write("")
        section_header("광고주별 효율 등급")
        grade_counts = adv_agg.groupby("efficiency_grade")["광고주"].count().reset_index()
        grade_counts.columns = ["등급", "광고주 수"]
        grade_color_map = {"S": "#3fb950", "A": "#58a6ff", "B": "#d29922", "C": "#f85149", "N/A": "#8b949e"}
        fig_grade = px.bar(
            grade_counts, x="등급", y="광고주 수",
            color="등급",
            color_discrete_map=grade_color_map,
            text="광고주 수",
        )
        fig_grade.update_traces(textposition="outside")
        apply_plotly_theme(fig_grade, height=260)
        st.plotly_chart(fig_grade, use_container_width=True)

        # 인사이트
        if not adv_agg.empty and "efficiency_score" in adv_agg.columns:
            top_adv = adv_agg.dropna(subset=["efficiency_score"]).nlargest(1, "efficiency_score")
            bot_adv = adv_agg.dropna(subset=["efficiency_score"]).nsmallest(1, "efficiency_score")
            if not top_adv.empty:
                insight_box("🏆", f"최고 효율 광고주: <strong>{top_adv.iloc[0]['광고주']}</strong> — 효율등급 <strong>{top_adv.iloc[0]['efficiency_grade']}</strong>")
            if not bot_adv.empty:
                insight_box("⚠️", f"개선 필요 광고주: <strong>{bot_adv.iloc[0]['광고주']}</strong> — 효율등급 <strong>{bot_adv.iloc[0]['efficiency_grade']}</strong>, CPC ₩{bot_adv.iloc[0]['CPC_total']:,.0f}" if pd.notna(bot_adv.iloc[0]["CPC_total"]) else f"개선 필요 광고주: {bot_adv.iloc[0]['광고주']}")

    # ════════════════════════════════════════════
    # TAB 2: 광고상품 심층분석
    # ════════════════════════════════════════════
    with tab2:
        section_header("광고상품 성과 벤치마크")

        prod_agg = aggregate(df, ["광고상품명_정리"])
        prod_agg = compute_efficiency_score(prod_agg)

        # 최소 노출 필터
        col_f1, col_f2 = st.columns([2, 5])
        with col_f1:
            min_imp = st.number_input("최소 노출 기준", min_value=0, value=10000, step=5000)
        with col_f2:
            sort_by = st.selectbox("정렬 기준", ["efficiency_score", "CTR_total", "CPC_total", "eCPM", "광고비", "노출수"], index=0)

        prod_agg_f = prod_agg[prod_agg["노출수"] >= min_imp].copy()
        asc_sort = sort_by in ("CPC_total", "eCPM", "CPV")
        prod_agg_f = prod_agg_f.sort_values(sort_by, ascending=asc_sort)

        # 벤치마크 기준선 계산
        bench = {
            "CTR_total": prod_agg_f["CTR_total"].median(),
            "CPC_total": prod_agg_f["CPC_total"].median(),
            "eCPM": prod_agg_f["eCPM"].median(),
        }

        # 테이블
        show_cols = ["광고상품명_정리", "노출수", "총클릭수", "광고비", "CTR_total", "CPC_total", "eCPM", "VTR", "CPV", "광고비비중", "efficiency_score", "efficiency_grade"]
        disp2 = prod_agg_f[[c for c in show_cols if c in prod_agg_f.columns]].copy()
        rename = {
            "광고상품명_정리": "상품명",
            "CTR_total": "CTR(%)",
            "CPC_total": "CPC(₩)",
            "eCPM": "eCPM(₩)",
            "VTR": "VTR(%)",
            "CPV": "CPV(₩)",
            "광고비비중": "광고비비중(%)",
            "efficiency_score": "효율점수",
            "efficiency_grade": "등급",
        }
        disp2 = disp2.rename(columns=rename)
        for col_, fmt_ in [("노출수", ","), ("총클릭수", ","), ("광고비", "won"), ("CTR(%)", "pct"), ("CPC(₩)", "won"), ("eCPM(₩)", "won"), ("VTR(%)", "pct"), ("CPV(₩)", "won"), ("광고비비중(%)", "pct")]:
            if col_ in disp2.columns:
                disp2[col_] = disp2[col_].apply(lambda x: fmt_num(x, fmt_) if pd.notna(x) else "N/A")
        if "효율점수" in disp2.columns:
            disp2["효율점수"] = disp2["효율점수"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        style_dataframe(disp2)

        st.write("")
        st.download_button(
            "📥 상품별 집계 다운로드 (CSV)",
            data=prod_agg_f.to_csv(index=False).encode("utf-8-sig"),
            file_name="product_benchmark.csv",
            mime="text/csv",
        )

        st.write("")
        c1, c2 = st.columns(2)

        # CTR TOP/WORST 수평 바 차트
        with c1:
            section_header("CTR TOP 10 상품")
            top10 = prod_agg_f.nlargest(10, "CTR_total")[["광고상품명_정리", "CTR_total"]].iloc[::-1]
            colors = ["#3fb950" if v >= bench["CTR_total"] else "#f85149" for v in top10["CTR_total"]]
            fig = go.Figure(go.Bar(
                x=top10["CTR_total"], y=top10["광고상품명_정리"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2%}" for v in top10["CTR_total"]],
                textposition="outside",
            ))
            fig.add_vline(x=bench["CTR_total"], line_dash="dot", line_color="#d29922", annotation_text="중앙값")
            apply_plotly_theme(fig, height=340)
            fig.update_xaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

        # CPC 최저 10 상품
        with c2:
            section_header("CPC 최저 10 상품 (효율적)")
            valid_cpc = prod_agg_f[(prod_agg_f["CPC_total"].notna()) & (prod_agg_f["광고비"] > 0)]
            low10 = valid_cpc.nsmallest(10, "CPC_total")[["광고상품명_정리", "CPC_total"]].iloc[::-1]
            colors2 = ["#3fb950" if v <= bench["CPC_total"] else "#f85149" for v in low10["CPC_total"]]
            fig2 = go.Figure(go.Bar(
                x=low10["CPC_total"], y=low10["광고상품명_정리"],
                orientation="h",
                marker_color=colors2,
                text=[f"₩{v:,.0f}" for v in low10["CPC_total"]],
                textposition="outside",
            ))
            fig2.add_vline(x=bench["CPC_total"], line_dash="dot", line_color="#d29922", annotation_text="중앙값")
            apply_plotly_theme(fig2, height=340)
            st.plotly_chart(fig2, use_container_width=True)

        # 효율 스코어 레이더 차트 (상위 5개)
        st.write("")
        section_header("상위 5개 상품 지표 비교")
        top5 = prod_agg_f.dropna(subset=["efficiency_score"]).nlargest(5, "efficiency_score")
        if len(top5) >= 2:
            radar_metrics = ["CTR_total", "VTR", "노출수", "총클릭수", "광고비"]
            fig_radar = go.Figure()
            for _, row in top5.iterrows():
                vals = []
                for m in radar_metrics:
                    col_max = prod_agg_f[m].max()
                    vals.append(row[m] / col_max if col_max and col_max > 0 else 0)
                vals.append(vals[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=str(row["광고상품명_정리"])[:20],
                    opacity=0.7,
                ))
            apply_plotly_theme(fig_radar, height=380)
            fig_radar.update_layout(polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d", color="#8b949e"),
                angularaxis=dict(gridcolor="#30363d", color="#8b949e"),
            ))
            st.plotly_chart(fig_radar, use_container_width=True)

        # 인사이트 박스
        if not prod_agg_f.empty:
            best = prod_agg_f.dropna(subset=["efficiency_score"]).nlargest(1, "efficiency_score")
            worst = prod_agg_f.dropna(subset=["efficiency_score"]).nsmallest(1, "efficiency_score")
            if not best.empty:
                insight_box("🥇", f"최고 효율 상품: <strong>{best.iloc[0]['광고상품명_정리']}</strong> — CTR {best.iloc[0]['CTR_total']:.2%}, CPC ₩{best.iloc[0]['CPC_total']:,.0f}" if pd.notna(best.iloc[0]['CPC_total']) else f"최고 효율 상품: {best.iloc[0]['광고상품명_정리']}")
            if not worst.empty and len(prod_agg_f) > 1:
                insight_box("📉", f"개선 필요 상품: <strong>{worst.iloc[0]['광고상품명_정리']}</strong> — 효율점수 {worst.iloc[0]['efficiency_score']:.3f}" if pd.notna(worst.iloc[0].get('efficiency_score')) else f"개선 필요 상품: {worst.iloc[0]['광고상품명_정리']}")
            high_spend_low_ctr = prod_agg_f[(prod_agg_f["광고비"] > prod_agg_f["광고비"].median()) & (prod_agg_f["CTR_total"] < bench["CTR_total"])]
            if not high_spend_low_ctr.empty:
                names = ", ".join(high_spend_low_ctr["광고상품명_정리"].tolist()[:3])
                insight_box("💸", f"광고비 高 · CTR 低 → 예산 재배분 검토 필요: <strong>{names}</strong>")

    # ════════════════════════════════════════════
    # TAB 3: 시계열 트렌드
    # ════════════════════════════════════════════
    with tab3:
        section_header("시계열 성과 트렌드")

        c_f1, c_f2, c_f3 = st.columns([2, 2, 3])
        with c_f1:
            time_unit = st.radio("시간 단위", ["일별", "월별"], horizontal=True)
        with c_f2:
            t_metric = st.selectbox("추이 지표", ["CTR_total", "CPC_total", "eCPM", "노출수", "총클릭수", "광고비", "VTR"])
        with c_f3:
            all_prods = sorted(df["광고상품명_정리"].unique().tolist())
            sel_prods = st.multiselect("상품 선택", all_prods, default=all_prods[:min(5, len(all_prods))])

        t_col = "일자" if time_unit == "일별" else "연월"
        target = df[df["광고상품명_정리"].isin(sel_prods)] if sel_prods else df
        trend = aggregate(target, [t_col, "광고상품명_정리"]).sort_values(t_col)

        # 선 차트
        fig_line = px.line(
            trend, x=t_col, y=t_metric, color="광고상품명_정리",
            markers=True,
        )
        apply_plotly_theme(fig_line, height=360)
        fig_line.update_layout(hovermode="x unified")
        if t_metric in ("CTR_total", "VTR"):
            fig_line.update_yaxes(tickformat=".2%")
        elif t_metric in ("CPC_total", "eCPM", "광고비"):
            fig_line.update_yaxes(tickprefix="₩", tickformat=",.0f")
        st.plotly_chart(fig_line, use_container_width=True)

        # 누적 광고비 영역 차트
        st.write("")
        section_header("누적 광고비 추이 (상품별)")
        spend_trend = aggregate(df, [t_col, "광고상품명_정리"]).sort_values(t_col)
        fig_area = px.area(spend_trend, x=t_col, y="광고비", color="광고상품명_정리")
        apply_plotly_theme(fig_area, height=300)
        fig_area.update_yaxes(tickprefix="₩", tickformat=",.0f")
        st.plotly_chart(fig_area, use_container_width=True)

        # 월별 KPI 비교 테이블
        section_header("월별 KPI 집계")
        monthly = aggregate(df, ["연월"]).sort_values("연월")
        monthly["연월_str"] = monthly["연월"].dt.strftime("%Y-%m")
        m_disp = monthly[["연월_str", "노출수", "총클릭수", "광고비", "CTR_total", "CPC_total", "eCPM"]].copy()
        m_disp.columns = ["연월", "노출수", "총클릭수", "광고비(₩)", "CTR(%)", "CPC(₩)", "eCPM(₩)"]
        for col_, fmt_ in [("노출수", ","), ("총클릭수", ","), ("광고비(₩)", "won"), ("CTR(%)", "pct"), ("CPC(₩)", "won"), ("eCPM(₩)", "won")]:
            m_disp[col_] = m_disp[col_].apply(lambda x: fmt_num(x, fmt_) if pd.notna(x) else "N/A")
        style_dataframe(m_disp)

    # ════════════════════════════════════════════
    # TAB 4: 요일·구분별 분석
    # ════════════════════════════════════════════
    with tab4:
        c1, c2 = st.columns(2)

        with c1:
            section_header("요일별 성과 분포")
            day_agg = aggregate(df, ["요일"])
            day_agg["요일"] = pd.Categorical(day_agg["요일"], categories=DAY_ORDER, ordered=True)
            day_agg = day_agg.sort_values("요일")

            # 복합 차트: 노출 바 + CTR 선
            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(
                x=day_agg["요일"], y=day_agg["노출수"],
                name="노출수", marker_color="#1f6feb", opacity=0.7, yaxis="y",
            ))
            fig_day.add_trace(go.Scatter(
                x=day_agg["요일"], y=day_agg["CTR_total"],
                name="CTR(전체)", mode="lines+markers",
                marker=dict(size=8), line=dict(color="#3fb950", width=2),
                yaxis="y2",
            ))
            apply_plotly_theme(fig_day, height=320)
            fig_day.update_layout(
                yaxis=dict(title="노출수", gridcolor="#21262d"),
                yaxis2=dict(title="CTR", overlaying="y", side="right", tickformat=".2%", gridcolor="rgba(0,0,0,0)"),
                barmode="overlay",
            )
            st.plotly_chart(fig_day, use_container_width=True)

            # 요일별 평균 CPC
            fig_cpc = px.bar(
                day_agg, x="요일", y="CPC_total",
                color_discrete_sequence=["#d29922"],
                text=day_agg["CPC_total"].apply(lambda x: f"₩{x:,.0f}" if pd.notna(x) else ""),
            )
            apply_plotly_theme(fig_cpc, "요일별 평균 CPC", height=260)
            fig_cpc.update_traces(textposition="outside")
            st.plotly_chart(fig_cpc, use_container_width=True)

        with c2:
            section_header("구분(Category)별 성과 비교")
            cat_agg = aggregate(df, ["구분"])

            for met, color, title in [("CTR_total", "#58a6ff", "구분별 CTR"), ("CPC_total", "#d29922", "구분별 CPC"), ("eCPM", "#8957e5", "구분별 eCPM")]:
                fig_c = px.bar(
                    cat_agg.sort_values(met, ascending=(met == "CPC_total")),
                    x="구분", y=met,
                    color_discrete_sequence=[color],
                    text=cat_agg.sort_values(met, ascending=(met=="CPC_total"))[met].apply(
                        lambda x: f"{x:.2%}" if met in ("CTR_total",) else f"₩{x:,.0f}" if pd.notna(x) else ""
                    ),
                )
                apply_plotly_theme(fig_c, title, height=220)
                fig_c.update_traces(textposition="outside")
                st.plotly_chart(fig_c, use_container_width=True)

        # 상품 × 요일 히트맵
        st.write("")
        section_header(f"상품 × 요일 {primary_metric} 히트맵")
        heat_sel = st.multiselect("히트맵 상품 선택", sorted(df["광고상품명_정리"].unique().tolist()), default=sorted(df["광고상품명_정리"].unique().tolist())[:10], key="heat_prods")
        heat_df = df[df["광고상품명_정리"].isin(heat_sel)] if heat_sel else df
        heat_agg = aggregate(heat_df, ["광고상품명_정리", "요일"])
        heat_agg = heat_agg[heat_agg["노출수"] >= 1000]
        if not heat_agg.empty and primary_metric in heat_agg.columns:
            heat_pivot = heat_agg.pivot(index="광고상품명_정리", columns="요일", values=primary_metric).reindex(columns=DAY_ORDER)
            fig_heat = go.Figure(go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale=COLOR_SCALE,
                hoverongaps=False,
                hovertemplate="%{y} · %{x}<br>값: %{z:.4f}<extra></extra>",
            ))
            apply_plotly_theme(fig_heat, height=max(200, len(heat_pivot) * 36 + 80))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("조건에 맞는 데이터가 없습니다. 상품을 선택하거나 최소 노출 기준을 낮춰주세요.")

    # ════════════════════════════════════════════
    # TAB 5: 프로모션 분석
    # ════════════════════════════════════════════
    with tab5:
        section_header("프로모션별 성과 분석")

        promo_opts = sorted(df["프로모션명"].dropna().unique().tolist())
        sel_promos = st.multiselect("프로모션 선택", promo_opts, default=promo_opts[:min(10, len(promo_opts))])
        promo_df = df[df["프로모션명"].isin(sel_promos)] if sel_promos else df

        promo_agg = aggregate(promo_df, ["프로모션명", "구분"])
        promo_agg = compute_efficiency_score(promo_agg)
        promo_agg = promo_agg.sort_values(["프로모션명", "efficiency_score"], ascending=[True, False])

        # 테이블
        disp3 = promo_agg[["프로모션명", "구분", "노출수", "클릭수", "총클릭수", "광고비", "CTR_total", "CPC_total", "eCPM", "efficiency_grade"]].copy()
        disp3.columns = ["프로모션명", "구분", "노출수", "클릭수", "총클릭수", "광고비(₩)", "CTR(%)", "CPC(₩)", "eCPM(₩)", "등급"]
        for col_, fmt_ in [("노출수", ","), ("클릭수", ","), ("총클릭수", ","), ("광고비(₩)", "won"), ("CTR(%)", "pct"), ("CPC(₩)", "won"), ("eCPM(₩)", "won")]:
            disp3[col_] = disp3[col_].apply(lambda x: fmt_num(x, fmt_) if pd.notna(x) else "N/A")
        style_dataframe(disp3)

        st.write("")
        c1, c2 = st.columns(2)

        with c1:
            section_header("프로모션별 CTR (구분별 그룹)")
            fig_pc = px.bar(
                promo_agg,
                x="프로모션명", y="CTR_total",
                color="구분", barmode="group",
                text=promo_agg["CTR_total"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else ""),
            )
            apply_plotly_theme(fig_pc, height=340)
            fig_pc.update_xaxes(tickangle=-30)
            fig_pc.update_yaxes(tickformat=".2%")
            fig_pc.update_traces(textposition="outside")
            st.plotly_chart(fig_pc, use_container_width=True)

        with c2:
            section_header("프로모션별 광고비 (구분별 그룹)")
            fig_ps = px.bar(
                promo_agg,
                x="프로모션명", y="광고비",
                color="구분", barmode="stack",
            )
            apply_plotly_theme(fig_ps, height=340)
            fig_ps.update_xaxes(tickangle=-30)
            fig_ps.update_yaxes(tickprefix="₩", tickformat=",.0f")
            st.plotly_chart(fig_ps, use_container_width=True)

        # 프로모션 효율 산점도
        st.write("")
        section_header("프로모션 효율 포지셔닝")
        promo_sum = aggregate(promo_df, ["프로모션명"])
        promo_sum = compute_efficiency_score(promo_sum)
        valid_p = promo_sum[(promo_sum["CTR_total"].notna()) & (promo_sum["CPC_total"].notna()) & (promo_sum["광고비"] > 0)]
        if not valid_p.empty:
            fig_pp = px.scatter(
                valid_p,
                x="CPC_total", y="CTR_total",
                size="광고비", color="efficiency_grade",
                hover_name="프로모션명",
                color_discrete_map={"S": "#3fb950", "A": "#58a6ff", "B": "#d29922", "C": "#f85149", "N/A": "#8b949e"},
                size_max=50,
                text="프로모션명",
            )
            avg_ctr = valid_p["CTR_total"].mean()
            avg_cpc = valid_p["CPC_total"].mean()
            fig_pp.add_hline(y=avg_ctr, line_dash="dot", line_color="#8b949e", annotation_text="평균 CTR")
            fig_pp.add_vline(x=avg_cpc, line_dash="dot", line_color="#8b949e", annotation_text="평균 CPC")
            apply_plotly_theme(fig_pp, height=400)
            fig_pp.update_xaxes(title="CPC(₩)")
            fig_pp.update_yaxes(title="CTR", tickformat=".2%")
            fig_pp.update_traces(textposition="top center", textfont=dict(size=10))
            st.plotly_chart(fig_pp, use_container_width=True)

        st.write("")
        st.download_button(
            "📥 프로모션 효율 보고서 다운로드 (CSV)",
            data=promo_agg.to_csv(index=False).encode("utf-8-sig"),
            file_name="promotion_report.csv",
            mime="text/csv",
        )

    # ── 사이드바 하단: 데이터 다운로드 ──────────────
    with st.sidebar:
        st.divider()
        st.markdown('<div class="sidebar-title">⬇️ Export</div>', unsafe_allow_html=True)
        st.download_button(
            "필터링 데이터 다운로드",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_data.csv",
            mime="text/csv",
        )
        st.caption(f"v2.0 · {len(df):,}건 선택됨")


if __name__ == "__main__":
    main()
