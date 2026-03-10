"""
AD Performance Benchmark Dashboard v3
실행: streamlit run app.py
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AD Performance Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
# 0-1. 글로벌 CSS  (고대비 라이트 테마)
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── 전체 기반 ─────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div,
.main .block-container {
    background-color: #F0F4F8 !important;
    color: #1A202C !important;
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif !important;
}

/* ── 사이드바 ───────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background-color: #334155 !important;
    color: #E2E8F0 !important;
    border-color: #475569 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #334155 !important;
    border-color: #475569 !important;
    color: #E2E8F0 !important;
}
.sidebar-section-title {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 2px !important;
    color: #94A3B8 !important;
    margin: 16px 0 8px 0 !important;
    display: block !important;
}

/* ── 리포트 헤더 배너 ───────────────────────── */
.report-header {
    background: linear-gradient(135deg, #1E293B 0%, #0F4C81 60%, #1E3A5F 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.report-header::after {
    content: '';
    position: absolute; right: -30px; top: -30px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.report-header h1 { font-size: 22px; font-weight: 700; color: #F0F9FF !important; margin: 0 0 6px 0; }
.report-header p  { color: #BAE6FD !important; font-size: 13px; margin: 0; }

/* ── 섹션 타이틀 ────────────────────────────── */
.sec-title {
    font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.8px;
    color: #2563EB;
    border-left: 3px solid #2563EB;
    padding-left: 10px;
    margin: 20px 0 10px 0;
}

/* ── KPI 카드 ───────────────────────────────── */
.kpi-wrap {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 18px 14px;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
    min-height: 132px;
    height: 132px;
}
.kpi-wrap:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 10px 10px 0 0; }
.kpi-label  { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #64748B; margin-bottom: 6px; }
.kpi-val    { font-size: 24px; font-weight: 700; font-family: 'IBM Plex Mono', monospace; color: #0F172A; line-height: 1.1; }
.kpi-delta-pos { font-size: 11px; color: #16A34A; margin-top: 4px; font-weight: 500; }
.kpi-delta-neg { font-size: 11px; color: #DC2626; margin-top: 4px; font-weight: 500; }
.kpi-delta-neu { font-size: 11px; color: #94A3B8; margin-top: 4px; }

/* ── 탭 ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    border-radius: 10px 10px 0 0;
    gap: 0; padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px; font-weight: 500;
    color: #64748B !important;
    padding: 12px 18px;
    border-bottom: 2px solid transparent;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
    background: transparent !important;
}

/* ── 위젯 라벨 ──────────────────────────────── */
label[data-testid="stWidgetLabel"],
.stSelectbox label, .stMultiSelect label,
.stRadio label, .stCheckbox label,
.stNumberInput label, .stTextInput label,
.stDateInput label, .stSlider label {
    color: #1A202C !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* ── 데이터프레임 ───────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
}

/* ── 인사이트 박스 ──────────────────────────── */
.insight      { background:#EFF6FF; border:1px solid #BFDBFE; border-left:4px solid #2563EB; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; color:#1E3A5F; }
.insight-warn { background:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid #D97706; border-radius:6px; padding:10px 14px; margin:6px 0; font-size:13px; color:#78350F; }

/* ── 배지 ───────────────────────────────────── */
.badge        { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }
.badge-blue   { background:#DBEAFE; color:#1D4ED8; }
.badge-green  { background:#DCFCE7; color:#15803D; }
.badge-gray   { background:#F1F5F9; color:#475569; }

/* ── Streamlit metric 숨기기 ────────────────── */
[data-testid="metric-container"] { display: none !important; }

/* ── 구분선 ─────────────────────────────────── */
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# 1. 상수
# ─────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "캠페인명", "일자", "노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수",
    "CTR", "CTR(전체)", "VTR", "광고비", "eCPM", "CPC", "CPV",
    "연도", "월", "광고상품명_정리", "구분", "프로모션별", "광고주",
]
NUMERIC_COLUMNS = [
    "노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수",
    "CTR", "CTR(전체)", "VTR", "광고비", "eCPM", "CPC", "CPV", "연도", "월",
]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP   = {0:"월", 1:"화", 2:"수", 3:"목", 4:"금", 5:"토", 6:"일"}

# 업종 키워드 매핑
INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "금융·보험": [],
    "통신·IT": ["소니", "폴라로이드", "ASL로지텍콜라보"],
    "유통·이커머스": ["네이버스토어", "핫딜", "꽃다발"],
    "자동차": [],
    "식음료": [
        "과자세트", "과자마켓", "과자", "킷캣", "킷켓", "암소갈비", "천의삼", "정원삼",
        "동원참치", "에브리워터", "오밀당X중앙해장", "네꼬닭", "사과당x여우티",
    ],
    "엔터·미디어": [],
    "공공·기관": [],
    "여행·숙박": [],
    "패션·뷰티": ["캘빈클라인", "아미", "아페쎄", "스투시", "듀이셀", "바세린", "헤넬"],
    "의료·헬스": ["광동멀티비타민", "데이팩"],
    "특집": ["설기획전", "설선물준비했설", "커부해설문조사", "커부해어울리는브랜드", "커부해스트리머추천"],
    "굿즈": ["마플샵", "민교교록앵콜", "LCK이벤트", "포토북", "감스트굿즈", "숲다이어리"],
    "기타": [],
}

ACCENT_COLORS = ["#2563EB","#16A34A","#D97706","#DC2626","#7C3AED","#0891B2","#DB2777"]

PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Noto Sans KR, Apple SD Gothic Neo, sans-serif", color="#1A202C", size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0"),
    legend=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", borderwidth=1, font=dict(size=11)),
    colorway=ACCENT_COLORS,
)

# ─────────────────────────────────────────────────────
# 2. 데이터 처리
# ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

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
    out["연도"]  = out["연도"].fillna(out["일자"].dt.year)
    out["월"]    = out["월"].fillna(out["일자"].dt.month)
    out["연월"]  = pd.to_datetime(
        dict(year=out["연도"].astype("Int64"), month=out["월"].astype("Int64"), day=1),
        errors="coerce")
    for col in ["캠페인명", "광고상품명_정리", "구분"]:
        out[col] = out[col].astype("string").fillna("미분류")

    def normalize_product_name(name: str) -> str:
        s = str(name).strip()
        s = re.sub(r"\s+", "", s)
        # '_my구좌' 같은 suffix가 있으면 다른 상품으로 유지
        if "_" in s:
            return s
        # 괄호가 문자열 끝에만 붙은 경우 동일 상품으로 통합
        s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
        return s

    out["광고상품명_원본"] = out["광고상품명_정리"]
    out["광고상품명_정리"] = out["광고상품명_정리"].apply(normalize_product_name).astype("string")
    out["프로모션명"] = (
        out["프로모션별"].astype("string").fillna("미분류")
        if "프로모션별" in out.columns else out["캠페인명"].astype("string").fillna("미분류"))
    out["광고주"] = (
        out["광고주"].astype("string").fillna("미분류")
        if "광고주" in out.columns else "미분류")
    out["총클릭수"]     = out["총클릭수"].fillna(out["클릭수"]).fillna(0)
    out["클릭수"]       = out["클릭수"].fillna(0)
    out["노출수"]       = out["노출수"].fillna(0)
    out["동영상조회수"] = out["동영상조회수"].fillna(0)
    out["컴패니언배너클릭수"] = out["컴패니언배너클릭수"].fillna(0)

    def classify(name: str) -> str:
        n = str(name).upper()
        for industry, kws in INDUSTRY_KEYWORDS.items():
            if any(kw.upper() in n for kw in kws):
                return industry
        return "기타"
    out["업종"] = out["광고주"].apply(classify)
    return out

def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)

@st.cache_data(show_spinner=False)
def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False, as_index=False)[
            ["노출수","클릭수","컴패니언배너클릭수","총클릭수","동영상조회수"]
        ].sum(min_count=1).fillna(0)
    )
    agg["CTR"]       = safe_div(agg["클릭수"],      agg["노출수"])
    agg["CTR_total"] = safe_div(agg["총클릭수"],    agg["노출수"])
    agg["VTR"]       = safe_div(agg["동영상조회수"], agg["노출수"])
    total = agg["노출수"].sum()
    agg["노출비중"] = agg["노출수"] / total if total > 0 else 0
    return agg

def compute_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out[out["CTR_total"].notna() & (out["노출수"] > 0)].copy()
    if len(valid) < 2 or valid["CTR_total"].std(ddof=0) == 0:
        out["eff_score"] = np.nan
        out["eff_grade"] = "N/A"
        return out
    mu, sd = valid["CTR_total"].mean(), valid["CTR_total"].std(ddof=0)
    valid["eff_score"] = ((valid["CTR_total"] - mu) / sd).round(3)
    def grade(s):
        if s >= 1.0: return "S"
        if s >= 0.3: return "A"
        if s >= -0.3: return "B"
        return "C"
    valid["eff_grade"] = valid["eff_score"].apply(grade)
    merged = out.merge(valid[["eff_score","eff_grade"]], left_index=True, right_index=True, how="left")
    merged["eff_grade"] = merged["eff_grade"].fillna("N/A")
    return merged

# ─────────────────────────────────────────────────────
# 3. 필터
# ─────────────────────────────────────────────────────
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
    out = df
    if f.date_min:     out = out[out["일자"] >= f.date_min]
    if f.date_max:     out = out[out["일자"] <= f.date_max]
    if f.years:        out = out[out["연도"].isin(f.years)]
    if f.months:       out = out[out["월"].isin(f.months)]
    if f.weekdays:     out = out[out["요일"].isin(f.weekdays)]
    if f.categories:   out = out[out["구분"].isin(f.categories)]
    if f.industries:   out = out[out["업종"].isin(f.industries)]
    if f.products:     out = out[out["광고상품명_정리"].isin(f.products)]
    if f.campaigns:    out = out[out["캠페인명"].isin(f.campaigns)]
    return out

# ─────────────────────────────────────────────────────
# 4. UI 헬퍼
# ─────────────────────────────────────────────────────
def sec(title: str, badge: str = "", badge_type: str = "blue"):
    b = f'<span class="badge badge-{badge_type}" style="margin-left:8px;">{badge}</span>' if badge else ""
    st.markdown(f'<div class="sec-title">{title}{b}</div>', unsafe_allow_html=True)

def kpi_card(label: str, value: str, delta: str = "", delta_good=None, color: str = "#2563EB"):
    dcls = "kpi-delta-neu"
    if delta_good is True:  dcls = "kpi-delta-pos"
    if delta_good is False: dcls = "kpi-delta-neg"
    d_html = f'<div class="{dcls}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-wrap">
        <div class="kpi-accent" style="background:{color}"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-val">{value}</div>
        {d_html}
    </div>""", unsafe_allow_html=True)

def insight(text: str, warn=False):
    cls = "insight-warn" if warn else "insight"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def theme(fig: go.Figure, title: str = "", h: int = 340) -> go.Figure:
    fig.update_layout(**PLOTLY_THEME, title=dict(text=title, font=dict(size=13, color="#1A202C")), height=h)
    return fig

def fmt(v, mode="num"):
    if pd.isna(v): return "N/A"
    if mode == "pct": return f"{v:.2%}"
    if mode == "cnt":
        return f"{float(v):,.0f}"
    if mode == "f3":  return f"{v:.3f}"
    return f"{v:,.0f}"

def fmt_kpi_compact(v):
    if pd.isna(v):
        return "N/A"
    n = float(v)
    if abs(n) >= 100_000_000:
        return f"{n/100_000_000:.1f}억"
    if abs(n) >= 10_000:
        return f"{n/10_000:.1f}만"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}천"
    return f"{n:,.0f}"

def period_delta(df: pd.DataFrame) -> Dict:
    months_sorted = sorted(df["연월"].dropna().unique())
    if len(months_sorted) < 2: return {}
    cur  = df[df["연월"] == months_sorted[-1]]
    prev = df[df["연월"] == months_sorted[-2]]
    result: Dict = {}
    for key in ["노출수","총클릭수","동영상조회수"]:
        c, p = cur[key].sum(), prev[key].sum()
        result[key] = (c, (c-p)/p if p else None)
    c_imp, c_clk = cur["노출수"].sum(), cur["총클릭수"].sum()
    p_imp, p_clk = prev["노출수"].sum(), prev["총클릭수"].sum()
    c_ctr = c_clk/c_imp if c_imp else np.nan
    p_ctr = p_clk/p_imp if p_imp else np.nan
    result["CTR_total"] = (c_ctr, (c_ctr-p_ctr)/p_ctr if p_ctr else None)
    return result

def dstr(ch) -> Tuple[str, Optional[bool]]:
    if ch is None or pd.isna(ch): return "", None
    sign = "▲" if ch > 0 else "▼"
    return f"{sign} {abs(ch):.1%} vs 전월", ch > 0

# ─────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────
def main():
    # ── 사이드바 ──────────────────────────────────────
    with st.sidebar:
        st.markdown('<span class="sidebar-section-title">📂 데이터 소스</span>', unsafe_allow_html=True)
        upload = st.file_uploader("CSV 업로드", type=["csv"])
        path   = st.text_input("또는 파일 경로 입력")

    if upload is None and not path:
        st.markdown("""
        <div class="report-header">
            <h1>📊 AD Performance Report</h1>
            <p>왼쪽 사이드바에서 광고 데이터(CSV)를 업로드하면 분석이 시작됩니다.</p>
        </div>""", unsafe_allow_html=True)
        st.info("👈 사이드바에서 CSV 파일을 업로드해주세요.")
        st.stop()

    try:
        with st.spinner("데이터 로딩 중..."):
            raw = load_csv(upload.getvalue()) if upload else load_csv_path(path)
    except Exception as e:
        st.error(f"로드 실패: {e}"); st.stop()

    df_raw = preprocess(raw)

    # ── 전역 필터 사이드바 ──────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown('<span class="sidebar-section-title">🔽 필터</span>', unsafe_allow_html=True)

        mn, mx = df_raw["일자"].min(), df_raw["일자"].max()
        g_date = st.date_input("날짜 범위",
            value=(mn.date(), mx.date()) if pd.notna(mn) and pd.notna(mx) else ())

        years  = sorted(df_raw["연도"].dropna().astype(int).unique().tolist())
        months = sorted(df_raw["월"].dropna().astype(int).unique().tolist())
        wdays  = [d for d in DAY_ORDER if d in set(df_raw["요일"].dropna().unique())]
        cats   = sorted(df_raw["구분"].dropna().unique().tolist())
        inds   = sorted(df_raw["업종"].dropna().unique().tolist())
        prods  = sorted(df_raw["광고상품명_정리"].dropna().unique().tolist())
        camps  = sorted(df_raw["캠페인명"].dropna().unique().tolist())

        g_years  = st.multiselect("연도",   years,  default=years)
        g_months = st.multiselect("월",     months, default=months)
        g_wdays  = st.multiselect("요일",   wdays,  default=wdays)
        g_cats   = st.multiselect("구분",   cats,   default=cats)
        g_inds   = st.multiselect("업종",   inds,   default=inds)
        g_prods  = st.multiselect("광고상품", prods, default=prods)
        srch     = st.text_input("캠페인 검색", placeholder="키워드...")
        c_opts   = [c for c in camps if srch.lower() in c.lower()]
        g_camps  = st.multiselect("캠페인명", c_opts, default=c_opts)

        st.divider()
        st.markdown('<span class="sidebar-section-title">📐 공통 설정</span>', unsafe_allow_html=True)
        min_imp = st.number_input("최소 노출 기준", min_value=0, value=10000, step=5000)

    d0 = pd.to_datetime(g_date[0]) if g_date and len(g_date) > 0 else None
    d1 = pd.to_datetime(g_date[1]) if g_date and len(g_date) > 1 else None
    gf = Filters(d0, d1, g_years, g_months, g_wdays, g_cats, g_inds, g_prods, g_camps)
    df = apply_filters(df_raw, gf)

    # ── 헤더 ──────────────────────────────────────
    dl = ""
    if pd.notna(df["일자"].min()) and pd.notna(df["일자"].max()):
        dl = f"{df['일자'].min().date()} ~ {df['일자'].max().date()}"
    st.markdown(f"""
    <div class="report-header">
        <h1>📊 AD Performance Report</h1>
        <p>
            분석 기간: <strong style="color:#7DD3FC">{dl}</strong> &nbsp;|&nbsp;
            데이터: <strong style="color:#7DD3FC">{len(df):,}건</strong> &nbsp;|&nbsp;
            상품: <strong style="color:#7DD3FC">{df['광고상품명_정리'].nunique()}개</strong> &nbsp;|&nbsp;
            프로모션: <strong style="color:#7DD3FC">{df['프로모션명'].nunique()}개</strong> &nbsp;|&nbsp;
            업종: <strong style="color:#7DD3FC">{df['업종'].nunique()}개</strong>
        </p>
    </div>""", unsafe_allow_html=True)

    # ── KPI 카드 ────────────────────────────────────
    sec("전체 KPI 요약")
    deltas = period_delta(df)

    total_imp  = df["노출수"].sum()
    total_clk  = df["클릭수"].sum()
    total_tclk = df["총클릭수"].sum()
    total_vw   = df["동영상조회수"].sum()
    ctr_t      = total_tclk / total_imp if total_imp else np.nan
    vtr        = total_vw   / total_imp if total_imp else np.nan
    cpb        = (total_tclk - total_clk) / total_clk if total_clk else np.nan  # 컴패니언클릭 비율

    kpi_defs = [
        ("노출수",      fmt_kpi_compact(total_imp),  "노출수",      "#2563EB"),
        ("클릭수",      fmt_kpi_compact(total_clk),  "클릭수",      "#0891B2"),
        ("총클릭수",    fmt_kpi_compact(total_tclk), "총클릭수",    "#16A34A"),
        ("CTR (전체)",  fmt(ctr_t,      "pct"), "CTR_total",   "#D97706"),
        ("동영상 조회", fmt_kpi_compact(total_vw),   "동영상조회수","#7C3AED"),
        ("VTR",         fmt(vtr,        "pct"), None,          "#DB2777"),
        ("동반클릭율",  fmt(cpb, "pct") if pd.notna(cpb) else "N/A", None, "#DC2626"),
    ]
    cols = st.columns(7)
    for col, (lbl, val, dk, color) in zip(cols, kpi_defs):
        with col:
            dv, dg = "", None
            if dk and dk in deltas:
                dv, dg = dstr(deltas[dk][1])
            kpi_card(lbl, val, dv, dg, color)

    st.write("")

    # ═══════════════════════════════════════════════
    # 탭 레이아웃
    # ═══════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📦 상품 분석",
        "🔀 상품 × 프로모션",
        "📈 시계열 트렌드",
        "📅 요일·구분 분석",
        "🏭 업종 분석",
        "📋 지표 정의 (Appendix)",
    ])

    # ═══════════════════════════════════════════════
    # TAB 1 : 상품 분석
    # ═══════════════════════════════════════════════
    with tab1:
        prod_agg = aggregate(df, ["광고상품명_정리"])
        prod_agg = compute_efficiency(prod_agg)
        prod_agg = prod_agg[prod_agg["노출수"] >= min_imp]

        sec("상품별 성과 테이블")
        sort_by = st.selectbox("정렬 기준",
            ["CTR_total","총클릭수","노출수","CTR","VTR","eff_score"], key="prod_sort")
        prod_agg = prod_agg.sort_values(sort_by, ascending=False, na_position="last")

        bench_ctr = prod_agg["CTR_total"].median()

        disp = prod_agg[["광고상품명_정리","노출수","클릭수","총클릭수","동영상조회수",
                          "CTR","CTR_total","VTR","노출비중","eff_score","eff_grade"]].copy()
        disp.columns = ["상품명","노출수","클릭수","총클릭수","조회수",
                        "CTR(%)","CTR_전체(%)","VTR(%)","노출비중(%)","효율점수","등급"]
        for c, m in [("노출수","cnt"),("클릭수","cnt"),("총클릭수","cnt"),("조회수","cnt"),
                     ("CTR(%)","pct"),("CTR_전체(%)","pct"),("VTR(%)","pct"),("노출비중(%)","pct")]:
            disp[c] = disp[c].apply(lambda x: fmt(x, m))
        disp["효율점수"] = disp["효율점수"].apply(lambda x: fmt(x,"f3"))
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.download_button("📥 상품별 집계 다운로드",
            data=prod_agg.to_csv(index=False).encode("utf-8-sig"),
            file_name="product_summary.csv", mime="text/csv")

        st.write("")
        c1, c2 = st.columns(2)

        # 노출 + CTR 이중 바 (TOP 15)
        with c1:
            sec("TOP 15 — 노출수 & CTR")
            top15 = prod_agg.nlargest(15, "노출수")[["광고상품명_정리","노출수","총클릭수","CTR_total"]].sort_values("노출수")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top15["광고상품명_정리"], x=top15["노출수"],
                name="노출수", orientation="h",
                marker_color="#2563EB", opacity=0.85, xaxis="x"))
            fig.add_trace(go.Scatter(
                y=top15["광고상품명_정리"], x=top15["총클릭수"],
                name="총클릭수", mode="lines+markers+text", marker=dict(size=7, color="#16A34A"),
                text=[f"CTR {v:.2%}" if pd.notna(v) else "CTR N/A" for v in top15["CTR_total"]],
                textposition="middle right", xaxis="x2"
            ))
            theme(fig, h=420)
            fig.update_layout(
                xaxis=dict(title="노출수", gridcolor="#F1F5F9"),
                xaxis2=dict(title="총클릭수", overlaying="x", side="top", gridcolor="rgba(0,0,0,0)"),
                barmode="overlay", legend=dict(x=0.7, y=0.02))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("클릭 구성 TOP 12 (클릭 + 컴패니언배너클릭)")
            stk = prod_agg.nlargest(12, "총클릭수")[["광고상품명_정리","클릭수","컴패니언배너클릭수"]].copy()
            stk["총클릭수"] = stk["클릭수"] + stk["컴패니언배너클릭수"]
            stk = stk.sort_values("총클릭수", ascending=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(y=stk["광고상품명_정리"], x=stk["클릭수"],
                name="클릭수", orientation="h", marker_color="#2563EB"))
            fig2.add_trace(go.Bar(y=stk["광고상품명_정리"], x=stk["컴패니언배너클릭수"],
                name="컴패니언배너클릭", orientation="h", marker_color="#0891B2"))
            theme(fig2, h=420)
            fig2.update_layout(barmode="stack")
            st.plotly_chart(fig2, use_container_width=True)

        st.write("")
        c3, c4 = st.columns(2)

        # CTR 분포
        with c3:
            sec("CTR 분포 (중앙값 기준선)")
            fig3 = px.histogram(prod_agg.dropna(subset=["CTR_total"]),
                x="CTR_total", nbins=20, color_discrete_sequence=["#2563EB"])
            fig3.add_vline(x=bench_ctr, line_dash="dash", line_color="#DC2626",
                annotation_text=f"중앙값 {bench_ctr:.2%}",
                annotation_position="top right")
            theme(fig3, h=280)
            fig3.update_xaxes(tickformat=".2%", title="CTR(전체)")
            fig3.update_yaxes(title="상품 수")
            st.plotly_chart(fig3, use_container_width=True)

        # 노출 vs 클릭 산점도
        with c4:
            sec("노출수 vs 총클릭수 (크기=CTR)")
            sc = prod_agg[prod_agg["노출수"] > 0].copy()
            fig4 = px.scatter(sc, x="노출수", y="총클릭수",
                size="CTR_total", color="CTR_total",
                hover_name="광고상품명_정리",
                color_continuous_scale=["#DBEAFE","#1D4ED8"],
                size_max=30)
            theme(fig4, h=280)
            fig4.update_coloraxes(colorbar=dict(tickformat=".2%", title="CTR"))
            st.plotly_chart(fig4, use_container_width=True)

        # 인사이트
        if not prod_agg.empty:
            top1 = prod_agg.dropna(subset=["eff_score"]).nlargest(1,"eff_score")
            bot1 = prod_agg.dropna(subset=["eff_score"]).nsmallest(1,"eff_score")
            hl = prod_agg[(prod_agg["노출수"] > prod_agg["노출수"].median()) &
                          (prod_agg["CTR_total"] < bench_ctr)]
            if not top1.empty:
                insight(f"🏆 최고 효율 상품: <strong>{top1.iloc[0]['광고상품명_정리']}</strong> "
                        f"— CTR {top1.iloc[0]['CTR_total']:.2%}, 등급 {top1.iloc[0]['eff_grade']}")
            if not bot1.empty and len(prod_agg) > 1:
                insight(f"📉 개선 필요 상품: <strong>{bot1.iloc[0]['광고상품명_정리']}</strong> "
                        f"— CTR {bot1.iloc[0]['CTR_total']:.2%}", warn=True)
            if not hl.empty:
                names = ", ".join(hl["광고상품명_정리"].tolist()[:3])
                insight(f"💡 노출 높고 CTR 낮은 상품 → 소재/타겟 개선 검토: <strong>{names}</strong>", warn=True)

    # ═══════════════════════════════════════════════
    # TAB 2 : 상품 × 프로모션
    # ═══════════════════════════════════════════════
    with tab2:
        sec("상품 선택")
        all_prods_t2 = sorted(df["광고상품명_정리"].unique().tolist())
        sel_prod = st.selectbox("분석할 광고상품", all_prods_t2, key="t2_prod")

        df_sp = df[df["광고상품명_정리"] == sel_prod]
        promo_prod = aggregate(df_sp, ["프로모션명"])
        promo_prod = compute_efficiency(promo_prod)

        # 테이블
        sec(f"'{sel_prod}' — 프로모션별 성과")
        disp_pp = promo_prod[["프로모션명","노출수","클릭수","총클릭수","동영상조회수",
                               "CTR","CTR_total","VTR","노출비중","eff_score","eff_grade"]].copy()
        disp_pp.columns = ["프로모션명","노출수","클릭수","총클릭수","조회수",
                           "CTR(%)","CTR_전체(%)","VTR(%)","노출비중(%)","효율점수","등급"]
        for c, m in [("노출수","cnt"),("클릭수","cnt"),("총클릭수","cnt"),("조회수","cnt"),
                     ("CTR(%)","pct"),("CTR_전체(%)","pct"),("VTR(%)","pct"),("노출비중(%)","pct")]:
            disp_pp[c] = disp_pp[c].apply(lambda x: fmt(x, m))
        disp_pp["효율점수"] = disp_pp["효율점수"].apply(lambda x: fmt(x,"f3"))
        st.dataframe(disp_pp.sort_values("CTR_전체(%)", ascending=False),
                     use_container_width=True, hide_index=True)

        st.write("")
        c1, c2 = st.columns(2)

        with c1:
            sec("프로모션별 노출 · 클릭")
            sp_s = promo_prod.sort_values("노출수", ascending=False).head(15)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sp_s["프로모션명"], y=sp_s["노출수"], name="노출수", marker_color="#2563EB", yaxis="y"))
            fig.add_trace(go.Bar(x=sp_s["프로모션명"], y=sp_s["총클릭수"], name="총클릭수", marker_color="#16A34A", yaxis="y"))
            fig.add_trace(go.Scatter(
                x=sp_s["프로모션명"], y=sp_s["CTR_total"], name="CTR",
                mode="lines+markers+text", text=[f"{v:.2%}" if pd.notna(v) else "N/A" for v in sp_s["CTR_total"]],
                textposition="top center", marker=dict(size=7, color="#D97706"),
                line=dict(color="#D97706", width=2), yaxis="y2"
            ))
            theme(fig, h=320)
            fig.update_layout(
                xaxis_tickangle=-30,
                yaxis=dict(title="노출수", tickformat=",.0f"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, showticklabels=False, visible=False),
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("프로모션별 CTR 비교 (중앙값 기준)")
            sp_ctr = promo_prod.sort_values("CTR_total", ascending=True).head(20)
            bench_sp = sp_ctr["CTR_total"].median()
            colors  = ["#16A34A" if v >= bench_sp else "#DC2626" for v in sp_ctr["CTR_total"]]
            fig2 = go.Figure(go.Bar(
                y=sp_ctr["프로모션명"], x=sp_ctr["CTR_total"],
                orientation="h", marker_color=colors,
                text=[f"{v:.2%}" for v in sp_ctr["CTR_total"]],
                textposition="outside"))
            fig2.add_vline(x=bench_sp, line_dash="dot", line_color="#D97706",
                annotation_text="중앙값")
            theme(fig2, h=320)
            fig2.update_xaxes(tickformat=".2%")
            st.plotly_chart(fig2, use_container_width=True)

        # 전체 상품 × 프로모션 히트맵
        st.write("")
        sec("전체 상품 × 프로모션 히트맵")
        cf1, cf2 = st.columns([3, 2])
        with cf1:
            hm_metric = st.selectbox("히트맵 지표",
                ["CTR_total","CTR","VTR","총클릭수","노출수"], key="hm_met")
        with cf2:
            top_n = st.slider("상위 프로모션 수 (노출 기준)", 5, 30, 15, key="hm_n")

        pp_all = aggregate(df, ["광고상품명_정리","프로모션명"])
        top_promos = (pp_all.groupby("프로모션명")["노출수"].sum()
                      .nlargest(top_n).index.tolist())
        top_prods_hm = (pp_all.groupby("광고상품명_정리")["노출수"].sum()
                        .nlargest(20).index.tolist())
        hm_df = pp_all[pp_all["프로모션명"].isin(top_promos) &
                       pp_all["광고상품명_정리"].isin(top_prods_hm)]

        if not hm_df.empty and hm_metric in hm_df.columns:
            pivot = hm_df.pivot_table(index="광고상품명_정리", columns="프로모션명",
                                       values=hm_metric, aggfunc="mean")
            fmt_cb = ".2%" if hm_metric in ("CTR_total","CTR","VTR") else ",.0f"
            fig3 = go.Figure(go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],
                hoverongaps=False,
                hovertemplate="%{y}<br>%{x}<br>값: %{z:.4f}<extra></extra>",
                colorbar=dict(tickformat=fmt_cb, len=0.8)))
            theme(fig3, f"상품 × 프로모션 {hm_metric} 히트맵",
                  h=max(240, len(pivot)*32+100))
            fig3.update_xaxes(tickangle=-35, tickfont=dict(size=10))
            fig3.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("히트맵을 표시하기에 충분한 데이터가 없습니다.")

        # 선택 상품 프로모션별 월별 추이
        st.write("")
        sec(f"'{sel_prod}' — 프로모션별 월별 노출·CTR 추이")
        pp_monthly = aggregate(df[df["광고상품명_정리"]==sel_prod], ["연월","프로모션명"]).sort_values("연월")
        if not pp_monthly.empty:
            figm = go.Figure()
            for i, pnm in enumerate(sorted(pp_monthly["프로모션명"].unique())):
                sub = pp_monthly[pp_monthly["프로모션명"] == pnm]
                color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
                figm.add_trace(go.Bar(x=sub["연월"], y=sub["노출수"], name=f"{pnm} 노출", marker_color=color, opacity=0.45, yaxis="y"))
                figm.add_trace(go.Scatter(
                    x=sub["연월"], y=sub["CTR_total"], name=f"{pnm} CTR",
                    mode="lines+markers+text",
                    text=[f"{v:.2%}" if pd.notna(v) else "N/A" for v in sub["CTR_total"]],
                    textposition="top center",
                    line=dict(color=color, width=2), marker=dict(size=6), yaxis="y2"
                ))
            theme(figm, "월별 노출/CTR 통합", h=320)
            figm.update_layout(
                barmode="group",
                yaxis=dict(title="노출수", tickformat=",.0f"),
                yaxis2=dict(title="CTR", overlaying="y", side="right", tickformat=".2%", showgrid=False),
            )
            st.plotly_chart(figm, use_container_width=True)

        # 인사이트
        if not promo_prod.empty:
            best_p = promo_prod.dropna(subset=["eff_score"]).nlargest(1,"eff_score")
            wrst_p = promo_prod.dropna(subset=["eff_score"]).nsmallest(1,"eff_score")
            if not best_p.empty:
                insight(f"🥇 <strong>{sel_prod}</strong>에서 최고 효율 프로모션: "
                        f"<strong>{best_p.iloc[0]['프로모션명']}</strong> "
                        f"— CTR {best_p.iloc[0]['CTR_total']:.2%}")
            if not wrst_p.empty and len(promo_prod) > 1:
                insight(f"⚠️ 낮은 효율 프로모션: <strong>{wrst_p.iloc[0]['프로모션명']}</strong> "
                        f"— CTR {wrst_p.iloc[0]['CTR_total']:.2%}", warn=True)

        st.write("")
        sec("프로모션 기준으로 상품 성과 보기")
        promo_opts_t2 = sorted(df["프로모션명"].dropna().unique().tolist())
        sel_promo = st.selectbox("분석할 프로모션", promo_opts_t2, key="t2_promo")
        promo_rows = df[df["프로모션명"] == sel_promo]

        promo_prod_detail = aggregate(promo_rows, ["광고상품명_정리", "구분"]).sort_values("총클릭수", ascending=False)
        st.dataframe(
            promo_prod_detail[["광고상품명_정리", "구분", "노출수", "클릭수", "총클릭수", "CTR_total", "VTR", "노출비중"]],
            use_container_width=True,
            hide_index=True,
        )

        figp = go.Figure()
        top_prod_in_promo = promo_prod_detail.head(15)
        figp.add_trace(go.Bar(x=top_prod_in_promo["광고상품명_정리"], y=top_prod_in_promo["노출수"], name="노출수", marker_color="#2563EB", yaxis="y"))
        figp.add_trace(go.Bar(x=top_prod_in_promo["광고상품명_정리"], y=top_prod_in_promo["총클릭수"], name="총클릭수", marker_color="#16A34A", yaxis="y"))
        figp.add_trace(go.Scatter(x=top_prod_in_promo["광고상품명_정리"], y=top_prod_in_promo["CTR_total"], name="CTR", mode="lines+markers+text", text=[f"{v:.2%}" if pd.notna(v) else "N/A" for v in top_prod_in_promo["CTR_total"]], textposition="top center", line=dict(color="#D97706", width=2), marker=dict(size=7, color="#D97706"), yaxis="y2"))
        theme(figp, f"{sel_promo} 내 상품별 노출/클릭", h=300)
        figp.update_layout(
            barmode="group",
            xaxis_tickangle=-30,
            yaxis=dict(title="노출수/총클릭수", tickformat=",.0f"),
            yaxis2=dict(overlaying="y", side="right", showticklabels=False, visible=False, showgrid=False),
        )
        st.plotly_chart(figp, use_container_width=True)

        # 동일 업종 평균 대비 성과 비교
        promo_industry_mix = promo_rows.groupby("업종")["노출수"].sum().sort_values(ascending=False)
        dominant_industry = promo_industry_mix.index[0] if not promo_industry_mix.empty else "기타"

        promo_dom_agg = aggregate(promo_rows[promo_rows["업종"] == dominant_industry], ["프로모션명"])
        industry_promos = df[df["업종"] == dominant_industry]
        industry_baseline = aggregate(industry_promos, ["프로모션명"])

        if not promo_dom_agg.empty and not industry_baseline.empty:
            pr = promo_dom_agg.iloc[0]
            avg_row = {
                "노출수": industry_baseline["노출수"].mean(),
                "총클릭수": industry_baseline["총클릭수"].mean(),
                "CTR_total": industry_baseline["CTR_total"].mean(),
                "VTR": industry_baseline["VTR"].mean(),
            }

            comp = pd.DataFrame(
                {
                    "지표": ["노출수", "총클릭수", "CTR_total", "VTR"],
                    "선택 프로모션": [pr["노출수"], pr["총클릭수"], pr["CTR_total"], pr["VTR"]],
                    "동일 업종 평균": [avg_row["노출수"], avg_row["총클릭수"], avg_row["CTR_total"], avg_row["VTR"]],
                }
            )
            comp["평균 대비 배율"] = np.where(comp["동일 업종 평균"] > 0, comp["선택 프로모션"] / comp["동일 업종 평균"], np.nan)

            st.caption(f"비교 업종: **{dominant_industry}** (선택 프로모션 내 노출 기준 최대 업종)")
            st.dataframe(comp, use_container_width=True, hide_index=True)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=comp["지표"], y=comp["평균 대비 배율"], marker_color="#7C3AED", text=[f"{v:.2f}x" if pd.notna(v) else "N/A" for v in comp["평균 대비 배율"]], textposition="outside"))
            fig_comp.add_hline(y=1.0, line_dash="dash", line_color="#DC2626", annotation_text="업종 평균 = 1.0x")
            theme(fig_comp, f"{sel_promo} vs 동일 업종 평균 성과 배율", h=280)
            fig_comp.update_yaxes(title="배율 (x)")
            st.plotly_chart(fig_comp, use_container_width=True)

            if comp[comp["지표"] == "CTR_total"]["평균 대비 배율"].iloc[0] >= 1:
                insight(f"✅ <strong>{sel_promo}</strong>은(는) 업종 평균 대비 CTR이 높습니다.")
            else:
                insight(f"⚠️ <strong>{sel_promo}</strong>은(는) 업종 평균 대비 CTR이 낮아 개선 여지가 있습니다.", warn=True)

    # ═══════════════════════════════════════════════
    # TAB 3 : 시계열 트렌드 (듀얼 Y축)
    # ═══════════════════════════════════════════════
    with tab3:
        sec("시계열 트렌드 — 듀얼 지표 비교")

        col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 3])
        with col_a:
            time_unit = st.radio("시간 단위", ["월별","일별"], horizontal=True, key="t3_unit")
        with col_b:
            met1 = st.selectbox("좌측 Y축 지표",
                ["총클릭수","노출수","클릭수","동영상조회수"], index=0, key="t3_m1")
        with col_c:
            met2 = st.selectbox("우측 Y축 지표",
                ["노출수","총클릭수","CTR_total","CTR","VTR","동영상조회수"], index=0, key="t3_m2")
        with col_d:
            all_p3 = sorted(df["광고상품명_정리"].unique().tolist())
            sel_p3 = st.multiselect("상품 선택", all_p3,
                default=all_p3[:min(4, len(all_p3))], key="t3_prods")

        t_col  = "연월" if time_unit == "월별" else "일자"
        target = df[df["광고상품명_정리"].isin(sel_p3)] if sel_p3 else df
        trend  = aggregate(target, [t_col, "광고상품명_정리"]).sort_values(t_col)

        PCT = {"CTR_total","CTR","VTR"}
        y1_pct = met1 in PCT
        y2_pct = met2 in PCT

        # 듀얼 Y축 차트
        fig_dual = go.Figure()
        prod_list = sel_p3 if sel_p3 else sorted(trend["광고상품명_정리"].unique())

        for i, prod in enumerate(prod_list):
            sub   = trend[trend["광고상품명_정리"] == prod]
            color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
            fig_dual.add_trace(go.Scatter(
                x=sub[t_col], y=sub[met1],
                name=f"{str(prod)[:18]} ({met1})",
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                yaxis="y"))
            fig_dual.add_trace(go.Scatter(
                x=sub[t_col], y=sub[met2],
                name=f"{str(prod)[:18]} ({met2})",
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                yaxis="y2"))

        theme(fig_dual, h=420)
        fig_dual.update_layout(
            hovermode="x unified",
            yaxis=dict(
                title=dict(text=f"◀ {met1}", font=dict(color="#2563EB")),
                tickformat=".2%" if y1_pct else ",.0f",
                gridcolor="#F1F5F9", side="left"),
            yaxis2=dict(
                title=dict(text=f"{met2} ▶", font=dict(color="#D97706")),
                tickformat=".2%" if y2_pct else ",.0f",
                overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=10)))
        st.plotly_chart(fig_dual, use_container_width=True)
        st.caption("실선 = 좌측 Y축 지표 / 점선 = 우측 Y축 지표")

        # 전체 합산 노출+클릭 영역 차트
        st.write("")
        sec("전체 기간 노출 · 클릭 합산 추이")
        total_trend = aggregate(df, [t_col]).sort_values(t_col)
        fig_tot = go.Figure()
        fig_tot.add_trace(go.Scatter(
            x=total_trend[t_col], y=total_trend["노출수"],
            name="노출수", mode="lines",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="#2563EB", width=2)))
        fig_tot.add_trace(go.Scatter(
            x=total_trend[t_col], y=total_trend["총클릭수"],
            name="총클릭수", mode="lines+markers",
            line=dict(color="#16A34A", width=2), marker=dict(size=5),
            yaxis="y2"))
        theme(fig_tot, h=280)
        fig_tot.update_layout(
            yaxis=dict(title="노출수", gridcolor="#F1F5F9"),
            yaxis2=dict(title="총클릭수", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)", showgrid=False))
        st.plotly_chart(fig_tot, use_container_width=True)

        st.write("")
        sec("기간별 노출 캠페인 수")
        prod_cnt = target.groupby(t_col)["캠페인명"].nunique().reset_index(name="캠페인수")
        fig_pc = px.bar(prod_cnt, x=t_col, y="캠페인수", text="캠페인수", color_discrete_sequence=["#7C3AED"])
        theme(fig_pc, "기간별 노출 캠페인 개수", h=220)
        fig_pc.update_traces(textposition="outside")
        fig_pc.update_yaxes(title="캠페인 수", tickformat=",.0f")
        st.plotly_chart(fig_pc, use_container_width=True)

        # 월별 요약 테이블 (MoM 포함)
        st.write("")
        sec("월별 지표 요약 (전월 대비 포함)")
        mt = aggregate(df, ["연월"]).sort_values("연월")
        mt["연월_str"]       = mt["연월"].dt.strftime("%Y-%m")
        mt["노출_MoM"]       = mt["노출수"].pct_change()
        mt["클릭_MoM"]       = mt["총클릭수"].pct_change()
        mt["CTR_MoM"]        = mt["CTR_total"].pct_change()
        mt["조회수_MoM"]     = mt["동영상조회수"].pct_change()
        m_disp = mt[["연월_str","노출수","클릭수","총클릭수","동영상조회수",
                      "CTR_total","VTR","노출_MoM","클릭_MoM","CTR_MoM"]].copy()
        m_disp.columns = ["연월","노출수","클릭수","총클릭수","조회수",
                          "CTR_전체(%)","VTR(%)","노출_MoM","클릭_MoM","CTR_MoM"]
        for c, m in [("노출수","cnt"),("클릭수","cnt"),("총클릭수","cnt"),("조회수","cnt"),
                     ("CTR_전체(%)","pct"),("VTR(%)","pct"),
                     ("노출_MoM","pct"),("클릭_MoM","pct"),("CTR_MoM","pct")]:
            m_disp[c] = m_disp[c].apply(lambda x: fmt(x, m))
        st.dataframe(m_disp, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════
    # TAB 4 : 요일·구분 분석
    # ═══════════════════════════════════════════════
    with tab4:
        c1, c2 = st.columns(2)

        with c1:
            sec("요일별 노출 · 클릭 · CTR")
            day_agg = aggregate(df, ["요일"])
            day_agg["요일"] = pd.Categorical(day_agg["요일"], categories=DAY_ORDER, ordered=True)
            day_agg = day_agg.sort_values("요일")

            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(
                x=day_agg["요일"], y=day_agg["노출수"],
                name="노출수", marker_color="rgba(37,99,235,0.6)", yaxis="y"))
            fig_day.add_trace(go.Bar(
                x=day_agg["요일"], y=day_agg["총클릭수"],
                name="총클릭수", marker_color="rgba(22,163,74,0.9)", yaxis="y"))
            fig_day.add_trace(go.Scatter(
                x=day_agg["요일"], y=day_agg["CTR_total"],
                name="CTR(전체)", mode="lines+markers+text",
                marker=dict(size=8, color="#D97706"),
                line=dict(color="#D97706", width=2),
                text=[f"{v:.2%}" if pd.notna(v) else "N/A" for v in day_agg["CTR_total"]],
                textposition="top center",
                yaxis="y2"))
            theme(fig_day, "요일별 노출 · 클릭 · CTR", h=320)
            fig_day.update_layout(
                barmode="group",
                yaxis=dict(title="노출수/총클릭수", gridcolor="#F1F5F9", tickformat=",.0f"),
                yaxis2=dict(overlaying="y", side="right", showticklabels=False, visible=False, showgrid=False),
            )
            st.plotly_chart(fig_day, use_container_width=True)

            fig_vtr = px.bar(day_agg, x="요일", y="VTR",
                color_discrete_sequence=["#7C3AED"],
                text=day_agg["VTR"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else ""))
            theme(fig_vtr, "요일별 VTR", h=240)
            fig_vtr.update_yaxes(tickformat=".2%")
            fig_vtr.update_traces(textposition="outside")
            st.plotly_chart(fig_vtr, use_container_width=True)

        with c2:
            sec("구분(Category)별 노출 · 클릭 · CTR")
            cat_agg = aggregate(df, ["구분"])
            fig_cat = go.Figure()
            fig_cat.add_trace(go.Bar(x=cat_agg["구분"], y=cat_agg["노출수"],
                name="노출수", marker_color="#2563EB", yaxis="y"))
            fig_cat.add_trace(go.Bar(x=cat_agg["구분"], y=cat_agg["총클릭수"],
                name="총클릭수", marker_color="#16A34A", yaxis="y"))
            fig_cat.add_trace(go.Scatter(
                x=cat_agg["구분"], y=cat_agg["CTR_total"],
                name="CTR", mode="lines+markers+text",
                text=[f"{v:.2%}" if pd.notna(v) else "N/A" for v in cat_agg["CTR_total"]],
                textposition="top center",
                line=dict(color="#D97706", width=2), marker=dict(size=7, color="#D97706"),
                yaxis="y2"
            ))
            theme(fig_cat, "구분별 노출수 · 총클릭수", h=260)
            fig_cat.update_layout(
                barmode="group",
                yaxis=dict(title="노출수/총클릭수", tickformat=",.0f"),
                yaxis2=dict(overlaying="y", side="right", showticklabels=False, visible=False, showgrid=False),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            ca_s = cat_agg.sort_values("CTR_total", ascending=False)
            fig_cc = px.bar(ca_s, x="구분", y="CTR_total",
                color_discrete_sequence=["#D97706"],
                text=ca_s["CTR_total"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else ""))
            theme(fig_cc, "구분별 CTR(전체)", h=240)
            fig_cc.update_yaxes(tickformat=".2%")
            fig_cc.update_traces(textposition="outside")
            st.plotly_chart(fig_cc, use_container_width=True)

        # 상품 × 요일 히트맵
        st.write("")
        sec("상품 × 요일 히트맵")
        cf3, cf4 = st.columns([3, 2])
        with cf3:
            hm_d_m = st.selectbox("지표", ["CTR_total","총클릭수","노출수","VTR"], key="hm_day")
        with cf4:
            hm_d_p = st.multiselect("상품 선택",
                sorted(df["광고상품명_정리"].unique()),
                default=sorted(df["광고상품명_정리"].unique())[:10], key="hm_day_p")

        hd_df  = df[df["광고상품명_정리"].isin(hm_d_p)] if hm_d_p else df
        hd_agg = aggregate(hd_df, ["광고상품명_정리","요일"])
        if not hd_agg.empty and hm_d_m in hd_agg.columns:
            hd_pv  = hd_agg.pivot(index="광고상품명_정리", columns="요일",
                                   values=hm_d_m).reindex(columns=DAY_ORDER)
            fig_hd = go.Figure(go.Heatmap(
                z=hd_pv.values, x=hd_pv.columns.tolist(), y=hd_pv.index.tolist(),
                colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],
                hoverongaps=False))
            theme(fig_hd, h=max(200, len(hd_pv)*36+80))
            st.plotly_chart(fig_hd, use_container_width=True)

    # ═══════════════════════════════════════════════
    # TAB 5 : 업종 분석
    # ═══════════════════════════════════════════════
    with tab5:
        ind_dist = df["업종"].value_counts()
        covered  = (ind_dist[ind_dist.index != "기타"].sum() / len(df) * 100) if len(df) > 0 else 0

        if covered < 5:
            st.warning("⚠️ 광고주명 기반 업종 자동 분류 결과 대부분이 '기타'입니다. "
                       "INDUSTRY_KEYWORDS 딕셔너리에 데이터에 맞는 키워드를 추가하면 분류 정확도가 높아집니다.")

        sec("업종별 광고 성과", f"자동분류 커버율 {covered:.0f}%",
            "gray" if covered < 20 else "blue")

        ind_agg = aggregate(df, ["업종"])
        ind_agg = compute_efficiency(ind_agg)

        c1, c2 = st.columns(2)
        with c1:
            sec("업종별 노출수 · 총클릭수")
            ia_s = ind_agg.sort_values("노출수", ascending=False)
            fig_i = go.Figure()
            fig_i.add_trace(go.Bar(x=ia_s["업종"], y=ia_s["노출수"],
                name="노출수", marker_color="#2563EB"))
            fig_i.add_trace(go.Bar(x=ia_s["업종"], y=ia_s["총클릭수"],
                name="총클릭수", marker_color="#16A34A"))
            theme(fig_i, h=300)
            fig_i.update_layout(barmode="group", xaxis_tickangle=-30)
            st.plotly_chart(fig_i, use_container_width=True)

        with c2:
            sec("업종별 CTR(전체) 비교")
            ia_c = ind_agg.sort_values("CTR_total", ascending=True)
            b_ind = ia_c["CTR_total"].median()
            cols_ind = ["#16A34A" if v >= b_ind else "#DC2626" for v in ia_c["CTR_total"]]
            fig_ic = go.Figure(go.Bar(
                y=ia_c["업종"], x=ia_c["CTR_total"],
                orientation="h", marker_color=cols_ind,
                text=[f"{v:.2%}" for v in ia_c["CTR_total"]], textposition="outside"))
            fig_ic.add_vline(x=b_ind, line_dash="dot", line_color="#D97706",
                annotation_text="중앙값")
            theme(fig_ic, h=300)
            fig_ic.update_xaxes(tickformat=".2%")
            st.plotly_chart(fig_ic, use_container_width=True)

        # 업종 × 상품 히트맵
        st.write("")
        sec("업종 × 광고상품 CTR 히트맵")
        ip_agg = aggregate(df, ["업종","광고상품명_정리"])
        ip_pv  = ip_agg.pivot_table(index="업종", columns="광고상품명_정리",
                                     values="CTR_total", aggfunc="mean")
        if not ip_pv.empty:
            fig_ip = go.Figure(go.Heatmap(
                z=ip_pv.values, x=list(ip_pv.columns), y=list(ip_pv.index),
                colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,"#1D4ED8"]],
                colorbar=dict(tickformat=".2%")))
            theme(fig_ip, h=max(220, len(ip_pv)*36+80))
            fig_ip.update_xaxes(tickangle=-35, tickfont=dict(size=10))
            st.plotly_chart(fig_ip, use_container_width=True)

        # 업종별 집계 테이블
        sec("업종별 집계 테이블")
        id_d = ind_agg[["업종","노출수","클릭수","총클릭수","동영상조회수",
                         "CTR","CTR_total","VTR","노출비중","eff_grade"]].copy()
        id_d.columns = ["업종","노출수","클릭수","총클릭수","조회수",
                        "CTR(%)","CTR_전체(%)","VTR(%)","노출비중(%)","효율등급"]
        for c, m in [("노출수","cnt"),("클릭수","cnt"),("총클릭수","cnt"),("조회수","cnt"),
                     ("CTR(%)","pct"),("CTR_전체(%)","pct"),("VTR(%)","pct"),("노출비중(%)","pct")]:
            id_d[c] = id_d[c].apply(lambda x: fmt(x, m))
        st.dataframe(id_d.sort_values("CTR_전체(%)", ascending=False),
                     use_container_width=True, hide_index=True)

        sec("업종별 광고주 구성")
        adv_by_ind = (
            df.groupby("업종")["광고주"]
            .agg(lambda s: sorted(set(map(str, s))))
            .reset_index(name="광고주목록")
        )
        adv_by_ind["광고주수"] = adv_by_ind["광고주목록"].apply(len)
        adv_by_ind["광고주목록"] = adv_by_ind["광고주목록"].apply(lambda x: ", ".join(x[:10]) + (" ..." if len(x) > 10 else ""))
        st.dataframe(adv_by_ind.sort_values("광고주수", ascending=False), use_container_width=True, hide_index=True)

        if covered < 5:
            insight("💡 업종 분류는 광고주명 키워드 기반으로 동작합니다. "
                    "정확한 분류를 위해 코드 상단 INDUSTRY_KEYWORDS 딕셔너리에 "
                    "실제 광고주명/키워드를 추가해주세요.", warn=True)

    # ═══════════════════════════════════════════════
    # TAB 6 : Appendix — 지표 정의 및 산출 방식
    # ═══════════════════════════════════════════════
    with tab6:
        # CSS: appendix 전용 스타일
        st.markdown("""
        <style>
        .apx-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px 24px;
            margin-bottom: 14px;
        }
        .apx-card h4 {
            font-size: 15px; font-weight: 700;
            color: #1E3A5F; margin: 0 0 6px 0;
        }
        .apx-tag {
            display: inline-block;
            background: #DBEAFE; color: #1D4ED8;
            font-size: 11px; font-weight: 700;
            padding: 2px 8px; border-radius: 4px;
            margin-bottom: 8px; letter-spacing: 0.5px;
        }
        .apx-formula {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            border-radius: 0 6px 6px 0;
            padding: 10px 16px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 13px; color: #1E293B;
            margin: 8px 0 10px 0;
        }
        .apx-desc {
            font-size: 13px; color: #475569;
            line-height: 1.7; margin: 0;
        }
        .apx-note {
            font-size: 12px; color: #94A3B8;
            margin-top: 8px; padding-top: 8px;
            border-top: 1px dashed #E2E8F0;
        }
        .apx-grade-table {
            width: 100%; border-collapse: collapse;
            font-size: 13px; margin-top: 10px;
        }
        .apx-grade-table th {
            background: #F1F5F9; color: #475569;
            padding: 8px 12px; text-align: left;
            font-weight: 600; font-size: 12px;
            border-bottom: 2px solid #E2E8F0;
        }
        .apx-grade-table td {
            padding: 8px 12px; border-bottom: 1px solid #F1F5F9;
            color: #1A202C;
        }
        .apx-grade-table tr:last-child td { border-bottom: none; }
        .g-S { background:#DCFCE7; color:#15803D; padding:2px 8px; border-radius:4px; font-weight:700; }
        .g-A { background:#DBEAFE; color:#1D4ED8; padding:2px 8px; border-radius:4px; font-weight:700; }
        .g-B { background:#FEF3C7; color:#B45309; padding:2px 8px; border-radius:4px; font-weight:700; }
        .g-C { background:#FEE2E2; color:#B91C1C; padding:2px 8px; border-radius:4px; font-weight:700; }
        .apx-section-divider {
            border: none; border-top: 2px solid #E2E8F0;
            margin: 28px 0 20px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#1E293B,#0F4C81);
                    border-radius:10px; padding:20px 28px; margin-bottom:24px;">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:2px;color:#94A3B8;margin-bottom:6px;">Appendix</div>
            <div style="font-size:20px;font-weight:700;color:#F0F9FF;">지표 정의 및 산출 방식</div>
            <div style="font-size:13px;color:#BAE6FD;margin-top:4px;">
                대시보드에서 사용되는 모든 지표의 계산 공식, 해석 방법, 주의사항을 설명합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 1: 기본 트래픽 지표 ──────────────
        sec("① 기본 트래픽 지표")

        metrics_basic = [
            {
                "name": "노출수 (Impressions)",
                "tag": "RAW",
                "formula": "노출수 = 원천 데이터 합산값",
                "desc": "광고가 화면에 노출된 횟수입니다. 동일 사용자가 여러 번 보더라도 각각 1회로 계산됩니다. "
                        "이 대시보드에서는 업로드된 CSV의 <code>노출수</code> 컬럼을 그룹별로 단순 합산합니다.",
                "note": "📌 원천 데이터에 이미 집계된 값이 들어있으므로 별도 중복 제거 처리는 하지 않습니다.",
            },
            {
                "name": "클릭수 (Clicks)",
                "tag": "RAW",
                "formula": "클릭수 = 원천 데이터 합산값  (배너 본체 클릭만 포함)",
                "desc": "광고 본체(메인 배너)를 클릭한 횟수입니다. 컴패니언 배너 클릭은 포함되지 않습니다.",
                "note": None,
            },
            {
                "name": "총클릭수 (Total Clicks)",
                "tag": "RAW / 보정",
                "formula": "총클릭수 = 클릭수 + 컴패니언배너클릭수\n"
                           "※ 원천 데이터에 총클릭수 컬럼이 있으면 그 값 사용,\n"
                           "   없으면 클릭수로 대체 (fillna)",
                "desc": "메인 배너 클릭과 컴패니언 배너 클릭을 합산한 값입니다. "
                        "CTR(전체) 산출의 분자로 사용됩니다.",
                "note": "📌 원천 데이터에 총클릭수가 없는 경우 클릭수와 동일하게 처리되므로 "
                        "컴패니언 클릭이 누락될 수 있습니다.",
            },
            {
                "name": "동영상조회수 (Video Views)",
                "tag": "RAW",
                "formula": "동영상조회수 = 원천 데이터 합산값",
                "desc": "동영상 광고가 일정 시간(통상 30초 또는 영상 끝) 이상 재생된 횟수입니다. "
                        "동영상 포맷이 아닌 상품의 경우 0으로 처리됩니다.",
                "note": None,
            },
        ]

        for m in metrics_basic:
            st.markdown(f"""
            <div class="apx-card">
                <div class="apx-tag">{m['tag']}</div>
                <h4>{m['name']}</h4>
                <div class="apx-formula">{m['formula']}</div>
                <p class="apx-desc">{m['desc']}</p>
                {"<p class='apx-note'>" + m['note'] + "</p>" if m['note'] else ""}
            </div>""", unsafe_allow_html=True)

        # ── Section 2: 비율 지표 ──────────────────────
        st.markdown('<hr class="apx-section-divider">', unsafe_allow_html=True)
        sec("② 비율(Rate) 지표")

        metrics_rate = [
            {
                "name": "CTR — Click-Through Rate (클릭률)",
                "tag": "CALCULATED",
                "formula": "CTR (%) = 클릭수 ÷ 노출수 × 100",
                "desc": "메인 배너 클릭만을 기준으로 산출한 클릭률입니다. "
                        "노출 대비 얼마나 많은 사용자가 광고를 클릭했는지를 나타냅니다.",
                "note": "⚠️ 노출수가 0이면 NaN(N/A)으로 표시됩니다.",
            },
            {
                "name": "CTR(전체) — Total CTR",
                "tag": "CALCULATED",
                "formula": "CTR(전체) (%) = 총클릭수 ÷ 노출수 × 100\n"
                           "             = (클릭수 + 컴패니언배너클릭수) ÷ 노출수 × 100",
                "desc": "컴패니언 배너 클릭까지 포함한 전체 클릭률입니다. "
                        "대시보드의 주요 효율 판단 기준 지표로, 단순 CTR보다 광고 전체 반응률을 더 잘 반영합니다. "
                        "테이블의 <strong>CTR_전체(%)</strong>, KPI 카드의 <strong>CTR(전체)</strong>가 이 값입니다.",
                "note": "📌 이 대시보드에서 '효율'을 언급할 때 기본적으로 CTR(전체)를 기준으로 합니다.",
            },
            {
                "name": "VTR — View-Through Rate (조회율)",
                "tag": "CALCULATED",
                "formula": "VTR (%) = 동영상조회수 ÷ 노출수 × 100",
                "desc": "동영상 광고가 노출된 횟수 대비 완료 조회된 비율입니다. "
                        "동영상 포맷(예: 인스트림, 아웃스트림)에서만 의미 있는 지표이며, "
                        "동영상조회수가 0인 상품은 VTR도 0%로 표시됩니다.",
                "note": None,
            },
            {
                "name": "동반클릭율 (Companion Click Rate)",
                "tag": "CALCULATED",
                "formula": "동반클릭율 (%) = (총클릭수 - 클릭수) ÷ 클릭수 × 100\n"
                           "              = 컴패니언배너클릭수 ÷ 클릭수 × 100",
                "desc": "메인 클릭 대비 컴패니언 배너 클릭 비율입니다. "
                        "이 값이 높을수록 본 광고 클릭 이후 추가 탐색 행동이 활발함을 의미합니다. "
                        "KPI 카드 우측에 표시됩니다.",
                "note": "📌 클릭수(메인)가 0인 경우 N/A로 표시됩니다.",
            },
        ]

        for m in metrics_rate:
            st.markdown(f"""
            <div class="apx-card">
                <div class="apx-tag">{m['tag']}</div>
                <h4>{m['name']}</h4>
                <div class="apx-formula">{m['formula']}</div>
                <p class="apx-desc">{m['desc']}</p>
                {"<p class='apx-note'>" + m['note'] + "</p>" if m['note'] else ""}
            </div>""", unsafe_allow_html=True)

        # ── Section 3: 분석 파생 지표 ─────────────────
        st.markdown('<hr class="apx-section-divider">', unsafe_allow_html=True)
        sec("③ 분석 파생 지표")

        # 노출비중
        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">CALCULATED</div>
            <h4>노출비중 (Impression Share)</h4>
            <div class="apx-formula">노출비중 (%) = 해당 그룹의 노출수 ÷ 전체 집계 노출수 합계 × 100</div>
            <p class="apx-desc">
                현재 테이블에 집계된 전체 노출수 합계 대비 해당 행(상품/프로모션/업종 등)의
                노출수 비중을 나타냅니다.<br>
                예를 들어 상품 테이블에서 노출비중이 30%라면, 필터 적용 후 전체 노출의 30%가
                해당 상품에 집중되었음을 의미합니다.
            </p>
            <p class="apx-note">📌 필터 조건이 바뀌면 분모(전체 노출합)도 바뀌기 때문에
                절대값이 아닌 상대 비교용 지표입니다.</p>
        </div>
        """, unsafe_allow_html=True)

        # 효율점수
        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">CALCULATED — Z-Score</div>
            <h4>효율점수 (Efficiency Score)</h4>
            <div class="apx-formula">① 평균(μ), 표준편차(σ) 계산 (노출수 > 0인 행 대상)
μ  = 해당 집계 그룹 CTR(전체) 평균
σ  = 해당 집계 그룹 CTR(전체) 표준편차 (모표준편차, ddof=0)

② Z-Score 산출
효율점수 = (해당 행의 CTR_total − μ) ÷ σ

예시) 전체 평균 CTR 2.0%, σ = 0.5%
      상품 A CTR = 3.0%  →  효율점수 = (3.0 − 2.0) / 0.5 = +2.0  ← 평균보다 2σ 높음
      상품 B CTR = 1.5%  →  효율점수 = (1.5 − 2.0) / 0.5 = −1.0  ← 평균보다 1σ 낮음</div>
            <p class="apx-desc">
                CTR(전체)를 기준으로 같은 집계 그룹 내에서 상대적인 위치를 나타내는 표준화 점수(Z-Score)입니다.<br>
                <strong>양수</strong>일수록 그룹 평균보다 효율이 높고,
                <strong>음수</strong>일수록 낮습니다.<br><br>
                동일한 상품이라도 <strong>어떤 집계 단위(상품/프로모션/업종)</strong>로 계산하느냐에 따라
                점수가 달라집니다. 각 탭에서 독립적으로 계산됩니다.
            </p>
            <p class="apx-note">📌 집계 대상이 2개 미만이거나 모든 CTR이 동일하면 N/A로 표시됩니다.</p>
        </div>
        """, unsafe_allow_html=True)

        # 효율등급
        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">CALCULATED — Grade</div>
            <h4>효율등급 (Efficiency Grade)</h4>
            <div class="apx-formula">효율점수(Z-Score) 구간에 따라 S / A / B / C 부여

S : 효율점수 ≥ +1.0   (상위 약 16% — 평균 대비 1σ 이상 높음)
A : +0.3 ≤ 효율점수 &lt; +1.0  (상위 약 16~62%)
B : -0.3 ≤ 효율점수 &lt; +0.3  (중간 구간 — 평균 근처)
C : 효율점수 &lt; -0.3          (하위 약 38% — 평균 대비 낮음)</div>
            <p class="apx-desc">
                효율점수를 4단계 구간으로 나눈 등급입니다. 절대 기준이 아닌
                <strong>같은 집계 내 상대 순위</strong>를 기반으로 합니다.<br><br>
                단, 이 등급은 CTR만을 기준으로 하므로 노출 규모, 조회율 등은 반영되지 않습니다.
                등급이 낮더라도 노출비중이 높거나 VTR이 높으면 다른 측면에서 가치 있는 상품일 수 있습니다.
            </p>
            <p class="apx-note">
                📌 구간 기준 근거: 정규분포 가정 하에 Z = ±1은 상하위 약 16%,
                Z = ±0.3은 상하위 약 38%에 해당합니다.
                데이터 분포가 정규분포와 크게 다를 경우 비율은 달라질 수 있습니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 등급 요약 테이블
        st.markdown("""
        <div class="apx-card">
            <h4>📊 효율등급 요약표</h4>
            <table class="apx-grade-table">
                <thead>
                    <tr>
                        <th>등급</th>
                        <th>효율점수(Z) 구간</th>
                        <th>해석</th>
                        <th>정규분포 기준 비율</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="g-S">S</span></td>
                        <td>≥ +1.0</td>
                        <td>그룹 내 최상위 효율 — 적극 확대 검토</td>
                        <td>상위 약 16%</td>
                    </tr>
                    <tr>
                        <td><span class="g-A">A</span></td>
                        <td>+0.3 이상 ~ +1.0 미만</td>
                        <td>평균 이상 효율 — 유지 및 모니터링</td>
                        <td>상위 약 16~38%</td>
                    </tr>
                    <tr>
                        <td><span class="g-B">B</span></td>
                        <td>-0.3 이상 ~ +0.3 미만</td>
                        <td>평균 수준 효율 — 소재/타겟 최적화 검토</td>
                        <td>중간 약 24%</td>
                    </tr>
                    <tr>
                        <td><span class="g-C">C</span></td>
                        <td>-0.3 미만</td>
                        <td>평균 이하 효율 — 개선 또는 예산 재배분 필요</td>
                        <td>하위 약 38%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 4: 집계 및 전처리 ─────────────────
        st.markdown('<hr class="apx-section-divider">', unsafe_allow_html=True)
        sec("④ 집계 및 전처리 방식")

        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">PREPROCESSING</div>
            <h4>데이터 전처리 흐름</h4>
            <div class="apx-formula">원천 CSV 로드
  → 일자 컬럼: pd.to_datetime 변환 (파싱 실패 시 NaT)
  → 수치 컬럼: pd.to_numeric 변환 (파싱 실패 시 NaN)
  → 연도·월: 원천값 우선, 없으면 일자에서 추출
  → 연월: 연도+월로 날짜 재조합 (일=1 고정, 월별 집계용)
  → 요일: 일자의 weekday() → 한글 요일명 매핑
  → 총클릭수: 원천 총클릭수 → 없으면 클릭수로 대체 (fillna)
  → 업종: 광고주명 키워드 매칭 → 미매칭 시 '기타'</div>
            <p class="apx-desc">
                모든 수치 집계는 그룹별 <strong>단순 합산(SUM)</strong> 후 비율 지표를 재계산합니다.
                원천 데이터의 CTR, VTR 컬럼은 참고용으로만 보존되며,
                대시보드 내 모든 차트와 테이블은 합산 후 재계산된 값을 사용합니다.
            </p>
            <p class="apx-note">
                📌 원천 데이터의 CTR을 단순 평균하면 노출수 가중치가 무시되어 왜곡됩니다.
                이 대시보드는 항상 <strong>합산 노출수 기준으로 CTR을 재계산</strong>하므로 정확합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 5: 전월 대비 ─────────────────────
        st.markdown('<hr class="apx-section-divider">', unsafe_allow_html=True)
        sec("⑤ 전월 대비 (MoM) 계산")

        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">CALCULATED — MoM</div>
            <h4>전월 대비 변화율 (KPI 카드 델타, 월별 테이블 MoM)</h4>
            <div class="apx-formula">MoM 변화율 (%) = (당월 값 − 전월 값) ÷ 전월 값 × 100

KPI 카드: 데이터 내 가장 최근 월 vs 직전 월 비교
월별 테이블: 각 행의 전행 대비 변화율 (pandas pct_change)</div>
            <p class="apx-desc">
                KPI 카드의 델타(▲▼)는 필터 적용 데이터 기준 가장 최근 연월과 바로 직전 연월을 비교합니다.<br>
                필터로 특정 월만 선택한 경우 비교 대상이 없으므로 델타가 표시되지 않습니다.
            </p>
            <p class="apx-note">
                📌 전월 값이 0인 경우 변화율을 계산할 수 없어 N/A로 표시됩니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 6: 벤치마크 기준선 ────────────────
        st.markdown('<hr class="apx-section-divider">', unsafe_allow_html=True)
        sec("⑥ 차트 내 벤치마크 기준선")

        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">REFERENCE LINE</div>
            <h4>중앙값(Median) 기준선</h4>
            <div class="apx-formula">기준선 = 해당 차트에 표시된 집계 행들의 CTR(전체) 중앙값 (Median)
         ← 평균(Mean)이 아닌 중앙값 사용</div>
            <p class="apx-desc">
                CTR 분포 히스토그램, 프로모션별 CTR 바차트 등에 빨간/노란 점선으로 표시되는 기준선입니다.<br>
                <strong>평균(Mean) 대신 중앙값(Median)</strong>을 사용하는 이유:
                극단적으로 높거나 낮은 CTR을 가진 상품이 있을 경우 평균이 왜곡될 수 있기 때문입니다.
                중앙값은 이상치에 강건(robust)하여 '전형적인 성과' 기준으로 더 적합합니다.
            </p>
            <p class="apx-note">
                📌 기준선은 현재 필터 조건 및 표시 범위 내 데이터 기준으로 동적으로 계산됩니다.
                필터를 바꾸면 기준선 위치도 변합니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 7: 업종 분류 ──────────────────────
        st.markdown('<hr class="apx-section-divider">', unsafe_allow_html=True)
        sec("⑦ 업종 자동 분류 방식")

        st.markdown("""
        <div class="apx-card">
            <div class="apx-tag">HEURISTIC</div>
            <h4>업종 분류 알고리즘</h4>
            <div class="apx-formula">for 광고주명 in 데이터:
    광고주명.upper() 변환
    for 업종, 키워드목록 in INDUSTRY_KEYWORDS:
        if 키워드 in 광고주명:
            → 해당 업종으로 분류 (첫 번째 매칭 우선)
    매칭 실패 → '기타'로 분류</div>
            <p class="apx-desc">
                광고주명 문자열에 사전 정의된 키워드가 포함되어 있는지 단순 문자열 포함 여부(substring match)로 판단합니다.<br>
                현재 등록된 업종: 금융·보험 / 통신·IT / 유통·이커머스 / 자동차 / 식음료 /
                엔터·미디어 / 공공·기관 / 여행·숙박 / 패션·뷰티 / 의료·헬스<br><br>
                광고주명이 약어, 영문, 브랜드명으로만 구성된 경우 자동 분류가 어렵습니다.
                정확한 분류가 필요한 경우 코드 상단 <code>INDUSTRY_KEYWORDS</code> 딕셔너리에
                실제 광고주명을 키워드로 추가해주세요.
            </p>
            <p class="apx-note">
                📌 동일 광고주명이 여러 업종 키워드에 매칭되는 경우 딕셔너리 상단의 업종이 우선 적용됩니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 푸터
        st.markdown("""
        <div style="margin-top:32px; padding:16px 20px; background:#F8FAFC;
                    border:1px solid #E2E8F0; border-radius:8px;
                    font-size:12px; color:#94A3B8; text-align:center;">
            AD Performance Report v3.0 &nbsp;|&nbsp;
            지표 계산 기준일: 데이터 업로드 시점 &nbsp;|&nbsp;
            문의: 대시보드 담당자
        </div>
        """, unsafe_allow_html=True)

    # ── 사이드바 하단 다운로드 ──────────────────────
    with st.sidebar:
        st.divider()
        st.markdown('<span class="sidebar-section-title">⬇️ Export</span>', unsafe_allow_html=True)
        st.download_button(
            "필터링 데이터 다운로드",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_ad_data.csv", mime="text/csv")
        st.caption(f"v3.0 · {len(df):,}건 선택됨")


if __name__ == "__main__":
    main()
