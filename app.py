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
.kpi-wrap:hover { border-color: #2563EB; box-shadow: 0 2px 10px rgba(37,99,235,0.12); transition: all 0.18s ease; cursor: default; }
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

REQUIRED_COLUMNS = ["프로모션별", "캠페인명", "일자", "노출수", "클릭수", "라이브일자", "연도", "월", "광고상품명_정리", "구분", "광고주"]
NUMERIC_COLUMNS = ["노출수", "클릭수", "연도", "월"]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
COLORS = ["#2563EB", "#1E40AF", "#16A34A", "#D97706", "#7C3AED", "#0891B2", "#DB2777", "#0EA5E9"]

INDUSTRY_KEYWORDS = {
    "패션·뷰티": ["캘빈클라인", "아미", "아페쎄", "스투시", "듀이셀", "바세린", "헤넬", "언더웨어"],
    "식음료": ["과자세트", "과자마켓", "과자", "킷캣", "킷켓", "암소갈비", "동원참치", "에브리워터", "네꼬닭", "사과당x여우티", "사과당X여우티", "오밀당X중앙해장", "오밀당x중앙해장", "삼겹살", "링티", "갈비", "건어물", "더블크런치", "순대국밥", "덴마크"],
    "통신·IT": ["소니", "폴라로이드", "ASL로지텍콜라보", "asl로지텍"],
    "의료·헬스": ["광동멀티비타민", "데이팩", "정원삼", "천의삼"],
    "굿즈·이벤트": ["마플샵", "포토북", "감스트굿즈", "민교교록앵콜", "LCK이벤트", "lck이벤트", "봉준"],
    "유통·이커머스": ["네이버스토어", "꽃다발", "핫딜", "커머스를부탁해"],
    "특집·기획": ["설기획전", "설선물준비했설", "숲다이어리", "커부해", "커부해어울리는브랜드", "커부해스트리머추천", "커부해설문조사"],
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
    out["라이브일자"] = pd.to_datetime(out["라이브일자"], errors="coerce")

    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["광고상품명_정리"] = out["광고상품명_정리"].astype(str).str.strip()
    out["광고상품명_원본"] = out["광고상품명_정리"]
    out["광고상품명_정리"] = out["광고상품명_정리"].apply(normalize_product_name)

    out["구분"] = out["구분"].astype(str).str.strip()
    out["광고주"] = out["광고주"].astype(str).str.strip()
    out["캠페인명"] = out["캠페인명"].astype(str).fillna("미분류")
    out["프로모션별"] = out["프로모션별"].astype(str).fillna("미분류")
    out["프로모션명"] = out["프로모션별"].astype(str).fillna("미분류")

    out["노출수"] = pd.to_numeric(out["노출수"], errors="coerce").fillna(0)
    out["클릭수"] = pd.to_numeric(out["클릭수"], errors="coerce").fillna(0)
    out["총클릭수"] = out["클릭수"]
    out["컴패니언배너클릭수"] = 0
    out["동영상조회수"] = 0
    out["is_video"] = out["광고상품명_정리"].str.contains("인스트림", na=False)

    out["요일"] = out["일자"].dt.weekday.map(DAY_MAP)
    out["연도"] = out["연도"].fillna(out["일자"].dt.year)
    out["월"] = out["월"].fillna(out["일자"].dt.month)
    out["연월"] = pd.NaT
    valid_mask = out["연도"].notna() & out["월"].notna()
    out.loc[valid_mask, "연월"] = pd.to_datetime(
        dict(
            year=out.loc[valid_mask, "연도"].astype(int),
            month=out.loc[valid_mask, "월"].astype(int),
            day=1,
        ),
        errors="coerce",
    )

    def classify(v: str) -> str:
        n = re.sub(r"\s+", "", str(v)).upper()
        for ind, kws in INDUSTRY_KEYWORDS.items():
            if any(re.sub(r"\s+", "", k).upper() in n for k in kws):
                return ind
        return "기타"

    out["업종"] = out["광고주"].apply(classify)
    return out


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


@st.cache_data(show_spinner=False)
def aggregate(df: pd.DataFrame, by: tuple) -> pd.DataFrame:
    g = (
        df.groupby(list(by), dropna=False, as_index=False)[["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수"]]
        .sum(min_count=1)
        .fillna(0)
    )
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
        return f"{sign}{n / 100_000_000:.1f}억"
    if n >= 10_000:
        return f"{sign}{n / 10_000:.1f}만"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.1f}천"
    return f"{sign}{n:,.0f}"


def fmt_kpi_compact(v) -> str:
    return fmt_kor_unit(v)


def ellipsis(v: str, n: int = 16) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


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
        margin=dict(l=28, r=28, t=100, b=160),
        legend=dict(orientation="h", y=-0.42, yanchor="top", x=0, xanchor="left"),
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


def promotion_product_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, min_imp: int) -> pd.DataFrame:
    a = aggregate(df_a, ("광고상품명_정리",))[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR"]]
    b = aggregate(df_b, ("광고상품명_정리",))[["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "VTR"]]
    a = a.rename(columns={"노출수": "A_노출수", "총클릭수": "A_총클릭수", "CTR_total": "A_CTR", "VTR": "A_VTR"})
    b = b.rename(columns={"노출수": "B_노출수", "총클릭수": "B_총클릭수", "CTR_total": "B_CTR", "VTR": "B_VTR"})
    m = a.merge(b, on="광고상품명_정리", how="outer").fillna(0)
    m["노출 증감"] = m["B_노출수"] - m["A_노출수"]
    m["총클릭 증감"] = m["B_총클릭수"] - m["A_총클릭수"]
    m["CTR 변화(%p)"] = (m["B_CTR"] - m["A_CTR"]) * 100
    m["VTR 변화(%p)"] = (m["B_VTR"] - m["A_VTR"]) * 100
    m["해석상태"] = np.where((m["A_노출수"] < min_imp) | (m["B_노출수"] < min_imp), "표본부족", "분석가능")
    return m.sort_values("노출 증감", ascending=False)


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


def delta_sentence_count(a: float, b: float) -> str:
    d = b - a
    if abs(d) < 1e-12:
        return "A와 B가 동일"
    return f"B가 A보다 {fmt_kor_unit(abs(d))} {'큼' if d > 0 else '적음'}"


def delta_sentence_pp(a: float, b: float) -> str:
    if pd.isna(a) or pd.isna(b):
        return "비교 불가"
    d = (b - a) * 100
    if abs(d) < 1e-12:
        return "A와 B가 동일"
    return f"B가 A보다 {abs(d):.2f}%p {'높음' if d > 0 else '낮음'}"


def build_volume_click_ctr_chart(dfv: pd.DataFrame, x: str, title: str) -> go.Figure:
    ctr_series = dfv["CTR_total"].fillna(0)
    ctr_max = float(ctr_series.max()) if not ctr_series.empty else 0.0
    ctr_upper = max(0.01, nice_axis_max(ctr_max, pad=0.25))

    fig = go.Figure()
    fig.add_bar(
        x=dfv[x],
        y=dfv["노출수"],
        name="노출수",
        marker_color="#2563EB",
        yaxis="y",
        offsetgroup="imp",
        legendgroup="imp",
        hovertemplate="노출수: %{customdata}<extra></extra>",
        customdata=[fmt_kor_unit(v) for v in dfv["노출수"]],
    )
    fig.add_bar(
        x=dfv[x],
        y=dfv["총클릭수"],
        name="클릭수",
        marker_color="#16A34A",
        yaxis="y2",
        offsetgroup="clk",
        legendgroup="clk",
        hovertemplate="클릭수: %{customdata}<extra></extra>",
        customdata=[fmt_kor_unit(v) for v in dfv["총클릭수"]],
    )
    fig.add_scatter(
        x=dfv[x],
        y=dfv["CTR_total"],
        name="CTR_total",
        mode="lines+markers+text",
        line=dict(color="#D97706", width=2),
        marker=dict(size=7),
        text=[fmt_pct(v) for v in dfv["CTR_total"]],
        textposition="top center",
        hovertemplate="CTR_total: %{text}<extra></extra>",
        cliponaxis=False,
        yaxis="y3",
    )
    apply_meta_theme(fig, title, 390)
    y1v, y1t = make_kor_ticks(float(dfv["노출수"].max()) if not dfv.empty else 0)
    y2v, y2t = make_kor_ticks(float(dfv["총클릭수"].max()) if not dfv.empty else 0)
    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-28, automargin=True),
        yaxis=dict(title="노출수", tickmode="array", tickvals=y1v, ticktext=y1t, range=[0, y1v[-1]], gridcolor="#EEF2F7"),
        yaxis2=dict(title="클릭수", overlaying="y", side="right", tickmode="array", tickvals=y2v, ticktext=y2t, range=[0, y2v[-1]], showgrid=False),
        yaxis3=dict(overlaying="y", side="right", range=[0, ctr_upper], showticklabels=False, visible=False, showgrid=False, zeroline=False),
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

    if not {"노출수", "클릭수"}.issubset(raw.columns):
        st.error("CSV에 필수 컬럼(노출수, 클릭수)이 없습니다.")
        st.stop()

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

    date_min = df["일자"].min()
    date_max = df["일자"].max()
    date_str = f"{date_min.date()} ~ {date_max.date()}" if pd.notna(date_min) and pd.notna(date_max) else "날짜 미상"
    st.markdown(
        f"<div class='report-header'><h1>📊 AD Performance Report</h1><p>기간 {date_str} · 행 {len(df):,} · 상품 {df['광고상품명_정리'].nunique()} · 프로모션 {df['프로모션별'].nunique()}</p></div>",
        unsafe_allow_html=True,
    )

    has_video_data = df["동영상조회수"].sum() > 0

    sec("전체 KPI 요약")
    total_imp = df["노출수"].sum()
    total_clk = df["클릭수"].sum()
    total_tclk = df["총클릭수"].sum()
    total_vw = df["동영상조회수"].sum()
    ctr_t = total_tclk / total_imp if total_imp else np.nan
    vtr = total_vw / total_imp if total_imp else np.nan

    kpis = [
        ("노출수", fmt_kpi_compact(total_imp), ""),
        ("클릭수", fmt_kpi_compact(total_clk), ""),
        ("총클릭수", fmt_kpi_compact(total_tclk), ""),
        ("CTR(전체)", fmt_pct(ctr_t), ""),
        ("동영상조회수", fmt_kpi_compact(total_vw), "") if has_video_data else ("광고상품수", fmt_kpi_compact(df["광고상품명_정리"].nunique()), ""),
        ("VTR", fmt_pct(vtr), "") if has_video_data else ("캠페인수", fmt_kpi_compact(df["캠페인명"].nunique()), ""),
    ]
    cols = st.columns(6)
    for i, (lbl, val, delta) in enumerate(kpis):
        with cols[i]:
            kpi_card(lbl, val, delta)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📦 상품 분석", "🔀 상품 × 프로모션", "📈 시계열", "🔁 프로모션 비교", "📅 요일·구분", "🏭 업종", "📋 Appendix"
    ])

    with tab1:
        agg = compute_efficiency(aggregate(df, ("광고상품명_정리",)))
        agg["표본부족"] = np.where(agg["노출수"] < min_imp, "표본부족", "분석가능")

        cols_base = ["광고상품명_정리", "노출수", "총클릭수", "CTR_total", "표본부족", "eff_grade"]
        if has_video_data:
            cols_base.insert(4, "VTR")
        disp = agg[cols_base].copy()
        sec("상품 성과 테이블")
        if "VTR" in disp.columns:
            disp.columns = ["상품", "노출수", "총클릭수", "CTR_total", "VTR", "해석상태", "효율등급"]
        else:
            disp.columns = ["상품", "노출수", "총클릭수", "CTR_total", "해석상태", "효율등급"]
        for c in ["노출수", "총클릭수"]:
            disp[c] = disp[c].map(fmt_table_number)
        disp["CTR_total"] = disp["CTR_total"].map(fmt_pct)
        if "VTR" in disp.columns:
            disp["VTR"] = disp["VTR"].map(fmt_pct)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.download_button("📥 CSV 다운로드", data=agg.to_csv(index=False, encoding="utf-8-sig"), file_name="product_summary.csv", mime="text/csv")

        plot_df = agg.sort_values("노출수", ascending=False).head(15).copy()
        plot_df["상품축"] = plot_df["광고상품명_정리"].map(lambda x: ellipsis(x, 14))
        sec("TOP 15 상품 — 노출수 · 클릭수 · CTR_total")
        st.plotly_chart(build_volume_click_ctr_chart(plot_df, "상품축", "볼륨 중심 비교 (CTR은 데이터레이블)"), use_container_width=True)

        sec("버블 포트폴리오 매트릭스")
        show_low_sample = st.checkbox("표본부족 포함", value=False, key="bubble_low_sample")
        bubble_df = agg.copy() if show_low_sample else agg[agg["노출수"] >= min_imp].copy()
        if not bubble_df.empty:
            med_x = bubble_df["노출수"].median()
            med_y = bubble_df["CTR_total"].median()
            x_max = max(float(bubble_df["노출수"].max()), 1)
            y_max = max(float(bubble_df["CTR_total"].max()), 0.01)
            y_min = max(0.0, float(bubble_df["CTR_total"].min()) * 0.9)
            x_left = med_x * 0.5 if med_x * 0.5 < med_x else med_x * 0.1
            x_right = min(med_x * 1.4, x_max * 0.92)

            fig_bubble = go.Figure()
            fig_bubble.add_shape(type="rect", x0=0, x1=med_x, y0=y_min, y1=med_y, fillcolor="rgba(239,68,68,0.05)", line=dict(width=0), layer="below")
            fig_bubble.add_shape(type="rect", x0=med_x, x1=x_max * 1.05, y0=y_min, y1=med_y, fillcolor="rgba(251,146,60,0.05)", line=dict(width=0), layer="below")
            fig_bubble.add_shape(type="rect", x0=0, x1=med_x, y0=med_y, y1=y_max * 1.15, fillcolor="rgba(59,130,246,0.06)", line=dict(width=0), layer="below")
            fig_bubble.add_shape(type="rect", x0=med_x, x1=x_max * 1.05, y0=med_y, y1=y_max * 1.15, fillcolor="rgba(16,185,129,0.07)", line=dict(width=0), layer="below")
            fig_bubble.add_vline(x=med_x, line=dict(color="#64748B", dash="dot"))
            fig_bubble.add_hline(y=med_y, line=dict(color="#64748B", dash="dot"))
            plot_b = bubble_df.sort_values("총클릭수", ascending=False).copy()
            bubble_text = [ellipsis(v, 12) if i < 10 else "" for i, v in enumerate(plot_b["광고상품명_정리"])]
            fig_bubble.add_trace(
                go.Scatter(
                    x=plot_b["노출수"],
                    y=plot_b["CTR_total"],
                    mode="markers+text",
                    text=bubble_text,
                    textposition="top center",
                    name="광고상품",
                    customdata=np.array(
                        [
                            [n, fmt_kor_unit(i), fmt_kor_unit(c), fmt_pct(ct), fmt_pct(v)]
                            for n, i, c, ct, v in zip(plot_b["광고상품명_정리"], plot_b["노출수"], plot_b["총클릭수"], plot_b["CTR_total"], plot_b["VTR"])
                        ],
                        dtype=object,
                    ),
                    hovertemplate="상품명: %{customdata[0]}<br>노출수: %{customdata[1]}<br>클릭수: %{customdata[2]}<br>CTR_total: %{customdata[3]}<br>VTR: %{customdata[4]}<extra></extra>",
                    marker=dict(size=np.clip(np.sqrt(plot_b["총클릭수"].fillna(0)) * 2.5, 12, 55), color="#2563EB", opacity=0.55, line=dict(color="#1D4ED8", width=1)),
                )
            )
            x_min_range, x_max_range = 0.0, x_max * 1.05
            y_min_range, y_max_range = y_min, y_max * 1.15

            def clamp(v: float, lo: float, hi: float) -> float:
                return max(lo, min(v, hi))

            label_style = dict(showarrow=False, borderpad=3, borderwidth=1, opacity=0.98)
            fig_bubble.add_annotation(
                x=clamp(x_left, x_min_range, x_max_range),
                y=clamp(med_y + (y_max - med_y) * 0.72, y_min_range, y_max_range),
                text="테스트 확대",
                font=dict(size=11, color="#1E3A8A"),
                bgcolor="rgba(219,234,254,0.92)",
                bordercolor="#93C5FD",
                **label_style,
            )
            fig_bubble.add_annotation(
                x=clamp(x_right, x_min_range, x_max_range),
                y=clamp(med_y + (y_max - med_y) * 0.72, y_min_range, y_max_range),
                text="확대 후보 ★",
                font=dict(size=11, color="#065F46"),
                bgcolor="rgba(209,250,229,0.92)",
                bordercolor="#6EE7B7",
                **label_style,
            )
            fig_bubble.add_annotation(
                x=clamp(x_right, x_min_range, x_max_range),
                y=clamp(max(y_min, med_y * 0.35), y_min_range, y_max_range),
                text="개선 우선",
                font=dict(size=11, color="#9A3412"),
                bgcolor="rgba(255,237,213,0.93)",
                bordercolor="#FDBA74",
                **label_style,
            )
            fig_bubble.add_annotation(
                x=clamp(x_left, x_min_range, x_max_range),
                y=clamp(max(y_min, med_y * 0.35), y_min_range, y_max_range),
                text="집행 축소 검토",
                font=dict(size=11, color="#991B1B"),
                bgcolor="rgba(254,226,226,0.93)",
                bordercolor="#FCA5A5",
                **label_style,
            )
            apply_meta_theme(fig_bubble, "상품 포지셔닝 매트릭스 (X:노출, Y:CTR_total, 버블:클릭수)", 480)
            xv, xt = make_kor_ticks(float(plot_b["노출수"].max()))
            yv, yt = make_pct_ticks(float(plot_b["CTR_total"].max()) if not plot_b.empty else 0.01)
            fig_bubble.update_layout(showlegend=False)
            fig_bubble.update_xaxes(title="노출수", tickmode="array", tickvals=xv, ticktext=xt, range=[0, xv[-1]], automargin=True)
            fig_bubble.update_yaxes(title="CTR_total", tickmode="array", tickvals=yv, ticktext=yt, range=[0, yv[-1]], automargin=True)
            st.plotly_chart(fig_bubble, use_container_width=True)

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
            x=top_imp["상품축"],
            y=top_imp["점유율"],
            name="노출 점유율",
            marker_color="#2563EB",
            customdata=np.array([[n, fmt_kor_unit(i), fmt_pct(s)] for n, i, s in zip(top_imp["광고상품명_정리"], top_imp["노출수"], top_imp["점유율"])], dtype=object),
            hovertemplate="상품명: %{customdata[0]}<br>노출수: %{customdata[1]}<br>점유율: %{customdata[2]}<extra></extra>",
        )
        fig_conc.add_scatter(
            x=top_imp["상품축"],
            y=top_imp["누적점유율"],
            name="누적 점유율",
            mode="lines+markers+text",
            line=dict(color="#D97706", width=2),
            marker=dict(size=6),
            text=[fmt_pct(v) for v in top_imp["누적점유율"]],
            textposition="top center",
            yaxis="y2",
            hovertemplate="누적 점유율: %{text}<extra></extra>",
        )
        apply_meta_theme(fig_conc, "상위 상품 노출 점유율 및 누적 비중", 420)
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
        with st.expander("💡 운영 인사이트 펼치기", expanded=True):
            insight(f"확대 후보(고노출·고CTR): {', '.join(qh['광고상품명_정리'].head(3).tolist()) or '없음'}")
            insight(f"개선 우선(고노출·저CTR): {', '.join(ql['광고상품명_정리'].head(3).tolist()) or '없음'}", warn=True)
            insight(f"재테스트 후보(저노출·고CTR): {', '.join(lh['광고상품명_정리'].head(3).tolist()) or '없음'}")
            insight(f"집행 축소 검토(저노출·저CTR): {', '.join(ll['광고상품명_정리'].head(3).tolist()) or '없음'}", warn=True)

    with tab2:
        allp = sorted(df["광고상품명_정리"].dropna().unique().tolist())
        sel = st.selectbox("상품 선택", allp) if allp else None
        if sel:
            d2 = aggregate(df[df["광고상품명_정리"] == sel], ("프로모션명", "연월"))
            pp = aggregate(df[df["광고상품명_정리"] == sel], ("프로모션명",)).sort_values("노출수", ascending=False)
            pp["프로모션축"] = pp["프로모션명"].map(lambda x: ellipsis(x, 16))

            sec("프로모션별 노출 · 클릭 · CTR")
            st.plotly_chart(build_volume_click_ctr_chart(pp.head(20), "프로모션축", f"{sel} 프로모션 성과"), use_container_width=True)

            sec("상품 — 프로모션별 월별 노출·CTR 추이 (단순화)")
            top_promos = pp.head(6)["프로모션명"].tolist()
            md = d2[d2["프로모션명"].isin(top_promos)].copy().sort_values("연월")
            md["연월_str"] = md["연월"].dt.strftime("%Y-%m")
            month_order = md.sort_values("연월")["연월_str"].dropna().drop_duplicates().tolist()
            figm = go.Figure()
            for i, prm in enumerate(top_promos):
                sub = md[md["프로모션명"] == prm].sort_values("연월")
                figm.add_bar(
                    x=sub["연월_str"],
                    y=sub["노출수"],
                    name=ellipsis(prm, 16),
                    marker_color=COLORS[i % len(COLORS)],
                    customdata=np.array([[fmt_kor_unit(v1), fmt_pct(v2)] for v1, v2 in zip(sub["노출수"], sub["CTR_total"])], dtype=object),
                    hovertemplate="월: %{x}<br>노출수: %{customdata[0]}<br>CTR_total: %{customdata[1]}<extra></extra>",
                )
            apply_meta_theme(figm, "월별 노출 비교(CTR은 hover 보조)", 390)
            yv, yt = make_kor_ticks(float(md["노출수"].max()) if not md.empty else 0)
            figm.update_layout(barmode="group", yaxis=dict(tickmode="array", tickvals=yv, ticktext=yt, range=[0, yv[-1]]))
            apply_month_axis(figm, month_order, angle=-20)
            st.plotly_chart(figm, use_container_width=True)

        st.caption("※ 아래 차트는 선택 상품이 아닌 전체 데이터 기준(상위 프로모션) 월간 비교입니다.")
        sec("월별 프로모션 전체 성과 (분리 보기)")
        top_n_all = st.slider("전체 프로모션 상위 N(노출수 기준)", 3, 8, 5, key="t2_allpromo_n")
        all_pm = aggregate(df, ("연월", "프로모션명")).sort_values("연월")
        top_promos_all = (
            all_pm.groupby("프로모션명", as_index=False)["노출수"].sum().sort_values("노출수", ascending=False).head(top_n_all)["프로모션명"].tolist()
        )
        all_pm = all_pm[all_pm["프로모션명"].isin(top_promos_all)].copy()
        all_pm = all_pm.sort_values("연월")
        all_pm["연월_str"] = all_pm["연월"].dt.strftime("%Y-%m")
        month_all = all_pm["연월_str"].dropna().drop_duplicates().tolist()

        c_all_1, c_all_2 = st.columns(2)
        with c_all_1:
            sec("1) 월별 프로모션 전체 노출수")
            fig_imp = go.Figure()
            for i, prm in enumerate(top_promos_all):
                sub = all_pm[all_pm["프로모션명"] == prm].sort_values("연월")
                fig_imp.add_bar(
                    x=sub["연월_str"],
                    y=sub["노출수"],
                    name=ellipsis(prm, 14),
                    offsetgroup=f"imp_{i}",
                    marker_color=COLORS[i % len(COLORS)],
                    customdata=[fmt_kor_unit(v) for v in sub["노출수"]],
                    hovertemplate="월: %{x}<br>노출수: %{customdata}<extra></extra>",
                )
            apply_meta_theme(fig_imp, "전체 데이터 기준 월별 프로모션 노출수", 420)
            yv_imp, yt_imp = make_kor_ticks(float(all_pm["노출수"].max()) if not all_pm.empty else 0)
            fig_imp.update_layout(barmode="group", yaxis=dict(tickmode="array", tickvals=yv_imp, ticktext=yt_imp, range=[0, yv_imp[-1]]))
            apply_month_axis(fig_imp, month_all, angle=-18)
            st.plotly_chart(fig_imp, use_container_width=True)

        with c_all_2:
            sec("2) 월별 프로모션 전체 클릭수")
            fig_clk = go.Figure()
            for i, prm in enumerate(top_promos_all):
                sub = all_pm[all_pm["프로모션명"] == prm].sort_values("연월")
                fig_clk.add_bar(
                    x=sub["연월_str"],
                    y=sub["총클릭수"],
                    name=ellipsis(prm, 14),
                    offsetgroup=f"clk_{i}",
                    marker_color=COLORS[i % len(COLORS)],
                    customdata=[fmt_kor_unit(v) for v in sub["총클릭수"]],
                    hovertemplate="월: %{x}<br>클릭수: %{customdata}<extra></extra>",
                )
            apply_meta_theme(fig_clk, "전체 데이터 기준 월별 프로모션 클릭수", 420)
            yv_clk, yt_clk = make_kor_ticks(float(all_pm["총클릭수"].max()) if not all_pm.empty else 0)
            fig_clk.update_layout(barmode="group", yaxis=dict(tickmode="array", tickvals=yv_clk, ticktext=yt_clk, range=[0, yv_clk[-1]]))
            apply_month_axis(fig_clk, month_all, angle=-18)
            st.plotly_chart(fig_clk, use_container_width=True)

        sec("3) 월별 프로모션 전체 CTR_total")
        fig_ctr = go.Figure()
        for i, prm in enumerate(top_promos_all):
            sub = all_pm[all_pm["프로모션명"] == prm].sort_values("연월")
            fig_ctr.add_scatter(
                x=sub["연월_str"],
                y=sub["CTR_total"],
                name=ellipsis(prm, 14),
                mode="lines+markers",
                line=dict(width=2, color=COLORS[i % len(COLORS)]),
                marker=dict(size=6),
                text=[fmt_pct(v) for v in sub["CTR_total"]],
                hovertemplate="월: %{x}<br>CTR_total: %{text}<extra></extra>",
            )
        apply_meta_theme(fig_ctr, "전체 데이터 기준 월별 프로모션 CTR_total", 400)
        yv_ctr, yt_ctr = make_pct_ticks(float(all_pm["CTR_total"].max()) if not all_pm.empty else 0.01)
        fig_ctr.update_layout(yaxis=dict(tickmode="array", tickvals=yv_ctr, ticktext=yt_ctr, range=[0, yv_ctr[-1]]))
        apply_month_axis(fig_ctr, month_all, angle=-18)
        st.plotly_chart(fig_ctr, use_container_width=True)

    with tab3:
        unit = st.radio("시간 단위", ["월별", "일별"], horizontal=True)
        t = "연월" if unit == "월별" else "일자"
        td = aggregate(df, (t,)).sort_values(t)
        td["x"] = td[t].dt.strftime("%Y-%m" if unit == "월별" else "%Y-%m-%d")

        sec("전체 추이 — 노출수(좌) / 클릭수(우) + CTR 라벨")
        fig_td = build_volume_click_ctr_chart(td, "x", "기간별 핵심 추이")

        if unit == "월별":
            df_sorted = td.sort_values(t)
            month_order = df_sorted["x"].dropna().drop_duplicates().tolist()
            apply_month_axis(fig_td, month_order, angle=-20)

        if unit == "일별":
            daily = aggregate(df, ("일자",)).sort_values("일자")
            mu = daily["노출수"].mean()
            sd = daily["노출수"].std(ddof=0)
            daily["Z점수"] = (daily["노출수"] - mu) / sd if sd and sd > 0 else 0
            anomalies = daily[np.abs(daily["Z점수"]) > 2.0].copy()
            for _, r in anomalies.iterrows():
                color = "red" if r["Z점수"] > 0 else "green"
                fig_td.add_vline(x=r["일자"].strftime("%Y-%m-%d"), line=dict(color=color, dash="dash", width=1.5))

        st.plotly_chart(fig_td, use_container_width=True)

        # anomaly table
        daily = aggregate(df, ("일자",)).sort_values("일자")
        mu = daily["노출수"].mean()
        sd = daily["노출수"].std(ddof=0)
        daily["Z점수"] = (daily["노출수"] - mu) / sd if sd and sd > 0 else 0
        daily["판정"] = np.where(daily["Z점수"] > 2, "🔺 급등", np.where(daily["Z점수"] < -2, "🔻 급락", ""))
        anom = daily[daily["판정"] != ""][["일자", "노출수", "Z점수", "판정"]].copy()
        if not anom.empty:
            sec("이상치 탐지 (일별 노출수)")
            anom["일자"] = anom["일자"].dt.strftime("%Y-%m-%d")
            anom["노출수"] = anom["노출수"].map(fmt_table_number)
            anom["Z점수"] = anom["Z점수"].map(lambda v: f"{v:.2f}")
            st.dataframe(anom, hide_index=True, use_container_width=True)
            st.download_button("📥 CSV 다운로드", data=anom.to_csv(index=False, encoding="utf-8-sig"), file_name="anomaly_daily.csv", mime="text/csv")

        # 라이브 일정 뷰
        sec("프로모션 라이브 일정 (구분별 색상)")
        live_df = df.groupby("프로모션명", as_index=False).agg(
            라이브시작=("라이브일자", "min"),
            라이브종료=("일자", "max"),
            노출수=("노출수", "sum"),
            구분=("구분", lambda s: s.mode().iloc[0] if len(s.mode()) else "기타"),
        )
        live_df = live_df.sort_values("노출수", ascending=False).head(15)
        fig_live = go.Figure()
        group_colors = {g: COLORS[i % len(COLORS)] for i, g in enumerate(sorted(live_df["구분"].dropna().unique().tolist()))}
        for _, r in live_df.iterrows():
            if pd.isna(r["라이브시작"]) or pd.isna(r["라이브종료"]):
                continue
            fig_live.add_trace(
                go.Scatter(
                    x=[r["라이브시작"], r["라이브종료"]],
                    y=[r["프로모션명"], r["프로모션명"]],
                    mode="lines",
                    line=dict(width=12, color=group_colors.get(r["구분"], "#64748B")),
                    name=str(r["구분"]),
                    showlegend=False,
                    hovertemplate=f"프로모션: {r['프로모션명']}<br>시작: {r['라이브시작'].date()}<br>종료: {r['라이브종료'].date()}<extra></extra>",
                )
            )
        # separate legend entries
        for k, c in group_colors.items():
            fig_live.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=12, color=c), name=str(k)))
        apply_meta_theme(fig_live, "프로모션 라이브 일정 (구분별 색상)", 420)
        fig_live.update_yaxes(categoryorder="array", categoryarray=live_df["프로모션명"].tolist())
        st.plotly_chart(fig_live, use_container_width=True)

        # MoM table
        sec("월별 MoM 비교")
        mt = aggregate(df, ("연월",)).sort_values("연월")
        mt["연월_str"] = mt["연월"].dt.strftime("%Y-%m")
        mt["CTR"] = mt["CTR_total"]
        mt["전월대비_노출수_증감"] = mt["노출수"].diff()
        mt["전월대비_클릭수_증감"] = mt["총클릭수"].diff()
        mt["전월대비_CTR_변화(%p)"] = (mt["CTR"].diff() * 100)
        mom = mt[["연월_str", "노출수", "총클릭수", "CTR", "전월대비_노출수_증감", "전월대비_클릭수_증감", "전월대비_CTR_변화(%p)"]].copy()
        mom.columns = ["연월", "노출수", "클릭수", "CTR", "전월대비_노출수_증감", "전월대비_클릭수_증감", "전월대비_CTR_변화(%p)"]

        mom_show = mom.copy()
        mom_show["노출수"] = mom_show["노출수"].map(fmt_table_number)
        mom_show["클릭수"] = mom_show["클릭수"].map(fmt_table_number)
        mom_show["CTR"] = mom_show["CTR"].map(fmt_pct)
        mom_show["전월대비_노출수_증감"] = mom["전월대비_노출수_증감"].map(lambda v: "" if pd.isna(v) else f"{v:+,.0f}")
        mom_show["전월대비_클릭수_증감"] = mom["전월대비_클릭수_증감"].map(lambda v: "" if pd.isna(v) else f"{v:+,.0f}")
        mom_show["전월대비_CTR_변화(%p)"] = mom["전월대비_CTR_변화(%p)"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}%p")

        def color_delta(v):
            if isinstance(v, str) and v.startswith("+"):
                return "background-color:#DCFCE7"
            if isinstance(v, str) and v.startswith("-"):
                return "background-color:#FEE2E2"
            return ""

        st.dataframe(
            mom_show.style.applymap(color_delta, subset=["전월대비_노출수_증감", "전월대비_클릭수_증감", "전월대비_CTR_변화(%p)"]),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button("📥 CSV 다운로드", data=mom.to_csv(index=False, encoding="utf-8-sig"), file_name="mom_summary.csv", mime="text/csv")

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
            st.download_button("📥 CSV 다운로드", data=cmp.to_csv(index=False, encoding="utf-8-sig"), file_name="promotion_compare_products.csv", mime="text/csv")

            inc = cmp.sort_values("노출 증감", ascending=False).head(3)["광고상품명_정리"].tolist()
            dec = cmp.sort_values("노출 증감", ascending=True).head(3)["광고상품명_정리"].tolist()
            ctr_up = cmp.sort_values("CTR 변화(%p)", ascending=False).head(3)["광고상품명_정리"].tolist()
            insight(f"확대 후보: {', '.join(inc) if inc else '없음'}")
            insight(f"집행 축소 검토: {', '.join(dec) if dec else '없음'}", warn=True)
            insight(f"재테스트 후보(CTR 개선): {', '.join(ctr_up) if ctr_up else '없음'}")

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            day = aggregate(df, ("요일",)).copy()
            day["요일"] = pd.Categorical(day["요일"], categories=DAY_ORDER, ordered=True)
            day = day.sort_values("요일")
            sec("요일별 노출 · 클릭 · CTR")
            st.plotly_chart(build_volume_click_ctr_chart(day, "요일", "요일별 성과"), use_container_width=True)
        with c2:
            cat = aggregate(df, ("구분",)).sort_values("노출수", ascending=False).copy()
            cat["구분축"] = cat["구분"].map(lambda x: ellipsis(x, 12))
            sec("구분별 노출 · 클릭 · CTR")
            st.plotly_chart(build_volume_click_ctr_chart(cat, "구분축", "구분별 성과"), use_container_width=True)

        sec("상품 × 요일 히트맵 + TOP 조합")
        h = aggregate(df, ("광고상품명_정리", "요일"))
        order_y = (
            aggregate(df, ("광고상품명_정리",))
            .sort_values("노출수", ascending=False)["광고상품명_정리"]
            .tolist()
        )
        pv = h.pivot(index="광고상품명_정리", columns="요일", values="CTR_total").reindex(columns=DAY_ORDER)
        pv = pv.reindex(index=[i for i in order_y if i in pv.index])
        fig_h = go.Figure(
            go.Heatmap(
                z=pv.values,
                x=pv.columns,
                y=[ellipsis(v, 14) for v in pv.index],
                zmin=0,
                colorscale="Blues",
                text=np.vectorize(lambda z: "" if pd.isna(z) else f"{z:.2%}")(pv.values),
                texttemplate="%{text}",
                hovertemplate="요일: %{x}<br>상품: %{y}<br>CTR_total: %{z:.2%}<extra></extra>",
            )
        )
        apply_meta_theme(fig_h, "CTR 히트맵", max(320, len(pv) * 28 + 160))
        st.plotly_chart(fig_h, use_container_width=True)

        if "총클릭수" in h.columns:
            top_combo = h.sort_values(["CTR_total", "노출수"], ascending=[False, False]).head(10)[["광고상품명_정리", "요일", "노출수", "총클릭수", "CTR_total"]]
            top_combo["노출수"] = top_combo["노출수"].map(fmt_table_number)
            top_combo["총클릭수"] = top_combo["총클릭수"].map(fmt_table_number)
            top_combo["CTR_total"] = top_combo["CTR_total"].map(fmt_pct)
            st.dataframe(top_combo, hide_index=True, use_container_width=True)
            st.download_button("📥 CSV 다운로드", data=h.to_csv(index=False, encoding="utf-8-sig"), file_name="weekday_heatmap_detail.csv", mime="text/csv")

    with tab6:
        ind = compute_efficiency(aggregate(df, ("업종",)))
        ind["업종축"] = ind["업종"].map(lambda x: ellipsis(x, 10))
        sec("업종별 노출 · 클릭 · CTR")
        st.plotly_chart(build_volume_click_ctr_chart(ind.sort_values("노출수", ascending=False), "업종축", "업종별 성과"), use_container_width=True)

        sec("업종별 광고주 수 + 광고주 목록")
        adv = df.groupby("업종")["광고주"].agg(lambda s: sorted(set(map(str, s)))).reset_index(name="광고주목록")
        adv["광고주 수"] = adv["광고주목록"].map(len)
        adv["광고주 목록"] = adv["광고주목록"].map(lambda x: ", ".join(x[:12]) + (" …" if len(x) > 12 else ""))
        out = adv[["업종", "광고주 수", "광고주 목록"]].sort_values("광고주 수", ascending=False)
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("📥 CSV 다운로드", data=out.to_csv(index=False, encoding="utf-8-sig"), file_name="industry_advertisers.csv", mime="text/csv")

        st.subheader("광고주 상세 분석")
        adv_list = sorted(df["광고주"].dropna().unique().tolist())
        if adv_list:
            sel_adv = st.selectbox("광고주 선택", adv_list, key="adv_drilldown")
            df_adv = df[df["광고주"] == sel_adv]
            s_adv = summarize_period(df_adv)
            c_adv = st.columns(3)
            with c_adv[0]:
                kpi_card("노출수", fmt_kpi_compact(s_adv["노출수"]))
            with c_adv[1]:
                kpi_card("클릭수", fmt_kpi_compact(s_adv["클릭수"]))
            with c_adv[2]:
                kpi_card("CTR", fmt_pct(s_adv["CTR_total"]))

            adv_m = aggregate(df_adv, ("연월",)).sort_values("연월")
            adv_m["연월_str"] = adv_m["연월"].dt.strftime("%Y-%m")
            m_order = adv_m["연월_str"].dropna().drop_duplicates().tolist()
            fig_adv = go.Figure(go.Scatter(x=adv_m["연월_str"], y=adv_m["노출수"], mode="lines+markers", name="노출수", line=dict(color="#2563EB", width=2)))
            apply_meta_theme(fig_adv, f"{sel_adv} 월별 노출 추이", 340)
            yv_adv, yt_adv = make_kor_ticks(float(adv_m["노출수"].max()) if not adv_m.empty else 0)
            fig_adv.update_yaxes(tickmode="array", tickvals=yv_adv, ticktext=yt_adv, range=[0, yv_adv[-1]])
            apply_month_axis(fig_adv, m_order, angle=-18)
            st.plotly_chart(fig_adv, use_container_width=True)

            p_adv = aggregate(df_adv, ("광고상품명_정리",)).sort_values("노출수", ascending=False)
            p_adv_show = p_adv[["광고상품명_정리", "노출수", "총클릭수", "CTR_total"]].copy()
            p_adv_show["노출수"] = p_adv_show["노출수"].map(fmt_table_number)
            p_adv_show["총클릭수"] = p_adv_show["총클릭수"].map(fmt_table_number)
            p_adv_show["CTR_total"] = p_adv_show["CTR_total"].map(fmt_pct)
            st.dataframe(p_adv_show, use_container_width=True, hide_index=True)
            st.download_button("📥 CSV 다운로드", data=p_adv.to_csv(index=False, encoding="utf-8-sig"), file_name="advertiser_products.csv", mime="text/csv")

    with tab7:
        sec("Appendix ① 입력 데이터 구조")
        st.markdown(
            """
- 본 대시보드는 다음 11개 컬럼을 기준으로 동작합니다.
- `프로모션별, 캠페인명, 일자, 노출수, 클릭수, 라이브일자, 연도, 월, 광고상품명_정리, 구분, 광고주`
- 내부 계산을 위해 `총클릭수=클릭수`로 통일하여 사용합니다.
"""
        )

        sec("Appendix ② 기본 지표")
        st.markdown(
            """
- **노출수**: 광고가 노출된 횟수
- **클릭수**: 실제 클릭 횟수
- **총클릭수**: 본 데이터에서는 클릭수와 동일
"""
        )

        sec("Appendix ③ 비율 지표")
        st.markdown(
            """
- **CTR** = 클릭수 ÷ 노출수
- **CTR(전체)** = 총클릭수 ÷ 노출수 (본 데이터에서는 CTR과 실질 동일)
- VTR은 입력 데이터에 조회수 컬럼이 없어 본문 지표에서 자동 숨김 처리됩니다.
"""
        )

        sec("Appendix ④ 파생 지표")
        st.markdown(
            """
- **노출비중**: 특정 항목 노출수 / 전체 노출수
- **효율점수(Z-score)**: 집계 그룹 내 CTR_total 상대 위치
- **효율등급**: S/A/B/C
- **표본부족**: `min_imp` 미만 항목
"""
        )

        sec("Appendix ⑤ 전처리/집계 규칙")
        st.markdown(
            """
- 일자/라이브일자 datetime 파싱
- 연/월 기반 연월 생성
- 광고상품명 정규화:
  - 공백 제거 및 lower
  - 문자열 끝 괄호 suffix 제거
  - `_` suffix 상품은 별도 유지
- 구분/광고주 공백 제거
- 업종은 광고주 키워드 매칭(공백/대소문자 정규화)
"""
        )

        sec("Appendix ⑥ 비교 해석")
        st.markdown(
            """
- 프로모션 비교에서 CTR/VTR 변화는 **%p**를 우선 표시합니다.
- KPI 차이는 문장형으로 표시합니다.
  - 예) `B가 A보다 25.4만 큼`, `B가 A보다 0.18%p 높음`
"""
        )

        sec("Appendix ⑦ 집중도 지표")
        st.markdown(
            """
- Top3/Top5/Top10 누적 노출 비중
- HHI = 점유율 제곱합
- HHI는 0~10000 스케일도 함께 제공하여 집중도 해석을 돕습니다.
"""
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"✅ 필터 적용 결과\n{len(df):,}행 | {df['광고상품명_정리'].nunique()}개 상품 | {df['프로모션별'].nunique()}개 프로모션")


if __name__ == "__main__":
    main()
