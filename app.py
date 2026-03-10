"""
AD Performance Benchmark Dashboard
실행 방법:
  pip install -r requirements.txt
  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AD Performance Benchmark", page_icon="📊", layout="wide")

# 가독성 중심 라이트 스타일
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] { background:#f7f9fc; color:#1f2937; }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5e7eb; }
    .report-header { background:#ffffff; border:1px solid #dbe3ef; border-radius:12px; padding:20px 24px; margin-bottom:18px; }
    .report-header h1 { margin:0 0 6px 0; color:#111827; font-size:26px; }
    .report-header p { margin:0; color:#4b5563; font-size:13px; }
    .section-title { font-size:15px; font-weight:700; margin:18px 0 10px 0; color:#1f3b75; }
    .kpi-card { background:#fff; border:1px solid #dbe3ef; border-radius:10px; padding:14px; text-align:center; }
    .kpi-label { color:#6b7280; font-size:12px; margin-bottom:6px; }
    .kpi-val { color:#111827; font-size:22px; font-weight:700; }
    .kpi-sub { color:#4b5563; font-size:12px; margin-top:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "프로모션별", "캠페인명", "일자", "노출수", "클릭수", "총클릭수", "연도", "월", "광고상품명_정리", "구분", "광고주"
]
NUMERIC_COLUMNS = ["노출수", "클릭수", "총클릭수", "연도", "월"]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


@dataclass
class Filters:
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    years: List[int]
    months: List[int]
    weekdays: List[str]
    categories: List[str]
    products: List[str]
    promotions: List[str]
    campaigns: List[str]


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_csv_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def map_industry(advertiser: str) -> str:
    s = str(advertiser)
    if any(k in s for k in ["패션", "언더웨어", "아미", "캘빈"]):
        return "패션"
    if any(k in s for k in ["과자", "식품", "푸드", "마켓"]):
        return "식품/커머스"
    if any(k in s for k in ["게임", "e스포츠"]):
        return "게임"
    if any(k in s for k in ["뷰티", "코스메", "화장"]):
        return "뷰티"
    return "기타"


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    out["일자"] = pd.to_datetime(out["일자"], errors="coerce")
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["연도"] = out["연도"].fillna(out["일자"].dt.year)
    out["월"] = out["월"].fillna(out["일자"].dt.month)
    out["연월"] = pd.to_datetime(dict(year=out["연도"].astype("Int64"), month=out["월"].astype("Int64"), day=1), errors="coerce")
    out["요일"] = out["일자"].dt.weekday.map(DAY_MAP)

    out["프로모션명"] = out["프로모션별"].astype("string").fillna("미분류")
    out["캠페인명"] = out["캠페인명"].astype("string").fillna("미분류")
    out["광고상품명_정리"] = out["광고상품명_정리"].astype("string").fillna("미분류")
    out["구분"] = out["구분"].astype("string").fillna("미분류")
    out["광고주"] = out["광고주"].astype("string").fillna("미분류")
    out["광고주업종"] = out["광고주"].map(map_industry)

    out["총클릭수"] = out["총클릭수"].fillna(out["클릭수"]).fillna(0)
    out["노출수"] = out["노출수"].fillna(0)
    out["클릭수"] = out["클릭수"].fillna(0)

    return out


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
    if f.promotions:
        out = out[out["프로모션명"].isin(f.promotions)]
    if f.campaigns:
        out = out[out["캠페인명"].isin(f.campaigns)]
    return out


@st.cache_data(show_spinner=False)
def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols, as_index=False)[["노출수", "클릭수", "총클릭수"]].sum(min_count=1).fillna(0)
    agg["CTR"] = np.where(agg["노출수"] > 0, agg["클릭수"] / agg["노출수"], np.nan)
    agg["CTR_total"] = np.where(agg["노출수"] > 0, agg["총클릭수"] / agg["노출수"], np.nan)
    total_imp = agg["노출수"].sum()
    agg["노출비중"] = np.where(total_imp > 0, agg["노출수"] / total_imp, np.nan)

    # 효율점수: CTR_total z + ClickShare z + (Inverse ImpressionShare) z
    click_share = np.where(agg["총클릭수"].sum() > 0, agg["총클릭수"] / agg["총클릭수"].sum(), 0)
    inv_imp = 1 - agg["노출비중"].fillna(0)

    def z(v: pd.Series) -> pd.Series:
        std = v.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=v.index)
        return (v - v.mean()) / std

    score = 0.5 * z(agg["CTR_total"].fillna(0)) + 0.3 * z(pd.Series(click_share, index=agg.index)) + 0.2 * z(inv_imp)
    agg["효율점수"] = score.round(3)
    agg["등급"] = pd.cut(agg["효율점수"], bins=[-np.inf, -0.3, 0.3, 1.0, np.inf], labels=["C", "B", "A", "S"])
    agg["등급"] = agg["등급"].astype("string").fillna("B")
    return agg


def metric_fmt(col: str):
    if col in ["CTR", "CTR_total", "노출비중"]:
        return ".2%"
    return ",.0f"


def main():
    with st.sidebar:
        st.subheader("데이터 로드")
        upload = st.file_uploader("CSV 업로드", type=["csv"])
        path = st.text_input("또는 CSV 경로")

    if upload is None and not path:
        st.markdown('<div class="report-header"><h1>AD Performance Dashboard</h1><p>CSV를 업로드하면 노출/클릭 중심 리포트를 생성합니다.</p></div>', unsafe_allow_html=True)
        st.stop()

    raw = load_csv(upload.getvalue()) if upload else load_csv_path(path)
    df_raw = preprocess(raw)

    with st.sidebar:
        st.subheader("Global Filters")
        min_d, max_d = df_raw["일자"].min(), df_raw["일자"].max()
        dr = st.date_input("날짜", value=(min_d.date(), max_d.date()) if pd.notna(min_d) and pd.notna(max_d) else ())

        years = sorted(df_raw["연도"].dropna().astype(int).unique().tolist())
        months = sorted(df_raw["월"].dropna().astype(int).unique().tolist())
        weekdays = [d for d in DAY_ORDER if d in set(df_raw["요일"].dropna().unique())]
        categories = sorted(df_raw["구분"].dropna().unique().tolist())
        products = sorted(df_raw["광고상품명_정리"].dropna().unique().tolist())
        promotions = sorted(df_raw["프로모션명"].dropna().unique().tolist())
        campaigns = sorted(df_raw["캠페인명"].dropna().unique().tolist())

        f = Filters(
            pd.to_datetime(dr[0]) if dr and len(dr) > 0 else None,
            pd.to_datetime(dr[1]) if dr and len(dr) > 1 else None,
            st.multiselect("연도", years, default=years),
            st.multiselect("월", months, default=months),
            st.multiselect("요일", weekdays, default=weekdays),
            st.multiselect("구분", categories, default=categories),
            st.multiselect("광고상품", products, default=products),
            st.multiselect("프로모션", promotions, default=promotions[:20]),
            st.multiselect("캠페인", campaigns, default=campaigns[:20]),
        )

    df = apply_filters(df_raw, f)

    st.markdown(
        f'<div class="report-header"><h1>광고 효율 벤치마크</h1><p>필터 적용 {len(df):,}건 / 전체 {len(df_raw):,}건 · 광고상품 {df["광고상품명_정리"].nunique()}개 · 프로모션 {df["프로모션명"].nunique()}개</p></div>',
        unsafe_allow_html=True,
    )

    total_imp, total_clk, total_tclk = df["노출수"].sum(), df["클릭수"].sum(), df["총클릭수"].sum()
    ctr = total_clk / total_imp if total_imp else np.nan
    ctr_t = total_tclk / total_imp if total_imp else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    for c, label, val, sub in [
        (k1, "노출수", f"{total_imp:,.0f}", ""),
        (k2, "클릭수", f"{total_clk:,.0f}", ""),
        (k3, "총클릭수", f"{total_tclk:,.0f}", ""),
        (k4, "CTR", f"{ctr:.2%}" if pd.notna(ctr) else "N/A", "클릭수/노출수"),
        (k5, "CTR(전체)", f"{ctr_t:.2%}" if pd.notna(ctr_t) else "N/A", "총클릭수/노출수"),
    ]:
        with c:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["상품 분석", "프로모션 분석", "상품×프로모션", "시계열", "요일/구분 + Appendix"])

    with tab1:
        st.markdown('<div class="section-title">상품 성과 (노출/클릭/CTR)</div>', unsafe_allow_html=True)
        p = aggregate(df, ["광고상품명_정리"]).sort_values("총클릭수", ascending=False)
        st.dataframe(p[["광고상품명_정리", "노출수", "클릭수", "총클릭수", "CTR", "CTR_total", "노출비중", "효율점수", "등급"]], use_container_width=True)

        fig = go.Figure()
        top = p.head(20)
        fig.add_bar(x=top["광고상품명_정리"], y=top["노출수"], name="노출수", marker_color="#93c5fd", yaxis="y")
        fig.add_scatter(x=top["광고상품명_정리"], y=top["총클릭수"], name="총클릭수", mode="lines+markers", marker_color="#2563eb", yaxis="y2")
        fig.update_layout(template="plotly_white", xaxis_tickangle=-25, yaxis=dict(title="노출수"), yaxis2=dict(title="총클릭수", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">프로모션 성과 (노출/클릭/CTR)</div>', unsafe_allow_html=True)
        r = aggregate(df, ["프로모션명", "구분"]).sort_values(["프로모션명", "총클릭수"], ascending=[True, False])
        st.dataframe(r[["프로모션명", "구분", "노출수", "클릭수", "총클릭수", "CTR_total", "노출비중", "효율점수", "등급"]], use_container_width=True, height=420)

        st.download_button("프로모션 리포트 다운로드", data=r.to_csv(index=False).encode("utf-8-sig"), file_name="promotion_report.csv", mime="text/csv")

    with tab3:
        st.markdown('<div class="section-title">같은 상품, 다른 프로모션 비교</div>', unsafe_allow_html=True)
        product_opts = sorted(df["광고상품명_정리"].unique().tolist())
        sel_product = st.selectbox("광고상품 선택", product_opts)
        pp = df[df["광고상품명_정리"] == sel_product]
        pp_agg = aggregate(pp, ["프로모션명", "구분"]).sort_values("총클릭수", ascending=False)
        st.dataframe(pp_agg[["프로모션명", "구분", "노출수", "총클릭수", "CTR_total", "노출비중", "효율점수", "등급"]], use_container_width=True)

        fig = go.Figure()
        fig.add_bar(x=pp_agg["프로모션명"], y=pp_agg["노출수"], name="노출수", marker_color="#bfdbfe")
        fig.add_bar(x=pp_agg["프로모션명"], y=pp_agg["총클릭수"], name="총클릭수", marker_color="#1d4ed8")
        fig.add_scatter(x=pp_agg["프로모션명"], y=pp_agg["CTR_total"], name="CTR(전체)", mode="lines+markers", yaxis="y2", marker_color="#059669")
        fig.update_layout(template="plotly_white", barmode="group", xaxis_tickangle=-25, yaxis=dict(title="노출/클릭"), yaxis2=dict(title="CTR", overlaying="y", side="right", tickformat=".2%"))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-title">시계열 트렌드 (지표 2개 동시 보기)</div>', unsafe_allow_html=True)
        metric_opts = ["노출수", "클릭수", "총클릭수", "CTR", "CTR_total"]
        m1, m2, unit = st.columns([2, 2, 1])
        m1_sel = m1.selectbox("지표 1", metric_opts, index=2)
        m2_sel = m2.selectbox("지표 2", metric_opts, index=0)
        unit_sel = unit.radio("단위", ["일별", "월별"])

        tcol = "일자" if unit_sel == "일별" else "연월"
        t = aggregate(df, [tcol]).sort_values(tcol)

        fig = go.Figure()
        fig.add_scatter(x=t[tcol], y=t[m1_sel], mode="lines+markers", name=m1_sel, yaxis="y", marker_color="#2563eb")
        fig.add_scatter(x=t[tcol], y=t[m2_sel], mode="lines+markers", name=m2_sel, yaxis="y2", marker_color="#059669")

        y2_pct = m2_sel in ["CTR", "CTR_total"]
        y1_pct = m1_sel in ["CTR", "CTR_total"]
        fig.update_layout(
            template="plotly_white",
            yaxis=dict(title=m1_sel, tickformat=".2%" if y1_pct else ",.0f"),
            yaxis2=dict(title=m2_sel, overlaying="y", side="right", tickformat=".2%" if y2_pct else ",.0f"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">요일별 성과</div>', unsafe_allow_html=True)
            d = aggregate(df, ["요일"])
            d["요일"] = pd.Categorical(d["요일"], categories=DAY_ORDER, ordered=True)
            d = d.sort_values("요일")
            fig = go.Figure()
            fig.add_bar(x=d["요일"], y=d["노출수"], name="노출수", marker_color="#93c5fd")
            fig.add_scatter(x=d["요일"], y=d["총클릭수"], name="총클릭수", mode="lines+markers", yaxis="y2", marker_color="#1d4ed8")
            fig.update_layout(template="plotly_white", yaxis2=dict(title="총클릭수", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">구분별 + 광고주업종(대분류) 성과</div>', unsafe_allow_html=True)
            g = aggregate(df, ["구분"])
            i = aggregate(df, ["광고주업종"])
            fig1 = go.Figure([go.Bar(x=g["구분"], y=g["총클릭수"], marker_color="#2563eb")])
            fig1.update_layout(template="plotly_white", title="구분별 총클릭수")
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = go.Figure([go.Bar(x=i["광고주업종"], y=i["총클릭수"], marker_color="#059669")])
            fig2.update_layout(template="plotly_white", title="광고주 업종(대분류)별 총클릭수")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Appendix: 계산 방식</div>', unsafe_allow_html=True)
        st.markdown(
            """
- **CTR** = 클릭수 / 노출수  
- **CTR(전체)** = 총클릭수 / 노출수  
- **노출비중** = 해당 그룹 노출수 / 전체 노출수  
- **효율점수** = `0.5*z(CTR_total) + 0.3*z(클릭점유율) + 0.2*z(1-노출비중)`  
  - 클릭점유율 = 해당 그룹 총클릭수 / 전체 총클릭수  
  - z(x) = (x - 평균) / 표준편차  
- **등급**: 효율점수 기준 `S(>=1.0) / A(>=0.3) / B(>=-0.3) / C(<-0.3)`
            """
        )

    with st.sidebar:
        st.divider()
        st.download_button("필터링 데이터 다운로드", data=df.to_csv(index=False).encode("utf-8-sig"), file_name="filtered_data.csv", mime="text/csv")


if __name__ == "__main__":
    main()
