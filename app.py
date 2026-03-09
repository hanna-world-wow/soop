"""
Streamlit 광고 효율 대시보드
실행 방법:
1) 의존성 설치: pip install -r requirements.txt
2) 실행: streamlit run app.py
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

# 1. 페이지 설정 및 테마 정의
st.set_page_config(page_title="AD Performance Insights", layout="wide", initial_sidebar_state="expanded")

# GA 스타일의 CSS 주입
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #1a73e8; }
    [data-testid="stMetricDelta"] { font-size: 16px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8f0fe !important;
        color: #1a73e8 !important;
        border-bottom: 2px solid #1a73e8 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
GA_BLUE = "#1a73e8"
SECTION_NAMES = ["KPI 카드", "상품별 집계 테이블", "TOP/WORST 랭킹", "월별 추이", "요일별 분석", "구분별 비교", "프로모션 보고서"]


@dataclass
class FilterValues:
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    years: List[int]
    months: List[int]
    weekdays: List[str]
    categories: List[str]
    products: List[str]
    campaigns: List[str]

    def to_summary(self) -> Dict[str, str]:
        return {
            "일자": f"{self.date_min.date() if self.date_min is not None else '전체'} ~ {self.date_max.date() if self.date_max is not None else '전체'}",
            "연도": ", ".join(map(str, self.years)) if self.years else "전체",
            "월": ", ".join(map(str, self.months)) if self.months else "전체",
            "요일": ", ".join(self.weekdays) if self.weekdays else "전체",
            "구분": ", ".join(self.categories) if self.categories else "전체",
            "광고상품명_정리": ", ".join(self.products[:10]) + (" ..." if len(self.products) > 10 else "") if self.products else "전체",
            "캠페인명": ", ".join(self.campaigns[:10]) + (" ..." if len(self.campaigns) > 10 else "") if self.campaigns else "전체",
        }


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
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
    out["연월"] = pd.to_datetime(dict(year=out["연도"].astype("Int64"), month=out["월"].astype("Int64"), day=1), errors="coerce")

    for col in ["캠페인명", "광고상품명_정리", "구분"]:
        out[col] = out[col].astype("string").fillna("미분류")

    if "프로모션별" in out.columns:
        out["프로모션명"] = out["프로모션별"].astype("string").fillna("미분류")
    else:
        out["프로모션명"] = out["캠페인명"].astype("string").fillna("미분류")
    out["광고주"] = out["광고주"].astype("string").fillna("미분류") if "광고주" in out.columns else "미분류"

    # 총클릭수가 없으면 클릭수 기반으로 보정
    out["총클릭수"] = out["총클릭수"].fillna(out["클릭수"]).fillna(0)
    out["클릭수"] = out["클릭수"].fillna(0)
    out["노출수"] = out["노출수"].fillna(0)

    return out


def make_filter_values(date_range, years, months, weekdays, categories, products, campaigns) -> FilterValues:
    date_min = pd.to_datetime(date_range[0]) if date_range and len(date_range) > 0 else None
    date_max = pd.to_datetime(date_range[1]) if date_range and len(date_range) > 1 else None
    return FilterValues(date_min, date_max, years, months, weekdays, categories, products, campaigns)


@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, f: FilterValues) -> pd.DataFrame:
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
    if f.campaigns:
        out = out[out["캠페인명"].isin(f.campaigns)]
    return out


@st.cache_data(show_spinner=False)
def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False, as_index=False)[["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "광고비", "동영상조회수"]]
        .sum(min_count=1)
        .fillna(0)
    )
    agg["CTR_recalc"] = safe_div(agg["클릭수"], agg["노출수"])
    agg["CTR_total_recalc"] = safe_div(agg["총클릭수"], agg["노출수"])
    agg["CPC_recalc"] = safe_div(agg["광고비"], agg["클릭수"])
    agg["CPC_total_recalc"] = safe_div(agg["광고비"], agg["총클릭수"])
    agg["eCPM_recalc"] = safe_div(agg["광고비"], agg["노출수"]) * 1000
    agg["VTR_recalc"] = safe_div(agg["동영상조회수"], agg["노출수"])
    agg["CPV_recalc"] = safe_div(agg["광고비"], agg["동영상조회수"])
    return agg


def filter_badge(status: str):
    color = {"Global 적용": "#1f77b4", "Local 적용": "#ff7f0e", "Unfiltered": "#7f7f7f"}[status]
    st.markdown(
        f"<span style='background:{color}; color:white; padding:2px 10px; border-radius:999px; font-size:12px;'>{status}</span>",
        unsafe_allow_html=True,
    )


def render_filter_summary(f: FilterValues):
    with st.expander("사용 중인 필터 조건", expanded=False):
        for k, v in f.to_summary().items():
            st.caption(f"- {k}: {v}")


def local_filter_ui(df: pd.DataFrame, key_prefix: str):
    use_global = st.checkbox("Use global filters", value=True, key=f"use_global_{key_prefix}")
    if use_global:
        return True, make_filter_values(None, [], [], [], [], [], [])

    min_date, max_date = df["일자"].min(), df["일자"].max()
    local_date = st.date_input(
        "일자 범위(로컬)",
        value=(min_date.date(), max_date.date()) if pd.notna(min_date) and pd.notna(max_date) else (),
        key=f"local_date_{key_prefix}",
    )

    years = sorted(df["연도"].dropna().astype(int).unique().tolist())
    months = sorted(df["월"].dropna().astype(int).unique().tolist())
    weekdays = [d for d in DAY_ORDER if d in set(df["요일"].dropna().unique())]
    categories = sorted(df["구분"].dropna().unique().tolist())
    products = sorted(df["광고상품명_정리"].dropna().unique().tolist())
    campaigns = sorted(df["캠페인명"].dropna().unique().tolist())

    l_years = st.multiselect("연도(로컬)", years, default=years, key=f"local_years_{key_prefix}")
    l_months = st.multiselect("월(로컬)", months, default=months, key=f"local_months_{key_prefix}")
    l_weekdays = st.multiselect("요일(로컬)", weekdays, default=weekdays, key=f"local_weekdays_{key_prefix}")
    l_categories = st.multiselect("구분(로컬)", categories, default=categories, key=f"local_categories_{key_prefix}")
    l_products = st.multiselect("광고상품명_정리(로컬)", products, default=products, key=f"local_products_{key_prefix}")
    search = st.text_input("캠페인명 검색(로컬)", key=f"local_campaign_search_{key_prefix}")
    campaign_options = [c for c in campaigns if search.lower() in c.lower()]
    l_campaigns = st.multiselect("캠페인명(로컬)", campaign_options, default=campaign_options, key=f"local_campaigns_{key_prefix}")

    return False, make_filter_values(local_date, l_years, l_months, l_weekdays, l_categories, l_products, l_campaigns)


def section_state(section_name: str, scope_map: Dict[str, bool], use_global: bool) -> str:
    if not use_global:
        return "Local 적용"
    return "Global 적용" if scope_map.get(section_name, False) else "Unfiltered"


def main():
    st.sidebar.title("📊 Filter Engine")
    upload = st.sidebar.file_uploader("CSV Data Upload", type=["csv"])
    path = st.sidebar.text_input("또는 CSV 경로 입력")

    if upload is None and not path:
        st.info("💡 왼측 사이드바에서 광고 데이터(CSV)를 업로드하거나 경로를 입력해주세요.")
        st.stop()

    try:
        df_loaded = load_csv_from_bytes(upload.getvalue()) if upload is not None else load_csv_from_path(path)
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        st.stop()

    df_raw = preprocess_df(df_loaded)

    # 전역 필터
    with st.sidebar:
        st.divider()
        min_date, max_date = df_raw["일자"].min(), df_raw["일자"].max()
        g_date = st.date_input("날짜 범위", value=(min_date.date(), max_date.date()) if pd.notna(min_date) and pd.notna(max_date) else ())

        years = sorted(df_raw["연도"].dropna().astype(int).unique().tolist())
        months = sorted(df_raw["월"].dropna().astype(int).unique().tolist())
        weekdays = [d for d in DAY_ORDER if d in set(df_raw["요일"].dropna().unique())]
        categories = sorted(df_raw["구분"].dropna().unique().tolist())
        products = sorted(df_raw["광고상품명_정리"].dropna().unique().tolist())
        campaigns = sorted(df_raw["캠페인명"].dropna().unique().tolist())

        g_years = st.multiselect("연도", years, default=years)
        g_months = st.multiselect("월", months, default=months)
        g_weekdays = st.multiselect("요일", weekdays, default=weekdays)
        g_categories = st.multiselect("구분", categories, default=categories)
        g_products = st.multiselect("광고상품명_정리", products, default=products)
        camp_search = st.text_input("캠페인명 검색")
        camp_options = [c for c in campaigns if camp_search.lower() in c.lower()]
        g_campaigns = st.multiselect("캠페인명", camp_options, default=camp_options)

        st.subheader("FILTER SCOPE")
        scope_map = {s: st.checkbox(s, value=True, key=f"scope_{s}") for s in SECTION_NAMES}

    global_filters = make_filter_values(g_date, g_years, g_months, g_weekdays, g_categories, g_products, g_campaigns)
    df_global_filtered = apply_filters(df_raw, global_filters)

    def get_section_df(section_name: str, local_filters: FilterValues, use_global: bool) -> pd.DataFrame:
        base = df_global_filtered if scope_map.get(section_name, False) else df_raw
        if use_global:
            return base
        return apply_filters(base, local_filters)

    st.title("광고 성과 분석 리포트")
    st.caption(f"원천 {len(df_raw):,}건 | 전역 필터 적용 {len(df_global_filtered):,}건")

    # KPI 카드 섹션
    c1, c2 = st.columns([7, 1])
    with c1:
        st.subheader("A. KPI 카드")
    with c2:
        filter_badge("Global 적용" if scope_map["KPI 카드"] else "Unfiltered")

    use_global_kpi, local_kpi = local_filter_ui(df_raw, "kpi")
    df_kpi = get_section_df("KPI 카드", local_kpi, use_global_kpi)
    render_filter_summary(global_filters if use_global_kpi and scope_map["KPI 카드"] else local_kpi)

    kpi_agg = aggregate_metrics(df_kpi, ["광고상품명_정리"])
    total_imps = kpi_agg["노출수"].sum()
    total_clicks = kpi_agg["총클릭수"].sum()
    total_spend = kpi_agg["광고비"].sum()
    ctr_total = total_clicks / total_imps if total_imps else np.nan
    cpc_total = total_spend / total_clicks if total_clicks else np.nan
    ecpm_total = total_spend / total_imps * 1000 if total_imps else np.nan

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("노출수", f"{total_imps:,.0f}")
    m2.metric("클릭수", f"{kpi_agg['클릭수'].sum():,.0f}")
    m3.metric("총클릭수", f"{total_clicks:,.0f}")
    m4.metric("CTR_total_recalc", f"{ctr_total:.2%}" if pd.notna(ctr_total) else "N/A")
    m5.metric("광고비", f"₩{total_spend:,.0f}")
    m6.metric("CPC_total_recalc", f"₩{cpc_total:,.0f}" if pd.notna(cpc_total) else "N/A")
    m7.metric("eCPM_recalc", f"₩{ecpm_total:,.0f}" if pd.notna(ecpm_total) else "N/A")

    st.write("---")

    tab1, tab2, tab3 = st.tabs(["📈 시계열 추이", "📦 상품/캠페인 분석", "📅 요일/구분별 특징"])

    with tab1:
        st.subheader("D. 월별/일별 성과 추이")
        use_global_month, local_month = local_filter_ui(df_raw, "monthly")
        status = section_state("월별 추이", scope_map, use_global_month)
        filter_badge(status)
        df_month = get_section_df("월별 추이", local_month, use_global_month)
        render_filter_summary(global_filters if status == "Global 적용" else local_month)

        products_sel = st.multiselect("광고상품 선택", sorted(df_month["광고상품명_정리"].unique().tolist()), default=sorted(df_month["광고상품명_정리"].unique().tolist())[:5])
        time_unit = st.radio("시간 단위", ["일별", "월별"], horizontal=True)
        t_col = "일자" if time_unit == "일별" else "연월"
        target = df_month[df_month["광고상품명_정리"].isin(products_sel)] if products_sel else df_month
        trend = aggregate_metrics(target, [t_col, "광고상품명_정리"]).sort_values(t_col)

        for metric in ["CTR_total_recalc", "CPC_total_recalc", "노출수"]:
            fig = px.line(trend, x=t_col, y=metric, color="광고상품명_정리", template="plotly_white")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_left, col_right = st.columns([6, 4])
        with col_left:
            st.subheader("B. 광고상품 성과 테이블")
            use_global_tbl, local_tbl = local_filter_ui(df_raw, "table")
            status = section_state("상품별 집계 테이블", scope_map, use_global_tbl)
            filter_badge(status)
            df_tbl = get_section_df("상품별 집계 테이블", local_tbl, use_global_tbl)

            min_imps = st.number_input("최소 노출 기준", min_value=0, value=10000, step=1000)
            sort_col = st.selectbox("정렬 기준", ["CTR_total_recalc", "CPC_total_recalc", "eCPM_recalc", "광고비"])
            asc = sort_col == "CPC_total_recalc"

            prod_table = aggregate_metrics(df_tbl, ["광고상품명_정리"])
            prod_table = prod_table[prod_table["노출수"] >= min_imps].sort_values(sort_col, ascending=asc)
            show_cols = ["광고상품명_정리", "노출수", "총클릭수", "CTR_total_recalc", "광고비", "CPC_total_recalc", "eCPM_recalc"]
            st.dataframe(prod_table[show_cols], use_container_width=True, height=420)

            st.download_button(
                "📥 상품별 집계 다운로드",
                data=prod_table.to_csv(index=False).encode("utf-8-sig"),
                file_name="product_agg.csv",
                mime="text/csv",
            )

        with col_right:
            st.subheader("C. TOP/WORST + Efficiency Score")
            use_global_rank, local_rank = local_filter_ui(df_raw, "rank")
            status = section_state("TOP/WORST 랭킹", scope_map, use_global_rank)
            filter_badge(status)
            df_rank = get_section_df("TOP/WORST 랭킹", local_rank, use_global_rank)
            rank_df = aggregate_metrics(df_rank, ["광고상품명_정리"])
            rank_df = rank_df[rank_df["노출수"] >= min_imps]

            top_ctr = rank_df.nlargest(5, "CTR_total_recalc")[["광고상품명_정리", "CTR_total_recalc"]]
            low_cpc = rank_df[(rank_df["광고비"] > 0) & (rank_df["총클릭수"] > 0)].nsmallest(5, "CPC_total_recalc")[["광고상품명_정리", "CPC_total_recalc"]]
            st.markdown("**CTR_total_recalc TOP5**")
            st.dataframe(top_ctr, use_container_width=True)
            st.markdown("**CPC_total_recalc LOW5**")
            st.dataframe(low_cpc, use_container_width=True)

            score_df = rank_df[(rank_df["CTR_total_recalc"].notna()) & (rank_df["CPC_total_recalc"].notna())].copy()
            if len(score_df) >= 2 and score_df["CTR_total_recalc"].std(ddof=0) > 0 and score_df["CPC_total_recalc"].std(ddof=0) > 0:
                score_df["score"] = (
                    (score_df["CTR_total_recalc"] - score_df["CTR_total_recalc"].mean()) / score_df["CTR_total_recalc"].std(ddof=0)
                    - (score_df["CPC_total_recalc"] - score_df["CPC_total_recalc"].mean()) / score_df["CPC_total_recalc"].std(ddof=0)
                )
                st.markdown("**Score TOP5 / WORST5**")
                s1, s2 = st.columns(2)
                s1.dataframe(score_df.nlargest(5, "score")[["광고상품명_정리", "score"]], use_container_width=True)
                s2.dataframe(score_df.nsmallest(5, "score")[["광고상품명_정리", "score"]], use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("E. 요일별 CTR_total_recalc")
            use_global_day, local_day = local_filter_ui(df_raw, "weekday")
            status = section_state("요일별 분석", scope_map, use_global_day)
            filter_badge(status)
            df_day = get_section_df("요일별 분석", local_day, use_global_day)
            day_agg = aggregate_metrics(df_day, ["요일"])
            day_agg["요일"] = pd.Categorical(day_agg["요일"], categories=DAY_ORDER, ordered=True)
            day_agg = day_agg.sort_values("요일")
            fig_day = px.bar(day_agg, x="요일", y="CTR_total_recalc", color_discrete_sequence=[GA_BLUE], template="plotly_white")
            st.plotly_chart(fig_day, use_container_width=True)

            heat_agg = aggregate_metrics(df_day, ["광고상품명_정리", "요일"])
            heat_agg = heat_agg[heat_agg["노출수"] >= 10000]
            if not heat_agg.empty:
                heat = heat_agg.pivot(index="광고상품명_정리", columns="요일", values="CTR_total_recalc").reindex(columns=DAY_ORDER)
                fig_heat = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Blues"))
                fig_heat.update_layout(title="상품×요일 CTR_total_recalc Heatmap", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)

        with c2:
            st.subheader("F. 구분(Category)별 비교")
            use_global_cat, local_cat = local_filter_ui(df_raw, "category")
            status = section_state("구분별 비교", scope_map, use_global_cat)
            filter_badge(status)
            df_cat = get_section_df("구분별 비교", local_cat, use_global_cat)
            cat_agg = aggregate_metrics(df_cat, ["구분"])

            for y in ["CTR_total_recalc", "CPC_total_recalc", "eCPM_recalc"]:
                fig = px.bar(cat_agg, x="구분", y=y, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("G. 프로모션/구분별 광고 효율 보고서")
        use_global_promo, local_promo = local_filter_ui(df_raw, "promotion")
        status = section_state("프로모션 보고서", scope_map, use_global_promo)
        filter_badge(status)
        df_promo = get_section_df("프로모션 보고서", local_promo, use_global_promo)
        render_filter_summary(global_filters if status == "Global 적용" else local_promo)

        promo_opts = sorted(df_promo["프로모션명"].dropna().unique().tolist())
        sel_promotions = st.multiselect("프로모션명 선택", promo_opts, default=promo_opts)
        promo_view = df_promo[df_promo["프로모션명"].isin(sel_promotions)] if sel_promotions else df_promo

        promo_agg = aggregate_metrics(promo_view, ["프로모션명", "구분"]) [["프로모션명", "구분", "노출수", "클릭수", "CTR_recalc"]]
        promo_agg = promo_agg.sort_values(["프로모션명", "CTR_recalc"], ascending=[True, False])
        st.dataframe(promo_agg, use_container_width=True, height=380)

        p1, p2 = st.columns(2)
        with p1:
            fig_promo_ctr = px.bar(
                promo_agg,
                x="프로모션명",
                y="CTR_recalc",
                color="구분",
                barmode="group",
                template="plotly_white",
                title="프로모션/구분별 CTR",
            )
            st.plotly_chart(fig_promo_ctr, use_container_width=True)
        with p2:
            fig_promo_click = px.bar(
                promo_agg,
                x="프로모션명",
                y="클릭수",
                color="구분",
                barmode="group",
                template="plotly_white",
                title="프로모션/구분별 클릭수",
            )
            st.plotly_chart(fig_promo_click, use_container_width=True)

        st.download_button(
            "📥 프로모션/구분 효율 보고서 다운로드",
            data=promo_agg.to_csv(index=False).encode("utf-8-sig"),
            file_name="promotion_category_efficiency_report.csv",
            mime="text/csv",
        )

    st.sidebar.divider()
    st.sidebar.download_button(
        "📥 필터링된 데이터 다운로드",
        data=df_global_filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="ad_report_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
