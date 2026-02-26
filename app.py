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
import streamlit as st

st.set_page_config(page_title="광고 효율 벤치마킹 대시보드", layout="wide")

REQUIRED_COLUMNS = [
    "캠페인명",
    "일자",
    "노출수",
    "클릭수",
    "컴패니언배너클릭수",
    "총클릭수",
    "동영상조회수",
    "CTR",
    "CTR(전체)",
    "VTR",
    "광고비",
    "eCPM",
    "CPC",
    "CPV",
    "연도",
    "월",
    "광고상품명_정리",
    "구분",
]

NUMERIC_COLUMNS = [
    "노출수",
    "클릭수",
    "컴패니언배너클릭수",
    "총클릭수",
    "동영상조회수",
    "CTR",
    "CTR(전체)",
    "VTR",
    "광고비",
    "eCPM",
    "CPC",
    "CPV",
    "연도",
    "월",
]

SECTION_NAMES = [
    "KPI 카드",
    "상품별 집계 테이블",
    "TOP/WORST 랭킹",
    "월별 추이",
    "요일별 분석",
    "구분별 비교",
]

DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}


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
            "광고상품명_정리": ", ".join(self.products) if self.products else "전체",
            "캠페인명": ", ".join(self.campaigns[:10]) + (" ..." if len(self.campaigns) > 10 else "") if self.campaigns else "전체",
        }


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den_clean = den.replace(0, np.nan)
    return num / den_clean


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

    for col in ["캠페인명", "광고상품명_정리", "구분"]:
        out[col] = out[col].astype("string").fillna("미정")

    out["연월"] = pd.to_datetime(
        dict(year=out["연도"].astype("Int64"), month=out["월"].astype("Int64"), day=1),
        errors="coerce",
    )
    return out


@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, filters: FilterValues) -> pd.DataFrame:
    out = df.copy()
    if filters.date_min is not None:
        out = out[out["일자"] >= filters.date_min]
    if filters.date_max is not None:
        out = out[out["일자"] <= filters.date_max]
    if filters.years:
        out = out[out["연도"].isin(filters.years)]
    if filters.months:
        out = out[out["월"].isin(filters.months)]
    if filters.weekdays:
        out = out[out["요일"].isin(filters.weekdays)]
    if filters.categories:
        out = out[out["구분"].isin(filters.categories)]
    if filters.products:
        out = out[out["광고상품명_정리"].isin(filters.products)]
    if filters.campaigns:
        out = out[out["캠페인명"].isin(filters.campaigns)]
    return out


@st.cache_data(show_spinner=False)
def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    base = (
        df.groupby(group_cols, dropna=False, as_index=False)[
            ["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "광고비", "동영상조회수"]
        ]
        .sum(min_count=1)
        .fillna(0)
    )

    base["CTR_recalc"] = safe_div(base["클릭수"], base["노출수"])
    base["CTR_total_recalc"] = safe_div(base["총클릭수"], base["노출수"])
    base["CPC_recalc"] = safe_div(base["광고비"], base["클릭수"])
    base["CPC_total_recalc"] = safe_div(base["광고비"], base["총클릭수"])
    base["eCPM_recalc"] = safe_div(base["광고비"], base["노출수"]) * 1000
    base["VTR_recalc"] = safe_div(base["동영상조회수"], base["노출수"])
    base["CPV_recalc"] = safe_div(base["광고비"], base["동영상조회수"])
    return base


def make_filter_values(
    *,
    date_range: Optional[tuple],
    years: List[int],
    months: List[int],
    weekdays: List[str],
    categories: List[str],
    products: List[str],
    campaigns: List[str],
) -> FilterValues:
    date_min = pd.to_datetime(date_range[0]) if date_range else None
    date_max = pd.to_datetime(date_range[1]) if date_range else None
    return FilterValues(
        date_min=date_min,
        date_max=date_max,
        years=years,
        months=months,
        weekdays=weekdays,
        categories=categories,
        products=products,
        campaigns=campaigns,
    )


def render_filter_summary(filters: FilterValues):
    with st.expander("사용 중인 필터 조건", expanded=False):
        for k, v in filters.to_summary().items():
            st.caption(f"- {k}: {v}")


def filter_badge(status: str):
    color = {"Global 적용": "#1f77b4", "Local 적용": "#ff7f0e", "Unfiltered": "#7f7f7f"}.get(status, "#7f7f7f")
    st.markdown(
        f"<span style='background:{color}; color:white; padding:2px 8px; border-radius:999px; font-size:12px;'>{status}</span>",
        unsafe_allow_html=True,
    )


def section_header(title: str, status: str):
    c1, c2 = st.columns([6, 1])
    with c1:
        st.subheader(title)
    with c2:
        filter_badge(status)


def local_filter_ui(df: pd.DataFrame, section_key: str) -> tuple[bool, FilterValues]:
    use_global = st.checkbox("Use global filters", value=True, key=f"use_global_{section_key}")
    if use_global:
        return True, make_filter_values(
            date_range=None,
            years=[],
            months=[],
            weekdays=[],
            categories=[],
            products=[],
            campaigns=[],
        )

    st.caption("이 섹션에만 적용되는 로컬 필터")
    min_date = df["일자"].min()
    max_date = df["일자"].max()
    local_date = st.date_input(
        "일자 범위(로컬)",
        value=(min_date.date(), max_date.date()) if pd.notna(min_date) and pd.notna(max_date) else (),
        key=f"local_date_{section_key}",
    )

    years = sorted(df["연도"].dropna().astype(int).unique().tolist())
    months = sorted(df["월"].dropna().astype(int).unique().tolist())
    weekdays = [d for d in DAY_ORDER if d in set(df["요일"].dropna().unique())]
    cats = sorted(df["구분"].dropna().unique().tolist())
    products = sorted(df["광고상품명_정리"].dropna().unique().tolist())
    campaigns = sorted(df["캠페인명"].dropna().unique().tolist())

    l_years = st.multiselect("연도(로컬)", options=years, default=[] if not years else years, key=f"local_years_{section_key}")
    l_months = st.multiselect("월(로컬)", options=months, default=[] if not months else months, key=f"local_months_{section_key}")
    l_weekdays = st.multiselect("요일(로컬)", options=weekdays, default=[] if not weekdays else weekdays, key=f"local_weekdays_{section_key}")
    l_cats = st.multiselect("구분(로컬)", options=cats, default=[] if not cats else cats, key=f"local_cats_{section_key}")
    l_products = st.multiselect("광고상품명_정리(로컬)", options=products, default=[] if not products else products, key=f"local_products_{section_key}")
    search = st.text_input("캠페인 검색(로컬)", key=f"local_campaign_search_{section_key}")
    filtered_campaigns = [c for c in campaigns if search.lower() in c.lower()]
    l_campaigns = st.multiselect("캠페인명(로컬)", options=filtered_campaigns, default=filtered_campaigns, key=f"local_campaigns_{section_key}")

    if isinstance(local_date, tuple) and len(local_date) == 2:
        date_range = local_date
    else:
        date_range = None

    local_filters = make_filter_values(
        date_range=date_range,
        years=l_years,
        months=l_months,
        weekdays=l_weekdays,
        categories=l_cats,
        products=l_products,
        campaigns=l_campaigns,
    )
    return False, local_filters


def main():
    st.title("광고상품 효율(노출/클릭/CTR) 벤치마킹 대시보드")

    st.sidebar.header("데이터 로드")
    upload = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    path = st.sidebar.text_input("또는 CSV 파일 경로 입력")

    if upload is None and not path:
        st.info("CSV 파일을 업로드하거나 경로를 입력해 주세요.")
        st.stop()

    try:
        if upload is not None:
            df_raw = load_csv_from_bytes(upload.getvalue())
        else:
            df_raw = load_csv_from_path(path)
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        st.stop()

    df_raw = preprocess_df(df_raw)

    st.sidebar.header("GLOBAL FILTER (전역 필터)")
    min_date = df_raw["일자"].min()
    max_date = df_raw["일자"].max()
    date_range = st.sidebar.date_input(
        "일자 범위",
        value=(min_date.date(), max_date.date()) if pd.notna(min_date) and pd.notna(max_date) else (),
    )

    all_years = sorted(df_raw["연도"].dropna().astype(int).unique().tolist())
    all_months = sorted(df_raw["월"].dropna().astype(int).unique().tolist())
    all_weekdays = [d for d in DAY_ORDER if d in set(df_raw["요일"].dropna().unique())]
    all_cats = sorted(df_raw["구분"].dropna().unique().tolist())
    all_products = sorted(df_raw["광고상품명_정리"].dropna().unique().tolist())
    all_campaigns = sorted(df_raw["캠페인명"].dropna().unique().tolist())

    g_years = st.sidebar.multiselect("연도", options=all_years, default=all_years)
    g_months = st.sidebar.multiselect("월", options=all_months, default=all_months)
    g_weekdays = st.sidebar.multiselect("요일", options=all_weekdays, default=all_weekdays)
    g_cats = st.sidebar.multiselect("구분", options=all_cats, default=all_cats)
    g_products = st.sidebar.multiselect("광고상품명_정리", options=all_products, default=all_products)
    g_campaign_search = st.sidebar.text_input("캠페인명 검색")
    g_filtered_campaigns = [c for c in all_campaigns if g_campaign_search.lower() in c.lower()]
    g_campaigns = st.sidebar.multiselect("캠페인명", options=g_filtered_campaigns, default=g_filtered_campaigns)

    st.sidebar.subheader("FILTER SCOPE (전역 필터 적용 섹션)")
    scope_map = {sec: st.sidebar.checkbox(sec, value=True, key=f"scope_{sec}") for sec in SECTION_NAMES}

    if isinstance(date_range, tuple) and len(date_range) == 2:
        g_date_range = date_range
    else:
        g_date_range = None

    global_filters = make_filter_values(
        date_range=g_date_range,
        years=g_years,
        months=g_months,
        weekdays=g_weekdays,
        categories=g_cats,
        products=g_products,
        campaigns=g_campaigns,
    )

    df_global_filtered = apply_filters(df_raw, global_filters)

    def get_section_df(section_name: str, local_filters: Optional[FilterValues], use_global: bool):
        base = df_global_filtered if scope_map.get(section_name, False) else df_raw
        if use_global:
            return base
        if local_filters is None:
            return base
        return apply_filters(base, local_filters)

    st.caption(f"원천 데이터: {len(df_raw):,}행 | 전역 필터 적용 후: {len(df_global_filtered):,}행")

    st.download_button(
        "필터 적용 데이터 다운로드(CSV)",
        data=df_global_filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_data.csv",
        mime="text/csv",
    )

    # A. KPI cards
    section = "KPI 카드"
    section_header("A. KPI 카드", "Global 적용" if scope_map[section] else "Unfiltered")
    use_global_kpi, local_kpi = local_filter_ui(df_raw, "kpi")
    df_kpi = get_section_df(section, local_kpi, use_global_kpi)
    status_kpi = "Local 적용" if not use_global_kpi else ("Global 적용" if scope_map[section] else "Unfiltered")
    section_header("", status_kpi)
    render_filter_summary(global_filters if use_global_kpi and scope_map[section] else (local_kpi if not use_global_kpi else make_filter_values(date_range=None, years=[], months=[], weekdays=[], categories=[], products=[], campaigns=[])))

    agg_kpi = aggregate_metrics(df_kpi, ["광고상품명_정리"])
    total = agg_kpi[["노출수", "클릭수", "총클릭수", "광고비"]].sum()
    total_ctr = (agg_kpi["총클릭수"].sum() / agg_kpi["노출수"].sum()) if agg_kpi["노출수"].sum() else np.nan
    total_cpc = (agg_kpi["광고비"].sum() / agg_kpi["총클릭수"].sum()) if agg_kpi["총클릭수"].sum() else np.nan
    total_ecpm = (agg_kpi["광고비"].sum() / agg_kpi["노출수"].sum() * 1000) if agg_kpi["노출수"].sum() else np.nan

    cols = st.columns(7)
    cols[0].metric("노출수", f"{total['노출수']:,.0f}")
    cols[1].metric("클릭수", f"{total['클릭수']:,.0f}")
    cols[2].metric("총클릭수", f"{total['총클릭수']:,.0f}")
    cols[3].metric("CTR_total_recalc", f"{total_ctr:.2%}" if pd.notna(total_ctr) else "N/A")
    cols[4].metric("광고비", f"{total['광고비']:,.0f}")
    cols[5].metric("CPC_total_recalc", f"{total_cpc:,.2f}" if pd.notna(total_cpc) else "N/A")
    cols[6].metric("eCPM_recalc", f"{total_ecpm:,.2f}" if pd.notna(total_ecpm) else "N/A")

    # B. Product table
    section = "상품별 집계 테이블"
    st.divider()
    section_header("B. 광고상품명_정리 성과 테이블", "Global 적용" if scope_map[section] else "Unfiltered")
    use_global_tbl, local_tbl = local_filter_ui(df_raw, "table")
    df_tbl = get_section_df(section, local_tbl, use_global_tbl)
    min_imps = st.number_input("최소 노출 임계값", min_value=0, value=10000, step=1000)
    sort_col = st.selectbox("정렬 기준", ["CTR_total_recalc", "CPC_total_recalc", "eCPM_recalc", "광고비"])
    asc = sort_col == "CPC_total_recalc"

    prod_agg = aggregate_metrics(df_tbl, ["광고상품명_정리"])
    prod_agg = prod_agg[prod_agg["노출수"] >= min_imps]
    prod_agg = prod_agg.sort_values(sort_col, ascending=asc)

    show_cols = ["광고상품명_정리", "노출수", "총클릭수", "CTR_total_recalc", "광고비", "CPC_total_recalc", "eCPM_recalc"]
    st.dataframe(prod_agg[show_cols], use_container_width=True)
    st.download_button(
        "상품별 집계 테이블 다운로드(CSV)",
        data=prod_agg.to_csv(index=False).encode("utf-8-sig"),
        file_name="product_agg.csv",
        mime="text/csv",
    )

    # C. TOP/WORST
    section = "TOP/WORST 랭킹"
    st.divider()
    section_header("C. TOP & WORST 자동 랭킹", "Global 적용" if scope_map[section] else "Unfiltered")
    use_global_rank, local_rank = local_filter_ui(df_raw, "rank")
    df_rank = get_section_df(section, local_rank, use_global_rank)
    rank_agg = aggregate_metrics(df_rank, ["광고상품명_정리"])
    rank_agg = rank_agg[rank_agg["노출수"] >= min_imps]

    top_ctr = rank_agg.nlargest(5, "CTR_total_recalc")[["광고상품명_정리", "CTR_total_recalc", "노출수", "총클릭수"]]
    low_cpc = rank_agg[(rank_agg["광고비"] > 0) & (rank_agg["총클릭수"] > 0)].nsmallest(5, "CPC_total_recalc")[["광고상품명_정리", "CPC_total_recalc", "광고비", "총클릭수"]]

    c1, c2 = st.columns(2)
    c1.markdown("**CTR_total_recalc TOP5**")
    c1.dataframe(top_ctr, use_container_width=True)
    c2.markdown("**CPC_total_recalc LOW5**")
    c2.dataframe(low_cpc, use_container_width=True)

    use_score = st.checkbox("효율 스코어(z(CTR_total_recalc)-z(CPC_total_recalc)) 사용", value=True)
    if use_score and len(rank_agg) >= 2:
        score_df = rank_agg.copy()
        score_df = score_df[(score_df["CTR_total_recalc"].notna()) & (score_df["CPC_total_recalc"].notna())]
        if len(score_df) >= 2:
            score_df["score"] = (
                (score_df["CTR_total_recalc"] - score_df["CTR_total_recalc"].mean()) / score_df["CTR_total_recalc"].std(ddof=0)
                - (score_df["CPC_total_recalc"] - score_df["CPC_total_recalc"].mean()) / score_df["CPC_total_recalc"].std(ddof=0)
            )
            s1, s2 = st.columns(2)
            s1.markdown("**Score TOP5**")
            s1.dataframe(score_df.nlargest(5, "score")[["광고상품명_정리", "score"]], use_container_width=True)
            s2.markdown("**Score WORST5**")
            s2.dataframe(score_df.nsmallest(5, "score")[["광고상품명_정리", "score"]], use_container_width=True)

    # D. Monthly trend
    section = "월별 추이"
    st.divider()
    section_header("D. 월별 추이", "Global 적용" if scope_map[section] else "Unfiltered")
    use_global_month, local_month = local_filter_ui(df_raw, "month")
    df_month = get_section_df(section, local_month, use_global_month)
    sel_products = st.multiselect(
        "월별 추이에 표시할 광고상품(멀티)",
        options=sorted(df_month["광고상품명_정리"].dropna().unique().tolist()),
        default=sorted(df_month["광고상품명_정리"].dropna().unique().tolist())[:5],
    )
    if sel_products:
        month_agg = aggregate_metrics(df_month[df_month["광고상품명_정리"].isin(sel_products)], ["연월", "광고상품명_정리"])
        month_agg = month_agg.sort_values("연월")
        fig_ctr = px.line(month_agg, x="연월", y="CTR_total_recalc", color="광고상품명_정리", title="월별 CTR_total_recalc")
        fig_cpc = px.line(month_agg, x="연월", y="CPC_total_recalc", color="광고상품명_정리", title="월별 CPC_total_recalc")
        fig_imp = px.line(month_agg, x="연월", y="노출수", color="광고상품명_정리", title="월별 노출수")
        st.plotly_chart(fig_ctr, use_container_width=True)
        st.plotly_chart(fig_cpc, use_container_width=True)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("표시할 광고상품을 선택해 주세요.")

    # E. Weekday
    section = "요일별 분석"
    st.divider()
    section_header("E. 요일별 분석", "Global 적용" if scope_map[section] else "Unfiltered")
    use_global_day, local_day = local_filter_ui(df_raw, "weekday")
    df_day = get_section_df(section, local_day, use_global_day)

    day_agg = aggregate_metrics(df_day, ["요일"])
    day_agg["요일"] = pd.Categorical(day_agg["요일"], categories=DAY_ORDER, ordered=True)
    day_agg = day_agg.sort_values("요일")
    fig_day = px.bar(day_agg, x="요일", y="CTR_total_recalc", title="요일별 CTR_total_recalc")
    st.plotly_chart(fig_day, use_container_width=True)

    heat_agg = aggregate_metrics(df_day, ["광고상품명_정리", "요일"])
    heat_agg = heat_agg[heat_agg["노출수"] >= min_imps]
    if not heat_agg.empty:
        heat = heat_agg.pivot(index="광고상품명_정리", columns="요일", values="CTR_total_recalc").reindex(columns=DAY_ORDER)
        fig_heat = px.imshow(heat, aspect="auto", title="상품×요일 CTR_total_recalc Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

    # F. Category
    section = "구분별 비교"
    st.divider()
    section_header("F. 구분별 비교", "Global 적용" if scope_map[section] else "Unfiltered")
    use_global_cat, local_cat = local_filter_ui(df_raw, "category")
    df_cat = get_section_df(section, local_cat, use_global_cat)

    cat_agg = aggregate_metrics(df_cat, ["구분"])
    f1 = px.bar(cat_agg, x="구분", y="CTR_total_recalc", title="구분별 CTR_total_recalc")
    f2 = px.bar(cat_agg, x="구분", y="CPC_total_recalc", title="구분별 CPC_total_recalc")
    f3 = px.bar(cat_agg, x="구분", y="eCPM_recalc", title="구분별 eCPM_recalc")
    st.plotly_chart(f1, use_container_width=True)
    st.plotly_chart(f2, use_container_width=True)
    st.plotly_chart(f3, use_container_width=True)

    with st.expander("원본 지표 보기(참고용)"):
        ref_cols = ["CTR", "CTR(전체)", "VTR", "CPC", "CPV", "eCPM"]
        st.dataframe(df_global_filtered[ref_cols].describe(include="all").T, use_container_width=True)


if __name__ == "__main__":
    main()
