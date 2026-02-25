import io
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="광고 캠페인 보고서", layout="wide")

REQUIRED_COLUMNS = [
    "광고계정",
    "회사",
    "캠페인명",
    "일자",
    "노출수",
    "클릭수",
    "컴패니언배너클릭수",
    "동영상조회수",
    "CTR",
    "CTR(전체)",
    "VTR",
    "광고비",
    "eCPM",
    "CPC",
    "CPV",
]

NUMERIC_COLUMNS = [
    "노출수",
    "클릭수",
    "컴패니언배너클릭수",
    "동영상조회수",
    "광고비",
]


def _to_number(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("%", "").strip()
        if cleaned in {"", "-"}:
            return 0.0
        return float(cleaned)
    return float(value)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def load_report(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    df = _normalize_columns(df)
    missing_cols = _validate_columns(df, REQUIRED_COLUMNS)
    if missing_cols:
        raise ValueError(
            "업로드 파일에 필요한 컬럼이 없습니다: " + ", ".join(missing_cols)
        )

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].apply(_to_number)

    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
    if df["일자"].isna().all():
        raise ValueError("'일자' 컬럼 날짜 파싱에 실패했습니다. 날짜 형식을 확인해주세요.")

    df = df.dropna(subset=["일자"]).sort_values("일자")

    df["총클릭수"] = df["클릭수"] + df["컴패니언배너클릭수"]
    df["CTR(재계산)"] = (df["클릭수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["CTR(전체_재계산)"] = (df["총클릭수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["VTR(재계산)"] = (df["동영상조회수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["CPC(재계산)"] = (df["광고비"] / df["클릭수"].replace(0, pd.NA)).fillna(0)
    df["CPV(재계산)"] = (df["광고비"] / df["동영상조회수"].replace(0, pd.NA)).fillna(0)
    df["eCPM(재계산)"] = (
        df["광고비"] / df["노출수"].replace(0, pd.NA) * 1000
    ).fillna(0)

    return df


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["회사", "광고계정", "캠페인명"], as_index=False)
        .agg(
            노출수=("노출수", "sum"),
            클릭수=("클릭수", "sum"),
            컴패니언배너클릭수=("컴패니언배너클릭수", "sum"),
            동영상조회수=("동영상조회수", "sum"),
            광고비=("광고비", "sum"),
            시작일=("일자", "min"),
            종료일=("일자", "max"),
        )
        .sort_values(["회사", "캠페인명"])
    )

    grouped["총클릭수"] = grouped["클릭수"] + grouped["컴패니언배너클릭수"]
    grouped["CTR"] = (grouped["클릭수"] / grouped["노출수"].replace(0, pd.NA) * 100).fillna(0)
    grouped["CTR(전체)"] = (
        grouped["총클릭수"] / grouped["노출수"].replace(0, pd.NA) * 100
    ).fillna(0)
    grouped["VTR"] = (
        grouped["동영상조회수"] / grouped["노출수"].replace(0, pd.NA) * 100
    ).fillna(0)
    grouped["eCPM"] = (
        grouped["광고비"] / grouped["노출수"].replace(0, pd.NA) * 1000
    ).fillna(0)
    grouped["CPC"] = (
        grouped["광고비"] / grouped["클릭수"].replace(0, pd.NA)
    ).fillna(0)
    grouped["CPV"] = (
        grouped["광고비"] / grouped["동영상조회수"].replace(0, pd.NA)
    ).fillna(0)

    return grouped


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("일자", as_index=False)
        .agg(
            노출수=("노출수", "sum"),
            클릭수=("클릭수", "sum"),
            컴패니언배너클릭수=("컴패니언배너클릭수", "sum"),
            동영상조회수=("동영상조회수", "sum"),
            광고비=("광고비", "sum"),
        )
        .sort_values("일자")
    )
    daily["총클릭수"] = daily["클릭수"] + daily["컴패니언배너클릭수"]
    daily["CTR"] = (daily["클릭수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)
    daily["CTR(전체)"] = (
        daily["총클릭수"] / daily["노출수"].replace(0, pd.NA) * 100
    ).fillna(0)
    daily["VTR"] = (daily["동영상조회수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)
    return daily


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("📊 광고 캠페인 리포트 대시보드")
    st.caption("XLSX/CSV 업로드 후 캠페인별·일자별 성과를 자동 요약합니다.")

    min_date = df["일자"].min().date()
    max_date = df["일자"].max().date()
    selected_range = st.date_input(
        "분석 기간",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        filtered = df[(df["일자"].dt.date >= start_date) & (df["일자"].dt.date <= end_date)]
    else:
        filtered = df.copy()

    companies = sorted(filtered["회사"].dropna().unique())
    campaigns = sorted(filtered["캠페인명"].dropna().unique())

    selected_companies = st.multiselect("회사 필터", companies, default=companies)
    if selected_companies:
        filtered = filtered[filtered["회사"].isin(selected_companies)]

    selected_campaigns = st.multiselect("캠페인 필터", campaigns, default=campaigns)
    if selected_campaigns:
        filtered = filtered[filtered["캠페인명"].isin(selected_campaigns)]

    if filtered.empty:
        st.warning("필터 결과가 비어 있습니다. 조건을 조정해주세요.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 노출수", f"{filtered['노출수'].sum():,.0f}")
    col2.metric("총 클릭수", f"{filtered['클릭수'].sum():,.0f}")
    ctr = filtered["클릭수"].sum() / filtered["노출수"].sum() * 100 if filtered["노출수"].sum() else 0
    col3.metric("평균 CTR", f"{ctr:.2f}%")
    col4.metric("총 광고비", f"{filtered['광고비'].sum():,.0f}")

    daily = daily_summary(filtered)
    campaign = campaign_summary(filtered)

    st.subheader("일자별 성과 추이")
    fig_imp_click = px.line(
        daily,
        x="일자",
        y=["노출수", "클릭수", "동영상조회수"],
        markers=True,
        title="노출/클릭/조회수 추이",
    )
    st.plotly_chart(fig_imp_click, use_container_width=True)

    fig_ctr = px.line(
        daily,
        x="일자",
        y=["CTR", "CTR(전체)", "VTR"],
        markers=True,
        title="CTR/VTR 추이(%)",
    )
    st.plotly_chart(fig_ctr, use_container_width=True)

    st.subheader("캠페인별 요약")
    st.dataframe(
        campaign,
        use_container_width=True,
        column_config={
            "시작일": st.column_config.DateColumn("시작일", format="YYYY-MM-DD"),
            "종료일": st.column_config.DateColumn("종료일", format="YYYY-MM-DD"),
            "CTR": st.column_config.NumberColumn("CTR(%)", format="%.2f"),
            "CTR(전체)": st.column_config.NumberColumn("CTR(전체, %)", format="%.2f"),
            "VTR": st.column_config.NumberColumn("VTR(%)", format="%.2f"),
            "eCPM": st.column_config.NumberColumn("eCPM", format="%.2f"),
            "CPC": st.column_config.NumberColumn("CPC", format="%.2f"),
            "CPV": st.column_config.NumberColumn("CPV", format="%.2f"),
        },
    )

    st.subheader("일자별 상세 테이블")
    st.dataframe(
        daily,
        use_container_width=True,
        column_config={
            "일자": st.column_config.DateColumn("일자", format="YYYY-MM-DD"),
            "CTR": st.column_config.NumberColumn("CTR(%)", format="%.2f"),
            "CTR(전체)": st.column_config.NumberColumn("CTR(전체, %)", format="%.2f"),
            "VTR": st.column_config.NumberColumn("VTR(%)", format="%.2f"),
        },
    )

    st.download_button(
        label="캠페인 요약 CSV 다운로드",
        data=campaign.to_csv(index=False).encode("utf-8-sig"),
        file_name="campaign_summary.csv",
        mime="text/csv",
    )


def main() -> None:
    st.sidebar.header("파일 업로드")
    upload = st.sidebar.file_uploader("엑셀(.xlsx) 또는 CSV 파일", type=["xlsx", "csv"])
    st.sidebar.caption("필수 컬럼: " + ", ".join(REQUIRED_COLUMNS))

    if upload is None:
        st.title("광고 캠페인 리포트 자동화")
        st.write(
            "좌측에서 파일을 업로드하면 광고주 공유용 캠페인 성과 대시보드가 자동 생성됩니다."
        )
        st.code("streamlit run app.py")
        return

    try:
        df = load_report(upload.getvalue(), upload.name)
    except Exception as exc:
        st.error(f"파일 처리 중 오류가 발생했습니다: {exc}")
        return

    render_dashboard(df)


if __name__ == "__main__":
    main()
