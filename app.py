import io
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

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
    if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        df = df.iloc[:, 1:]
    return df


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def _parse_campaign_info(campaign_name: str) -> tuple[str, str, str]:
    parts = str(campaign_name).split("_")
    advertiser = parts[2] if len(parts) >= 3 else "미분류"
    product = parts[3] if len(parts) >= 4 else "미분류"

    product_upper = product.upper()
    campaign_upper = str(campaign_name).upper()
    campaign_lower = str(campaign_name).lower()

    if "MO" in product_upper or "모바일" in product or "모바일" in str(campaign_name):
        device = "MO"
    elif "PC" in product_upper or "pc" in campaign_lower:
        device = "PC"
    else:
        if "_MO" in campaign_upper:
            device = "MO"
        elif "_PC" in campaign_upper:
            device = "PC"
        else:
            device = "공통"

    return advertiser, product, device


def load_report(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    file_name_lower = file_name.lower()

    if file_name_lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        excel_bytes = io.BytesIO(file_bytes)
        df = pd.read_excel(excel_bytes, header=1)
        df = _normalize_columns(df)
        if _validate_columns(df, REQUIRED_COLUMNS):
            excel_bytes.seek(0)
            df = pd.read_excel(excel_bytes, header=0)

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

    campaign_info = df["캠페인명"].apply(_parse_campaign_info)
    df["광고주명"] = campaign_info.apply(lambda x: x[0])
    df["광고상품"] = campaign_info.apply(lambda x: x[1])
    df["기기구분"] = campaign_info.apply(lambda x: x[2])

    df["총클릭수"] = df["클릭수"] + df["컴패니언배너클릭수"]
    df["CTR(재계산)"] = (df["클릭수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["CTR(전체_재계산)"] = (df["총클릭수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["VTR(재계산)"] = (df["동영상조회수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["CPC(재계산)"] = (df["광고비"] / df["총클릭수"].replace(0, pd.NA)).fillna(0)
    df["CPV(재계산)"] = (df["광고비"] / df["동영상조회수"].replace(0, pd.NA)).fillna(0)
    df["eCPM(재계산)"] = (
        df["광고비"] / df["노출수"].replace(0, pd.NA) * 1000
    ).fillna(0)

    return df


def _with_share_columns(table: pd.DataFrame, base_totals: dict[str, float]) -> pd.DataFrame:
    display = table.copy()
    for col in ["노출수", "총클릭수", "동영상조회수", "광고비"]:
        denom = base_totals.get(col, 0)
        share = (display[col] / denom * 100).fillna(0) if denom else 0
        if isinstance(share, int):
            share = pd.Series([0] * len(display), index=display.index)
        display[f"{col}(비중)"] = display[col].map(lambda x: f"{x:,.0f}") + share.map(
            lambda x: f" ({x:.1f}%)"
        )
    return display


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["광고주명", "광고상품", "기기구분", "광고계정", "캠페인명"], as_index=False)
        .agg(
            노출수=("노출수", "sum"),
            클릭수=("클릭수", "sum"),
            컴패니언배너클릭수=("컴패니언배너클릭수", "sum"),
            동영상조회수=("동영상조회수", "sum"),
            광고비=("광고비", "sum"),
            시작일=("일자", "min"),
            종료일=("일자", "max"),
        )
        .sort_values(["광고주명", "캠페인명"])
    )

    grouped["총클릭수"] = grouped["클릭수"] + grouped["컴패니언배너클릭수"]
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
        grouped["광고비"] / grouped["총클릭수"].replace(0, pd.NA)
    ).fillna(0)
    grouped["CPV"] = (
        grouped["광고비"] / grouped["동영상조회수"].replace(0, pd.NA)
    ).fillna(0)

    totals = {
        "노출수": grouped["노출수"].sum(),
        "총클릭수": grouped["총클릭수"].sum(),
        "동영상조회수": grouped["동영상조회수"].sum(),
        "광고비": grouped["광고비"].sum(),
    }
    return _with_share_columns(grouped, totals)


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
    daily["CTR(전체)"] = (
        daily["총클릭수"] / daily["노출수"].replace(0, pd.NA) * 100
    ).fillna(0)
    daily["VTR"] = (daily["동영상조회수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)

    totals = {
        "노출수": daily["노출수"].sum(),
        "총클릭수": daily["총클릭수"].sum(),
        "동영상조회수": daily["동영상조회수"].sum(),
        "광고비": daily["광고비"].sum(),
    }
    return _with_share_columns(daily, totals)


def _build_pdf(campaign: pd.DataFrame, daily: pd.DataFrame, filtered: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    pdf.setFont("HYSMyeongJo-Medium", 12)

    y = 800
    pdf.drawString(40, y, "광고 캠페인 리포트")
    y -= 20
    pdf.drawString(40, y, f"기간: {filtered['일자'].min().date()} ~ {filtered['일자'].max().date()}")
    y -= 20
    pdf.drawString(
        40,
        y,
        f"총 노출수 {filtered['노출수'].sum():,.0f} / 총 클릭수 {filtered['총클릭수'].sum():,.0f} / 총 광고비 {filtered['광고비'].sum():,.0f}",
    )

    y -= 30
    pdf.drawString(40, y, "[캠페인별 요약 상위 12건]")
    y -= 18
    campaign_rows = campaign.head(12)
    for _, row in campaign_rows.iterrows():
        line = (
            f"{row['광고주명']} | {row['광고상품']} | {row['기기구분']} | "
            f"노출 {row['노출수(비중)']} | 클릭 {row['총클릭수(비중)']} | CTR(전체) {row['CTR(전체)']:.2f}%"
        )
        pdf.drawString(40, y, line[:110])
        y -= 14
        if y < 60:
            pdf.showPage()
            pdf.setFont("HYSMyeongJo-Medium", 11)
            y = 800

    y -= 10
    pdf.drawString(40, y, "[일자별 요약]")
    y -= 18
    daily_rows = daily.head(20)
    for _, row in daily_rows.iterrows():
        line = (
            f"{row['일자'].date()} | 노출 {row['노출수(비중)']} | 클릭 {row['총클릭수(비중)']} | "
            f"조회 {row['동영상조회수(비중)']} | CTR(전체) {row['CTR(전체)']:.2f}%"
        )
        pdf.drawString(40, y, line[:110])
        y -= 14
        if y < 60:
            pdf.showPage()
            pdf.setFont("HYSMyeongJo-Medium", 11)
            y = 800

    pdf.save()
    buf.seek(0)
    return buf.read()


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("📊 광고 캠페인 리포트 대시보드")
    st.caption("XLS/XLSX/CSV 업로드 후 캠페인별·일자별 성과를 자동 요약합니다.")

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

    campaigns = sorted(filtered["캠페인명"].dropna().unique())
    products = sorted(filtered["광고상품"].dropna().unique())
    devices = sorted(filtered["기기구분"].dropna().unique())

    selected_campaigns = st.multiselect("캠페인 필터", campaigns, default=campaigns)
    if selected_campaigns:
        filtered = filtered[filtered["캠페인명"].isin(selected_campaigns)]

    selected_products = st.multiselect("광고상품 필터", products, default=products)
    if selected_products:
        filtered = filtered[filtered["광고상품"].isin(selected_products)]

    selected_devices = st.multiselect("기기구분 필터", devices, default=devices)
    if selected_devices:
        filtered = filtered[filtered["기기구분"].isin(selected_devices)]

    if filtered.empty:
        st.warning("필터 결과가 비어 있습니다. 조건을 조정해주세요.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 노출수", f"{filtered['노출수'].sum():,.0f}")
    col2.metric("총 클릭수(전체)", f"{filtered['총클릭수'].sum():,.0f}")
    ctr_total = (
        filtered["총클릭수"].sum() / filtered["노출수"].sum() * 100 if filtered["노출수"].sum() else 0
    )
    col3.metric("평균 CTR(전체)", f"{ctr_total:.2f}%")
    col4.metric("총 광고비", f"{filtered['광고비'].sum():,.0f}")

    daily = daily_summary(filtered)
    campaign = campaign_summary(filtered)

    st.subheader("일자별 성과 추이")
    fig_imp_click = make_subplots(specs=[[{"secondary_y": True}]])
    fig_imp_click.add_trace(
        go.Scatter(x=daily["일자"], y=daily["노출수"], mode="lines+markers", name="노출수"),
        secondary_y=False,
    )
    fig_imp_click.add_trace(
        go.Scatter(x=daily["일자"], y=daily["총클릭수"], mode="lines+markers", name="총 클릭수"),
        secondary_y=True,
    )
    fig_imp_click.add_trace(
        go.Scatter(x=daily["일자"], y=daily["동영상조회수"], mode="lines+markers", name="동영상조회수"),
        secondary_y=True,
    )
    fig_imp_click.update_layout(title="노출수(좌측축) / 총클릭수·조회수(우측축)")
    fig_imp_click.update_yaxes(title_text="노출수", secondary_y=False)
    fig_imp_click.update_yaxes(title_text="총클릭수·조회수", secondary_y=True)
    st.plotly_chart(fig_imp_click, use_container_width=True)

    fig_ctr = go.Figure()
    fig_ctr.add_trace(
        go.Scatter(x=daily["일자"], y=daily["CTR(전체)"], mode="lines+markers", name="CTR(전체)")
    )
    fig_ctr.add_trace(
        go.Scatter(x=daily["일자"], y=daily["VTR"], mode="lines+markers", name="VTR")
    )
    fig_ctr.update_layout(title="CTR(전체) / VTR 추이(%)", yaxis_title="비율(%)")
    st.plotly_chart(fig_ctr, use_container_width=True)

    st.subheader("캠페인별 요약")
    st.dataframe(
        campaign,
        use_container_width=True,
        column_config={
            "시작일": st.column_config.DateColumn("시작일", format="YYYY-MM-DD"),
            "종료일": st.column_config.DateColumn("종료일", format="YYYY-MM-DD"),
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
            "CTR(전체)": st.column_config.NumberColumn("CTR(전체, %)", format="%.2f"),
            "VTR": st.column_config.NumberColumn("VTR(%)", format="%.2f"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="캠페인 요약 CSV 다운로드",
            data=campaign.to_csv(index=False).encode("utf-8-sig"),
            file_name="campaign_summary.csv",
            mime="text/csv",
        )
    with c2:
        pdf_bytes = _build_pdf(campaign, daily, filtered)
        st.download_button(
            label="광고주 보고용 PDF 다운로드",
            data=pdf_bytes,
            file_name="ad_report.pdf",
            mime="application/pdf",
        )


def main() -> None:
    st.sidebar.header("파일 업로드")
    upload = st.sidebar.file_uploader("엑셀(.xls/.xlsx) 또는 CSV 파일", type=["xls", "xlsx", "csv"])
    st.sidebar.caption("필수 컬럼: " + ", ".join(REQUIRED_COLUMNS))

    if upload is None:
        st.title("광고 캠페인 리포트 자동화")
        st.write(
            "좌측에서 파일을 업로드하면 광고주 공유용 캠페인 성과 대시보드가 자동 생성됩니다."
        )
        st.code("python run_dashboard.py")
        return

    try:
        df = load_report(upload.getvalue(), upload.name)
    except Exception as exc:
        st.error(f"파일 처리 중 오류가 발생했습니다: {exc}")
        return

    render_dashboard(df)


if __name__ == "__main__":
    main()
