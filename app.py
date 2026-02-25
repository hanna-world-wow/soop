import io
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
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

NUMERIC_COLUMNS = ["노출수", "클릭수", "컴패니언배너클릭수", "동영상조회수", "광고비"]


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
    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]
    if len(normalized.columns) > 0 and str(normalized.columns[0]).startswith("Unnamed"):
        normalized = normalized.iloc[:, 1:]
    return normalized


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def _parse_campaign_info(campaign_name: str) -> tuple[str, str, str]:
    parts = str(campaign_name).split("_")
    advertiser = parts[2] if len(parts) >= 3 else "미분류"
    product = parts[3] if len(parts) >= 4 else "미분류"

    campaign_upper = str(campaign_name).upper()
    campaign_lower = str(campaign_name).lower()
    if "MO" in product.upper() or "모바일" in str(campaign_name):
        device = "MO"
    elif "PC" in product.upper() or "pc" in campaign_lower:
        device = "PC"
    elif "_MO" in campaign_upper:
        device = "MO"
    elif "_PC" in campaign_upper:
        device = "PC"
    else:
        device = "공통"

    return advertiser, product, device


def load_report(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
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
        raise ValueError("업로드 파일에 필요한 컬럼이 없습니다: " + ", ".join(missing_cols))

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].apply(_to_number)

    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
    if df["일자"].isna().all():
        raise ValueError("'일자' 컬럼 날짜 파싱에 실패했습니다. 날짜 형식을 확인해주세요.")

    df = df.dropna(subset=["일자"]).sort_values("일자")

    parsed = df["캠페인명"].apply(_parse_campaign_info)
    df["광고주명"] = parsed.apply(lambda x: x[0])
    df["광고상품"] = parsed.apply(lambda x: x[1])
    df["기기구분"] = parsed.apply(lambda x: x[2])

    df["총클릭수"] = df["클릭수"] + df["컴패니언배너클릭수"]
    df["CTR(전체_재계산)"] = (df["총클릭수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["VTR(재계산)"] = (df["동영상조회수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["CPC(재계산)"] = (df["광고비"] / df["총클릭수"].replace(0, pd.NA)).fillna(0)
    df["CPV(재계산)"] = (df["광고비"] / df["동영상조회수"].replace(0, pd.NA)).fillna(0)
    df["eCPM(재계산)"] = (df["광고비"] / df["노출수"].replace(0, pd.NA) * 1000).fillna(0)

    return df


def _active_dates(df: pd.DataFrame) -> pd.Index:
    daily = df.groupby("일자", as_index=False).agg(노출수=("노출수", "sum"))
    return daily[daily["노출수"] > 0]["일자"]


def _with_share_columns(table: pd.DataFrame, total_map: dict[str, float]) -> pd.DataFrame:
    display = table.copy()
    for col in ["노출수", "총클릭수", "동영상조회수", "광고비"]:
        denom = total_map.get(col, 0)
        if denom:
            share = (display[col] / denom * 100).fillna(0)
        else:
            share = pd.Series([0.0] * len(display), index=display.index)
        display[f"{col}(비중)"] = display[col].map(lambda x: f"{x:,.0f}") + share.map(
            lambda x: f" ({x:.1f}%)"
        )
    return display


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["광고주명", "광고상품", "기기구분", "광고계정", "캠페인명"]
    grouped = (
        df.groupby(keys, as_index=False)
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

    active_days = (
        df[df["노출수"] > 0].groupby(keys, as_index=False)["일자"].nunique().rename(columns={"일자": "광고일수"})
    )
    grouped = grouped.merge(active_days, on=keys, how="left")
    grouped["광고일수"] = grouped["광고일수"].fillna(0).astype(int)

    grouped["총클릭수"] = grouped["클릭수"] + grouped["컴패니언배너클릭수"]
    grouped["CTR(전체)"] = (grouped["총클릭수"] / grouped["노출수"].replace(0, pd.NA) * 100).fillna(0)
    grouped["VTR"] = (grouped["동영상조회수"] / grouped["노출수"].replace(0, pd.NA) * 100).fillna(0)
    grouped["eCPM"] = (grouped["광고비"] / grouped["노출수"].replace(0, pd.NA) * 1000).fillna(0)
    grouped["CPC"] = (grouped["광고비"] / grouped["총클릭수"].replace(0, pd.NA)).fillna(0)
    grouped["CPV"] = (grouped["광고비"] / grouped["동영상조회수"].replace(0, pd.NA)).fillna(0)
    return grouped


def product_device_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["광고상품", "기기구분"], as_index=False)
        .agg(
            노출수=("노출수", "sum"),
            클릭수=("클릭수", "sum"),
            컴패니언배너클릭수=("컴패니언배너클릭수", "sum"),
            동영상조회수=("동영상조회수", "sum"),
            광고비=("광고비", "sum"),
        )
        .sort_values(["광고상품", "기기구분"])
    )
    summary["총클릭수"] = summary["클릭수"] + summary["컴패니언배너클릭수"]
    summary["CTR(전체)"] = (summary["총클릭수"] / summary["노출수"].replace(0, pd.NA) * 100).fillna(0)
    summary["VTR"] = (summary["동영상조회수"] / summary["노출수"].replace(0, pd.NA) * 100).fillna(0)
    return summary


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
    daily = daily[daily["노출수"] > 0].copy()
    daily["총클릭수"] = daily["클릭수"] + daily["컴패니언배너클릭수"]
    daily["CTR(전체)"] = (daily["총클릭수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)
    daily["VTR"] = (daily["동영상조회수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)
    return daily


def _trend_figures(daily: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
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
    fig_imp_click.update_layout(
        title="노출수(좌측축) / 총클릭수·조회수(우측축)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig_imp_click.update_yaxes(title_text="노출수", secondary_y=False)
    fig_imp_click.update_yaxes(title_text="총클릭수·조회수", secondary_y=True)

    fig_ctr = go.Figure()
    fig_ctr.add_trace(
        go.Scatter(x=daily["일자"], y=daily["CTR(전체)"], mode="lines+markers", name="CTR(전체)")
    )
    fig_ctr.add_trace(go.Scatter(x=daily["일자"], y=daily["VTR"], mode="lines+markers", name="VTR"))
    fig_ctr.update_layout(
        title="CTR(전체) / VTR 추이(%)",
        yaxis_title="비율(%)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig_imp_click, fig_ctr


def _fig_to_png_bytes(fig: go.Figure) -> bytes | None:
    try:
        return fig.to_image(format="png", width=1400, height=500, scale=2)
    except Exception:
        return None


def _build_pdf(
    filtered: pd.DataFrame,
    daily: pd.DataFrame,
    top_impression: pd.DataFrame,
    top_ctr: pd.DataFrame,
    fig_imp_click: go.Figure,
    fig_ctr: go.Figure,
) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)

    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("HYSMyeongJo-Medium", 14)
    pdf.drawString(30, 810, "광고 캠페인 보고서")
    pdf.setFont("HYSMyeongJo-Medium", 10)
    pdf.drawString(30, 793, f"분석기간: {daily['일자'].min().date()} ~ {daily['일자'].max().date()}")
    pdf.drawString(
        30,
        778,
        f"총 노출수 {filtered['노출수'].sum():,.0f} | 총 클릭수 {filtered['총클릭수'].sum():,.0f} | 총 광고비 {filtered['광고비'].sum():,.0f}",
    )

    y = 760
    for fig in [fig_imp_click, fig_ctr]:
        image_bytes = _fig_to_png_bytes(fig)
        if image_bytes is not None:
            img = ImageReader(io.BytesIO(image_bytes))
            pdf.drawImage(img, 30, y - 170, width=535, height=160, preserveAspectRatio=True)
            y -= 180

    pdf.setFont("HYSMyeongJo-Medium", 11)
    pdf.drawString(30, y, "[TOP3 노출수 캠페인]")
    y -= 15
    pdf.setFont("HYSMyeongJo-Medium", 10)
    for _, row in top_impression.iterrows():
        pdf.drawString(30, y, f"- {row['캠페인명']} | 노출 {row['노출수']:,.0f} | CTR(전체) {row['CTR(전체)']:.2f}%")
        y -= 13

    y -= 8
    pdf.setFont("HYSMyeongJo-Medium", 11)
    pdf.drawString(30, y, "[TOP3 CTR(전체) 캠페인]")
    y -= 15
    pdf.setFont("HYSMyeongJo-Medium", 10)
    for _, row in top_ctr.iterrows():
        pdf.drawString(30, y, f"- {row['캠페인명']} | CTR(전체) {row['CTR(전체)']:.2f}% | 노출 {row['노출수']:,.0f}")
        y -= 13

    pdf.save()
    buf.seek(0)
    return buf.read()


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("📊 광고 캠페인 리포트 대시보드")
    st.caption("노출수 0인 날짜는 자동 제외하고 분석합니다.")

    valid_dates = _active_dates(df)
    if len(valid_dates) == 0:
        st.warning("전체 기간에서 노출수가 0보다 큰 데이터가 없습니다.")
        return

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    selected_range = st.date_input(
        "분석 기간(노출수 있는 날짜 기준)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        filtered = df[(df["일자"].dt.date >= start_date) & (df["일자"].dt.date <= end_date)]
    else:
        filtered = df.copy()

    valid_dates_set = set(_active_dates(filtered))
    filtered = filtered[filtered["일자"].isin(valid_dates_set)]

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

    daily = daily_summary(filtered)
    if daily.empty:
        st.warning("필터 결과에서 노출수 0보다 큰 날짜가 없습니다.")
        return

    campaign_raw = campaign_summary(filtered)
    product_device_raw = product_device_summary(filtered)

    totals = {
        "노출수": campaign_raw["노출수"].sum(),
        "총클릭수": campaign_raw["총클릭수"].sum(),
        "동영상조회수": campaign_raw["동영상조회수"].sum(),
        "광고비": campaign_raw["광고비"].sum(),
    }
    campaign_display = _with_share_columns(campaign_raw, totals)
    product_device_display = _with_share_columns(product_device_raw, totals)
    daily_display = _with_share_columns(
        daily,
        {
            "노출수": daily["노출수"].sum(),
            "총클릭수": daily["총클릭수"].sum(),
            "동영상조회수": daily["동영상조회수"].sum(),
            "광고비": daily["광고비"].sum(),
        },
    )

    top_impression = campaign_raw.nlargest(3, "노출수")[["캠페인명", "노출수", "CTR(전체)", "광고일수"]]
    ctr_candidates = campaign_raw[campaign_raw["노출수"] > 0]
    top_ctr = ctr_candidates.nlargest(3, "CTR(전체)")[["캠페인명", "CTR(전체)", "노출수", "광고일수"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 노출수", f"{filtered['노출수'].sum():,.0f}")
    col2.metric("총 클릭수(전체)", f"{filtered['총클릭수'].sum():,.0f}")
    col3.metric("평균 CTR(전체)", f"{(filtered['총클릭수'].sum()/filtered['노출수'].sum()*100):.2f}%")
    col4.metric("총 광고비", f"{filtered['광고비'].sum():,.0f}")

    st.subheader("캠페인 효율 한눈에 보기")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**노출수 TOP3 캠페인**")
        st.dataframe(top_impression, use_container_width=True)
    with c2:
        st.markdown("**CTR(전체) TOP3 캠페인**")
        st.dataframe(top_ctr, use_container_width=True)

    fig_imp_click, fig_ctr = _trend_figures(daily)

    st.subheader("일자별 성과 추이")
    st.plotly_chart(fig_imp_click, use_container_width=True)
    st.plotly_chart(fig_ctr, use_container_width=True)

    st.subheader("캠페인별 요약")
    st.dataframe(
        campaign_display,
        use_container_width=True,
        column_config={
            "시작일": st.column_config.DateColumn("시작일", format="YYYY-MM-DD"),
            "종료일": st.column_config.DateColumn("종료일", format="YYYY-MM-DD"),
            "광고일수": st.column_config.NumberColumn("광고일수", format="%d"),
            "CTR(전체)": st.column_config.NumberColumn("CTR(전체, %)", format="%.2f"),
            "VTR": st.column_config.NumberColumn("VTR(%)", format="%.2f"),
            "eCPM": st.column_config.NumberColumn("eCPM", format="%.2f"),
            "CPC": st.column_config.NumberColumn("CPC", format="%.2f"),
            "CPV": st.column_config.NumberColumn("CPV", format="%.2f"),
        },
    )

    st.subheader("광고상품·기기별 요약")
    st.dataframe(
        product_device_display,
        use_container_width=True,
        column_config={
            "CTR(전체)": st.column_config.NumberColumn("CTR(전체, %)", format="%.2f"),
            "VTR": st.column_config.NumberColumn("VTR(%)", format="%.2f"),
        },
    )

    st.subheader("일자별 상세 테이블")
    st.dataframe(
        daily_display,
        use_container_width=True,
        column_config={
            "일자": st.column_config.DateColumn("일자", format="YYYY-MM-DD"),
            "CTR(전체)": st.column_config.NumberColumn("CTR(전체, %)", format="%.2f"),
            "VTR": st.column_config.NumberColumn("VTR(%)", format="%.2f"),
        },
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="캠페인 요약 CSV 다운로드",
            data=campaign_display.to_csv(index=False).encode("utf-8-sig"),
            file_name="campaign_summary.csv",
            mime="text/csv",
        )
    with d2:
        pdf_bytes = _build_pdf(filtered, daily, top_impression, top_ctr, fig_imp_click, fig_ctr)
        st.download_button(
            label="광고주 보고용 PDF 다운로드",
            data=pdf_bytes,
            file_name="ad_dashboard_report.pdf",
            mime="application/pdf",
        )


def main() -> None:
    st.sidebar.header("파일 업로드")
    upload = st.sidebar.file_uploader("엑셀(.xls/.xlsx) 또는 CSV 파일", type=["xls", "xlsx", "csv"])
    st.sidebar.caption("필수 컬럼: " + ", ".join(REQUIRED_COLUMNS))

    if upload is None:
        st.title("광고 캠페인 리포트 자동화")
        st.write("좌측에서 파일 업로드 후 대시보드와 PDF를 생성하세요.")
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
