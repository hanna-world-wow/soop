import io
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, Table, TableStyle

# 폰트 등록 (한글 깨짐 방지)
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))

# --- [1. 기본 설정 및 스타일] ---
st.set_page_config(page_title="광고 캠페인 보고서", layout="wide")


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stMetric {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #eee;
        }
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            background-color: white;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
PERCENT_COLUMNS = {"CTR(전체)", "VTR"}
AD_REQUIRED_COLUMNS = ["광고명", "캠페인명", "노출수", "클릭수", "CTR", "CTR(전체)"]
AD_NUMERIC_COLUMNS = ["노출수", "클릭수", "컴패니언배너클릭수", "광고비"]


# --- [2. 데이터 처리 및 유틸리티] ---
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


def _format_number(value: float) -> str:
    text = f"{float(value):,.2f}"
    return text.rstrip("0").rstrip(".")


def _format_percent(value: float) -> str:
    text = f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _parse_campaign_info(campaign_name: str) -> tuple[str, str, str]:
    parts = str(campaign_name).split("_")
    advertiser = parts[2] if len(parts) >= 3 else "미분류"
    product = parts[3] if len(parts) >= 4 else "미분류"
    campaign_upper = str(campaign_name).upper()
    if "MO" in product.upper() or "모바일" in campaign_upper or "_MO" in campaign_upper:
        device = "MO"
    elif "PC" in product.upper() or "pc" in campaign_upper.lower() or "_PC" in campaign_upper:
        device = "PC"
    else:
        device = "공통"
    return advertiser, product, device


def _report_title(df: pd.DataFrame) -> str:
    advertisers = [str(v) for v in df["광고주명"].dropna().unique() if str(v).strip()]
    if not advertisers:
        advertiser_text = "미분류"
    elif len(advertisers) == 1:
        advertiser_text = advertisers[0]
    else:
        advertiser_text = f"{advertisers[0]} 외 {len(advertisers) - 1}"
    return f"[{advertiser_text}] 광고 캠페인 통합 리포트"


def load_report(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    excel_bytes = io.BytesIO(file_bytes)
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(excel_bytes)
    else:
        df = pd.read_excel(excel_bytes, header=1)
        df = _normalize_columns(df)
        if _validate_columns(df, REQUIRED_COLUMNS):
            excel_bytes.seek(0)
            df = pd.read_excel(excel_bytes, header=0)

    df = _normalize_columns(df)
    missing = _validate_columns(df, ["일자", "캠페인명", *NUMERIC_COLUMNS])
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(missing)}")

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].apply(_to_number)

    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
    df = df.dropna(subset=["일자"]).sort_values("일자")
    if df.empty:
        raise ValueError("유효한 일자 데이터가 없습니다.")

    parsed = df["캠페인명"].apply(_parse_campaign_info)
    df["광고주명"], df["광고상품"], df["기기구분"] = zip(*parsed)
    df["총클릭수"] = df["클릭수"] + df["컴패니언배너클릭수"]

    # 재계산 컬럼들
    df["CTR(전체_재계산)"] = (df["총클릭수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["VTR(재계산)"] = (df["동영상조회수"] / df["노출수"].replace(0, pd.NA) * 100).fillna(0)
    df["CPC(재계산)"] = (df["광고비"] / df["총클릭수"].replace(0, pd.NA)).fillna(0)
    df["CPV(재계산)"] = (df["광고비"] / df["동영상조회수"].replace(0, pd.NA)).fillna(0)
    df["eCPM(재계산)"] = (df["광고비"] / df["노출수"].replace(0, pd.NA) * 1000).fillna(0)
    return df

def _parse_asset_name(ad_name: str) -> str:
    parts = str(ad_name).split("_")
    return parts[-1] if parts else str(ad_name)


def load_ad_report(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    excel_bytes = io.BytesIO(file_bytes)
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(excel_bytes)
    else:
        df = pd.read_excel(excel_bytes, header=1)

    df = _normalize_columns(df)
    missing = _validate_columns(df, AD_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"광고 단위 파일 필수 컬럼 누락: {', '.join(missing)}")

    for col in AD_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    # CTR 문자열 비율 정규화
    for col in ["CTR", "CTR(전체)"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    df["애셋구분"] = df["광고명"].apply(_parse_asset_name)
    grouped = (
        df.groupby(["캠페인명", "광고상품명", "애셋구분", "광고명"], as_index=False)
        .agg(
            노출수=("노출수", "sum"),
            클릭수=("클릭수", "sum"),
            총클릭수=("클릭수", "sum"),
        )
    )
    grouped["CTR(전체)"] = (grouped["총클릭수"] / grouped["노출수"].replace(0, pd.NA) * 100).fillna(0)

    # 같은 캠페인 내 광고 2개 이상만 분석 대상
    counts = grouped.groupby("캠페인명")["광고명"].nunique().reset_index(name="광고수")
    valid_campaigns = counts[counts["광고수"] > 1]["캠페인명"]
    grouped = grouped[grouped["캠페인명"].isin(valid_campaigns)].copy()

    if grouped.empty:
        raise ValueError("광고가 2개 이상인 캠페인이 없습니다. 광고 분석 대상이 없습니다.")

    return grouped.sort_values(["캠페인명", "노출수"], ascending=[True, False])



# --- [3. 집계 및 요약 함수] ---
def _active_dates(df: pd.DataFrame) -> pd.Index:
    daily = df.groupby("일자", as_index=False).agg(노출수=("노출수", "sum"))
    return daily[daily["노출수"] > 0]["일자"]


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["광고주명", "광고상품", "기기구분", "광고계정", "캠페인명"]
    grouped = df.groupby(keys, as_index=False).agg(
        노출수=("노출수", "sum"),
        클릭수=("클릭수", "sum"),
        컴패니언배너클릭수=("컴패니언배너클릭수", "sum"),
        동영상조회수=("동영상조회수", "sum"),
        광고비=("광고비", "sum"),
        시작일=("일자", "min"),
        종료일=("일자", "max"),
    )
    active_days = (
        df[df["노출수"] > 0]
        .groupby(keys, as_index=False)["일자"]
        .nunique()
        .rename(columns={"일자": "광고일수"})
    )
    grouped = grouped.merge(active_days, on=keys, how="left")
    grouped["광고일수"] = grouped["광고일수"].fillna(0).astype(int)
    grouped["총클릭수"] = grouped["클릭수"] + grouped["컴패니언배너클릭수"]
    grouped["CTR(전체)"] = (grouped["총클릭수"] / grouped["노출수"].replace(0, pd.NA) * 100).fillna(0)
    grouped["VTR"] = (grouped["동영상조회수"] / grouped["노출수"].replace(0, pd.NA) * 100).fillna(0)
    return grouped


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("일자", as_index=False).agg(
        노출수=("노출수", "sum"),
        클릭수=("클릭수", "sum"),
        컴패니언배너클릭수=("컴패니언배너클릭수", "sum"),
        동영상조회수=("동영상조회수", "sum"),
        광고비=("광고비", "sum"),
    )
    daily = daily[daily["노출수"] > 0].copy()
    daily["총클릭수"] = daily["클릭수"] + daily["컴패니언배너클릭수"]
    daily["CTR(전체)"] = (daily["총클릭수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)
    daily["VTR"] = (daily["동영상조회수"] / daily["노출수"].replace(0, pd.NA) * 100).fillna(0)
    return daily


def _with_share_only_columns(table: pd.DataFrame, total_map: dict[str, float]) -> pd.DataFrame:
    display = table.copy()
    for col in ["노출수", "총클릭수", "동영상조회수", "광고비"]:
        denom = total_map.get(col, 0)
        share = (display[col] / denom * 100).fillna(0) if denom else pd.Series([0.0] * len(display), index=display.index)
        display[col] = display[col].map(_format_number) + share.map(lambda x: f" ({_format_number(x)}%)")

    if "CTR(전체)" in display.columns:
        display["CTR(전체)"] = display["CTR(전체)"].map(_format_percent)
    if "VTR" in display.columns:
        display["VTR"] = display["VTR"].map(_format_percent)
    if "광고일수" in display.columns:
        display["광고일수"] = display["광고일수"].map(_format_number)
    if "시작일" in display.columns:
        display["시작일"] = pd.to_datetime(display["시작일"]).dt.strftime("%Y-%m-%d")
    if "종료일" in display.columns:
        display["종료일"] = pd.to_datetime(display["종료일"]).dt.strftime("%Y-%m-%d")

    return display


def _append_campaign_total_row(display: pd.DataFrame, raw: pd.DataFrame, totals: dict[str, float]) -> pd.DataFrame:
    ctr = (totals["총클릭수"] / totals["노출수"] * 100) if totals["노출수"] else 0.0
    vtr = (totals["동영상조회수"] / totals["노출수"] * 100) if totals["노출수"] else 0.0
    total_row = {
        "광고상품": "-",
        "기기구분": "-",
        "캠페인명": "합계",
        "노출수": f"{_format_number(totals['노출수'])} (100%)",
        "클릭수": _format_number(raw["클릭수"].sum()),
        "컴패니언배너클릭수": _format_number(raw["컴패니언배너클릭수"].sum()),
        "동영상조회수": f"{_format_number(totals['동영상조회수'])} (100%)",
        "광고비": f"{_format_number(totals['광고비'])} (100%)",
        "시작일": pd.to_datetime(raw["시작일"]).min().strftime("%Y-%m-%d"),
        "종료일": pd.to_datetime(raw["종료일"]).max().strftime("%Y-%m-%d"),
        "광고일수": _format_number(raw["광고일수"].sum()),
        "총클릭수": f"{_format_number(totals['총클릭수'])} (100%)",
        "CTR(전체)": _format_percent(ctr),
        "VTR": _format_percent(vtr),
    }
    return pd.concat([display, pd.DataFrame([total_row])], ignore_index=True)


def _append_daily_total_row(display: pd.DataFrame, raw: pd.DataFrame, totals: dict[str, float]) -> pd.DataFrame:
    ctr = (totals["총클릭수"] / totals["노출수"] * 100) if totals["노출수"] else 0.0
    vtr = (totals["동영상조회수"] / totals["노출수"] * 100) if totals["노출수"] else 0.0
    total_row = {
        "일자": "합계",
        "노출수": f"{_format_number(totals['노출수'])} (100%)",
        "클릭수": _format_number(raw["클릭수"].sum()),
        "컴패니언배너클릭수": _format_number(raw["컴패니언배너클릭수"].sum()),
        "동영상조회수": f"{_format_number(totals['동영상조회수'])} (100%)",
        "광고비": f"{_format_number(totals['광고비'])} (100%)",
        "총클릭수": f"{_format_number(totals['총클릭수'])} (100%)",
        "CTR(전체)": _format_percent(ctr),
        "VTR": _format_percent(vtr),
    }
    return pd.concat([display, pd.DataFrame([total_row])], ignore_index=True)


# --- [4. 시각화 및 PDF] ---
def _trend_figure(daily: pd.DataFrame, selected_metrics: list[str], title: str) -> go.Figure:
    fig = go.Figure()

    for idx, metric in enumerate(selected_metrics):
        axis = "y" if idx == 0 else "y2"
        fig.add_trace(go.Scatter(x=daily["일자"], y=daily[metric], mode="lines+markers", name=metric, yaxis=axis))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(title="일자"),
        yaxis=dict(title=selected_metrics[0], tickformat=",.2f", rangemode="tozero"),
        legend=dict(orientation="h"),
    )

    if selected_metrics[0] in PERCENT_COLUMNS:
        fig.update_yaxes(ticksuffix="%")

    if len(selected_metrics) == 2:
        fig.update_layout(yaxis2=dict(title=selected_metrics[1], overlaying="y", side="right", tickformat=",.2f", showgrid=False, rangemode="tozero"))
        if selected_metrics[1] in PERCENT_COLUMNS:
            fig.update_layout(yaxis2_ticksuffix="%")

    return fig


def _get_pdf_style() -> ParagraphStyle:
    styles = getSampleStyleSheet()
    _ = styles["Normal"]
    return ParagraphStyle(
        "CellStyle",
        fontName="HYSMyeongJo-Medium",
        fontSize=7,
        leading=9,
        wordWrap="CJK",
        alignment=1,
    )


def _draw_df_table(c: canvas.Canvas, df: pd.DataFrame, x: float, y_top: float, total_width: float, rows: int, title: str) -> float:
    c.setFont("HYSMyeongJo-Medium", 11)
    c.drawString(x, y_top, title)

    sample = df.head(rows).copy()
    if sample.empty:
        return y_top - 20

    cell_style = _get_pdf_style()
    data = []
    header = [Paragraph(f"<b>{str(col)}</b>", cell_style) for col in sample.columns]
    data.append(header)

    for _, row in sample.iterrows():
        data.append([Paragraph(str(val), cell_style) for val in row])

    col_count = len(sample.columns)
    if "캠페인명" in sample.columns and col_count > 1:
        other_width = (total_width * 0.7) / (col_count - 1)
        col_widths = [total_width * 0.3 if col == "캠페인명" else other_width for col in sample.columns]
    else:
        col_widths = [total_width / col_count] * col_count

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    w, h = table.wrap(total_width, 1000)
    table.drawOn(c, x, y_top - h - 10)
    return y_top - h - 30
def _build_pdf(
    filtered: pd.DataFrame,
    daily: pd.DataFrame,
    volume_fig: go.Figure,
    efficiency_fig: go.Figure,
    top_imp: pd.DataFrame,
    top_ctr: pd.DataFrame,
    campaign_display: pd.DataFrame,
    daily_display: pd.DataFrame,
) -> bytes:
    if daily.empty:
        raise ValueError("PDF 생성용 일자 데이터가 없습니다. 필터를 변경해 주세요.")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    width, height = landscape(A4)

    c.setFillColorRGB(0.13, 0.19, 0.25)
    c.rect(0, height - 60, width, 60, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("HYSMyeongJo-Medium", 18)
    c.drawString(30, height - 38, "AD CAMPAIGN PERFORMANCE REPORT")

    c.setFillColorRGB(0, 0, 0)
    c.setFont("HYSMyeongJo-Medium", 10)
    c.drawString(30, height - 85, f"분석 기간: {daily['일자'].min().strftime('%Y-%m-%d')} ~ {daily['일자'].max().strftime('%Y-%m-%d')}")

    c.setFillColorRGB(0.96, 0.97, 0.98)
    c.roundRect(30, height - 135, width - 60, 40, 5, fill=1, stroke=0)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    avg_ctr = (filtered['총클릭수'].sum() / filtered['노출수'].sum() * 100) if filtered['노출수'].sum() else 0
    summary_text = (
        f"총 노출: {_format_number(filtered['노출수'].sum())}   |   "
        f"총 클릭: {_format_number(filtered['총클릭수'].sum())}   |   "
        f"평균 CTR: {_format_percent(avg_ctr)}   |   "
        f"총 광고비: ₩{_format_number(filtered['광고비'].sum())}"
    )
    c.drawCentredString(width / 2, height - 120, summary_text)

    vol_bytes = volume_fig.to_image(format="png", width=800, height=350, scale=2)
    eff_bytes = efficiency_fig.to_image(format="png", width=800, height=350, scale=2)
    c.drawImage(ImageReader(io.BytesIO(vol_bytes)), 30, height - 330, width=380, height=180)
    c.drawImage(ImageReader(io.BytesIO(eff_bytes)), 430, height - 330, width=380, height=180)

    y_cursor = height - 355
    _draw_df_table(c, top_imp, 30, y_cursor, 380, rows=5, title="노출수 TOP 5")
    _draw_df_table(c, top_ctr, 430, y_cursor, 380, rows=5, title="CTR TOP 5")

    c.showPage()
    y_detail = height - 40
    y_detail = _draw_df_table(c, campaign_display, 30, y_detail, width - 60, rows=15, title="캠페인 성과 상세 (상위 15개)")
    _draw_df_table(c, daily_display, 30, y_detail, width - 60, rows=15, title="일자별 상세 성과 (상위 15개)")

    c.save()
    buf.seek(0)
    return buf.read()


# --- [5. 메인 렌더링 함수] ---
def render_dashboard(df: pd.DataFrame) -> None:
    apply_custom_style()
    st.title(_report_title(df))

    with st.expander("🔍 상세 필터 및 기간 설정", expanded=False):
        f1, f2 = st.columns(2)
        v_dates = _active_dates(df)
        if v_dates.empty:
            v_dates = df["일자"]

        with f1:
            s_range = st.date_input("분석 기간", value=(v_dates.min().date(), v_dates.max().date()))

        if isinstance(s_range, tuple) and len(s_range) == 2:
            filtered = df[(df["일자"].dt.date >= s_range[0]) & (df["일자"].dt.date <= s_range[1])]
        else:
            filtered = df.copy()

        with f2:
            sel_campaigns = st.multiselect("캠페인 필터", sorted(filtered["캠페인명"].unique()))
            if sel_campaigns:
                filtered = filtered[filtered["캠페인명"].isin(sel_campaigns)]

    if filtered.empty:
        st.warning("선택된 조건에 맞는 데이터가 없습니다.")
        return

    daily = daily_summary(filtered)
    campaign_raw = campaign_summary(filtered)

    totals = {
        "노출수": filtered["노출수"].sum(),
        "총클릭수": filtered["총클릭수"].sum(),
        "동영상조회수": filtered["동영상조회수"].sum(),
        "광고비": filtered["광고비"].sum(),
    }

    campaign_display_raw = campaign_raw.drop(columns=["광고계정", "광고주명"], errors="ignore")
    campaign_display = _with_share_only_columns(campaign_display_raw, totals)
    campaign_display = _append_campaign_total_row(campaign_display, campaign_raw, totals)

    daily_for_display = daily.copy()
    daily_for_display["일자"] = daily_for_display["일자"].dt.strftime("%Y-%m-%d")
    daily_display = _with_share_only_columns(daily_for_display, totals)
    daily_display = _append_daily_total_row(daily_display, daily, totals)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 노출수", _format_number(totals["노출수"]))
    m2.metric("총 클릭수", _format_number(totals["총클릭수"]))
    m3.metric("평균 CTR", _format_percent((totals["총클릭수"] / totals["노출수"] * 100) if totals["노출수"] else 0.0))
    m4.metric("총 광고비", _format_number(totals["광고비"]))

    st.markdown("### 📊 성과 추이")
    volume_selected = st.multiselect(
        "노출/클릭/조회 추이 지표 (최대 2개)",
        options=["노출수", "총클릭수", "동영상조회수"],
        default=["노출수", "총클릭수"],
        max_selections=2,
    )
    if not volume_selected:
        volume_selected = ["노출수"]

    efficiency_selected = st.multiselect(
        "효율 지표",
        options=["CTR(전체)", "VTR"],
        default=["CTR(전체)", "VTR"],
        max_selections=2,
    )
    if not efficiency_selected:
        efficiency_selected = ["CTR(전체)"]

    volume_fig = _trend_figure(daily, volume_selected, "<b>일자별 노출/총클릭/동영상조회 추이</b>")
    efficiency_fig = _trend_figure(daily, efficiency_selected, "<b>광고 효율 추이 (CTR / VTR)</b>")

    st.plotly_chart(volume_fig, use_container_width=True)
    st.plotly_chart(efficiency_fig, use_container_width=True)

    st.markdown("### 🏆 캠페인 순위")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 노출수 TOP 5")
        top_imp = campaign_raw.nlargest(5, "노출수")[["캠페인명", "노출수", "CTR(전체)"]].copy().reset_index(drop=True)
        top_imp.index = top_imp.index + 1
        top_imp["노출수"] = top_imp["노출수"].map(_format_number)
        top_imp["CTR(전체)"] = top_imp["CTR(전체)"].map(_format_percent)
        st.dataframe(top_imp, use_container_width=True)
    with c2:
        st.markdown("##### CTR TOP 5")
        top_ctr = campaign_raw.nlargest(5, "CTR(전체)")[["캠페인명", "CTR(전체)", "노출수"]].copy().reset_index(drop=True)
        top_ctr.index = top_ctr.index + 1
        top_ctr["CTR(전체)"] = top_ctr["CTR(전체)"].map(_format_percent)
        top_ctr["노출수"] = top_ctr["노출수"].map(_format_number)
        st.dataframe(top_ctr, use_container_width=True)

    st.markdown("### 📑 상세 데이터")
    st.markdown("##### 캠페인 성과 테이블 (비중 포함)")
    st.dataframe(campaign_display, use_container_width=True)
    st.markdown("##### 일자별 상세 테이블")
    st.dataframe(daily_display, use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("📥 리포트 추출")
    st.sidebar.download_button(
        "엑셀/CSV 추출",
        campaign_display.to_csv(index=False).encode("utf-8-sig"),
        "campaign_summary.csv",
    )

    try:
        pdf_bytes = _build_pdf(filtered, daily, volume_fig, efficiency_fig, top_imp, top_ctr, campaign_display, daily_display)
        st.sidebar.download_button("광고주용 PDF 추출", pdf_bytes, "Ad_Performance_Report.pdf", "application/pdf")
    except Exception as exc:
        st.sidebar.warning(f"PDF 생성 실패: {exc}")


def render_ad_analysis(ad_df: pd.DataFrame) -> None:
    st.subheader("🧪 광고 분석")
    st.caption("같은 캠페인 내 애셋(광고명 마지막 토큰) 성과 비교")

    campaigns = sorted(ad_df["캠페인명"].unique())
    selected_campaign = st.selectbox("캠페인 선택", campaigns)
    campaign_df = ad_df[ad_df["캠페인명"] == selected_campaign].copy()

    placements = sorted([p for p in campaign_df["광고상품명"].dropna().unique() if str(p).strip()])
    if placements:
        selected_placement = st.selectbox("지면(광고상품명) 선택", placements)
        campaign_df = campaign_df[campaign_df["광고상품명"] == selected_placement].copy()

    if campaign_df["광고명"].nunique() <= 1:
        st.info("선택 조건에서 비교할 광고가 2개 미만입니다.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(x=campaign_df["애셋구분"], y=campaign_df["노출수"], name="노출수"))
    fig.add_trace(go.Bar(x=campaign_df["애셋구분"], y=campaign_df["클릭수"], name="클릭수"))
    fig.add_trace(go.Scatter(x=campaign_df["애셋구분"], y=campaign_df["CTR(전체)"], mode="lines+markers", name="CTR(전체)", yaxis="y2"))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title="<b>애셋별 노출/클릭/CTR 비교</b>",
        xaxis=dict(title="애셋"),
        yaxis=dict(title="노출수 / 클릭수", tickformat=",.2f", rangemode="tozero"),
        yaxis2=dict(title="CTR(전체)", overlaying="y", side="right", ticksuffix="%", rangemode="tozero"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    display = campaign_df[["애셋구분", "광고명", "노출수", "클릭수", "CTR(전체)"]].copy()
    display = display.sort_values("노출수", ascending=False).reset_index(drop=True)
    display.index = display.index + 1
    display["노출수"] = display["노출수"].map(_format_number)
    display["클릭수"] = display["클릭수"].map(_format_number)
    display["CTR(전체)"] = display["CTR(전체)"].map(_format_percent)
    st.dataframe(display, use_container_width=True)


def main() -> None:
    st.sidebar.header("📂 리포트 업로드")
    campaign_upload = st.sidebar.file_uploader("캠페인 리포트 파일", type=["xlsx", "xls", "csv"], key="campaign_upload")
    ad_upload = st.sidebar.file_uploader("광고 단위 리포트 파일", type=["xlsx", "xls", "csv"], key="ad_upload")

    tab_campaign, tab_ad = st.tabs(["📈 캠페인 통합 리포트", "🧪 광고 분석"])

    with tab_campaign:
        if campaign_upload:
            try:
                df = load_report(campaign_upload.getvalue(), campaign_upload.name)
                render_dashboard(df)
            except Exception as e:
                st.error(f"캠페인 리포트 처리 중 오류 발생: {e}")
        else:
            st.info("사이드바에서 캠페인 리포트 파일을 업로드해 주세요.")

    with tab_ad:
        if ad_upload:
            try:
                ad_df = load_ad_report(ad_upload.getvalue(), ad_upload.name)
                render_ad_analysis(ad_df)
            except Exception as e:
                st.error(f"광고 단위 리포트 처리 중 오류 발생: {e}")
        else:
            st.info("사이드바에서 광고 단위 리포트 파일을 업로드해 주세요.")


if __name__ == "__main__":
    main()
