"""
광고주 보고용 2-파일 분석 Streamlit 앱
실행: streamlit run app.py
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────
# 0. 페이지 설정 / 스타일
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advertiser Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], section[data-testid="stMain"] > div,
.main .block-container {
    background-color: #F5F9FF !important;
    color: #1A202C !important;
    font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif !important;
}
[data-testid="stSidebar"] { background-color: #1E3A5F !important; }
[data-testid="stSidebar"] * { color: #EAF2FF !important; }
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] label { color: #EAF2FF !important; }
.report-header {
    background: linear-gradient(135deg, #163B72 0%, #2563EB 70%, #60A5FA 100%);
    border-radius: 16px; padding: 24px 28px; margin-bottom: 18px;
    color: white;
}
.report-header h1 { margin: 0 0 6px 0; font-size: 26px; }
.report-header p { margin: 0; color: #DBEAFE; }
.section-title {
    font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.6px;
    color: #2563EB; margin: 18px 0 10px 0; border-left: 4px solid #2563EB; padding-left: 10px;
}
.kpi-wrap {
    background: white; border: 1px solid #DBEAFE; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.06);
}
.kpi-label { font-size: 12px; color: #64748B; font-weight: 600; margin-bottom: 6px; }
.kpi-value { font-size: 24px; font-weight: 700; color: #163B72; }
.note-box {
    background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB;
    border-radius: 10px; padding: 12px 14px; color: #1E3A5F; margin-bottom: 14px;
}
[data-testid="stDataFrame"] {
    background: white !important; border: 1px solid #DBEAFE !important; border-radius: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────
# 1. 상수 / 메타
# ─────────────────────────────────────────────────────
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
DAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
COLORWAY = ["#2563EB", "#60A5FA", "#1D4ED8", "#93C5FD", "#0F172A"]

INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "금융·보험": ["은행", "보험", "카드", "증권", "금융", "저축", "신한", "KB", "하나", "우리", "NH", "IBK"],
    "통신·IT": ["KT", "SKT", "LG유플러스", "SK텔레콤", "통신", "인터넷", "네이버", "카카오", "구글", "애플"],
    "유통·이커머스": ["쇼핑", "마트", "백화점", "이마트", "롯데", "쿠팡", "11번가", "G마켓", "옥션", "SSG"],
    "자동차": ["자동차", "모터스", "현대차", "기아", "BMW", "벤츠", "아우디"],
    "식음료": ["식품", "음료", "맥주", "소주", "커피", "제과", "오뚜기", "농심", "빙그레"],
    "엔터·미디어": ["영화", "게임", "음악", "OTT", "방송", "스튜디오", "넷플릭스"],
    "공공·기관": ["정부", "공단", "공사", "청", "시청", "도청", "교육부"],
    "여행·숙박": ["여행", "항공", "호텔", "리조트", "투어", "야놀자", "여기어때"],
    "패션·뷰티": ["패션", "뷰티", "화장품", "의류", "아모레", "나이키", "아디다스"],
    "의료·헬스": ["병원", "의료", "제약", "헬스", "건강", "약국"],
}

COLUMN_SYNONYMS = {
    "캠페인명": ["캠페인명", "캠페인", "광고명", "Campaign", "campaign"],
    "일자": ["일자", "날짜", "date", "Date"],
    "노출수": ["노출수", "노출", "impression", "impressions"],
    "클릭수": ["클릭수", "클릭", "click", "clicks"],
    "컴패니언배너클릭수": ["컴패니언배너클릭수", "컴패니언 클릭수", "배너클릭수", "companion_clicks"],
    "동영상조회수": ["동영상조회수", "동영상 조회수", "비디오조회수", "영상조회수", "video_views"],
    "방문수": ["방문수", "방문", "세션수", "유입수"],
    "총구매수": ["총구매수", "구매수", "구매", "구매건수", "orders"],
    "총매출액": ["총매출액", "매출액", "매출", "sales", "revenue"],
    "광고상품명_정리": ["광고상품명_정리", "광고상품명", "상품명", "상품"],
    "구분": ["구분", "카테고리"],
    "프로모션별": ["프로모션별", "프로모션", "프로모션명"],
    "광고주": ["광고주", "광고주명", "브랜드"],
}


# ─────────────────────────────────────────────────────
# 2. 데이터 구조
# ─────────────────────────────────────────────────────
@dataclass
class Filters:
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    campaigns: List[str]
    products: List[str]
    categories: List[str]
    industries: List[str]


# ─────────────────────────────────────────────────────
# 3. 공통 유틸
# ─────────────────────────────────────────────────────
def sec(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)



def info_box(message: str) -> None:
    st.markdown(f'<div class="note-box">{message}</div>', unsafe_allow_html=True)



def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-wrap">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def safe_div(num: pd.Series | float, den: pd.Series | float) -> pd.Series | float:
    if isinstance(den, pd.Series):
        den = den.replace(0, np.nan)
    elif den == 0:
        den = np.nan
    result = num / den
    if isinstance(result, pd.Series):
        return result.fillna(0)
    return 0 if pd.isna(result) else result



def normalize_campaign_name(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", text)



def extract_product_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = re.split(r"[>/|\\_\-:\[\]\(\)]+", text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts[-1] if parts else text



def classify_industry(advertiser: object) -> str:
    name = str(advertiser or "").upper()
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        if any(keyword.upper() in name for keyword in keywords):
            return industry
    return "기타"



def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")



def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for canonical, candidates in COLUMN_SYNONYMS.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = canonical
                break
    out = df.rename(columns=rename_map).copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out



def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label}에 필요한 컬럼이 없습니다: {missing}. 현재 컬럼: {list(df.columns)}")



def ensure_non_empty(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label} 데이터가 비어 있습니다. 파일 내용을 확인해주세요.")



def add_campaign_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["캠페인명_원본"] = out["캠페인명"].astype(str).str.strip()
    out["캠페인명"] = out["캠페인명_원본"]
    out["캠페인명_정규화"] = out["캠페인명_원본"].apply(normalize_campaign_name)
    out["상품명_추정값"] = out["캠페인명_원본"].apply(extract_product_token)
    out["상품명_추정값_정규화"] = out["상품명_추정값"].apply(normalize_campaign_name)
    return out



def format_number(value: float) -> str:
    return f"{value:,.0f}"



def format_percent(value: float) -> str:
    return f"{value:.2%}"



def format_currency(value: float) -> str:
    return f"{value:,.0f}"


# ─────────────────────────────────────────────────────
# 4. 파일 로드 / 전처리
# ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_conversion_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def detect_excel_header_row(file_bytes: bytes, keywords: Sequence[str], max_rows: int = 15) -> int:
    preview = pd.read_excel(io.BytesIO(file_bytes), header=None, nrows=max_rows)
    for idx, row in preview.iterrows():
        joined = " ".join(row.fillna("").astype(str).str.strip().tolist())
        if any(keyword in joined for keyword in keywords):
            return idx
    return 0


@st.cache_data(show_spinner=False)
def load_conversion_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    extension = file_name.lower().split(".")[-1]
    try:
        if extension == "csv":
            return pd.read_csv(io.BytesIO(file_bytes))
        if extension in {"xls", "xlsx"}:
            header_row = detect_excel_header_row(file_bytes, ["캠페인", "방문", "구매", "매출"])
            return pd.read_excel(io.BytesIO(file_bytes), header=header_row)
    except Exception as error:
        raise ValueError("전환/매출 파일을 읽지 못했습니다. csv 또는 xls/xlsx 형식을 확인해주세요.") from error
    raise ValueError("전환/매출 파일은 csv 또는 xls/xlsx 형식만 지원합니다.")


@st.cache_data(show_spinner=False)
def load_ad_excel(file_bytes: bytes) -> pd.DataFrame:
    try:
        header_row = detect_excel_header_row(file_bytes, ["캠페인", "노출", "클릭", "일자", "날짜"])
        return pd.read_excel(io.BytesIO(file_bytes), header=header_row)
    except Exception as error:
        raise ValueError("일별 광고 xls/xlsx 파일을 읽지 못했습니다. 파일 형식과 헤더 행을 확인해주세요.") from error



def preprocess_conversion_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(df)
    require_columns(out, ["캠페인명", "방문수", "총구매수", "총매출액"], "전환/매출 CSV")
    ensure_non_empty(out, "전환/매출 CSV")

    for column in ["방문수", "총구매수", "총매출액"]:
        out[column] = clean_numeric_series(out[column]).fillna(0)

    out = add_campaign_helper_columns(out)
    return out[["캠페인명", "캠페인명_원본", "캠페인명_정규화", "상품명_추정값", "상품명_추정값_정규화", "방문수", "총구매수", "총매출액"]].copy()



def preprocess_ad_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = standardize_columns(df)
    require_columns(out, ["캠페인명", "일자", "노출수", "클릭수"], "일별 광고 파일")
    ensure_non_empty(out, "일별 광고 파일")

    if "컴패니언배너클릭수" not in out.columns:
        out["컴패니언배너클릭수"] = 0
    if "동영상조회수" not in out.columns:
        out["동영상조회수"] = 0
    if "광고상품명_정리" not in out.columns:
        out["광고상품명_정리"] = np.nan
    if "구분" not in out.columns:
        out["구분"] = np.nan
    if "프로모션별" not in out.columns:
        out["프로모션별"] = np.nan
    if "광고주" not in out.columns:
        out["광고주"] = np.nan

    out["일자"] = pd.to_datetime(out["일자"], errors="coerce")
    if out["일자"].isna().all():
        raise ValueError("일별 광고 파일의 날짜 컬럼을 datetime으로 변환하지 못했습니다.")

    for column in ["노출수", "클릭수", "컴패니언배너클릭수", "동영상조회수"]:
        out[column] = clean_numeric_series(out[column]).fillna(0)

    out = add_campaign_helper_columns(out)
    out["연도"] = out["일자"].dt.year
    out["월"] = out["일자"].dt.month
    out["연월"] = out["일자"].dt.to_period("M").dt.to_timestamp()
    out["요일"] = out["일자"].dt.weekday.map(DAY_MAP)
    out["총클릭수"] = out["클릭수"] + out["컴패니언배너클릭수"]
    out["CTR"] = safe_div(out["총클릭수"], out["노출수"])
    out["프로모션명"] = out["프로모션별"].astype("string").fillna(out["캠페인명"])
    out["광고상품명_정리"] = out["광고상품명_정리"].astype("string").fillna(out["상품명_추정값"])
    out["구분"] = out["구분"].astype("string").fillna("미분류")
    out["광고주"] = out["광고주"].astype("string").fillna("미분류")
    out["업종"] = out["광고주"].apply(classify_industry)
    ensure_non_empty(out.dropna(subset=["일자", "캠페인명"]), "전처리된 일별 광고")
    return out


# ─────────────────────────────────────────────────────
# 5. 분석 로직
# ─────────────────────────────────────────────────────
def aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, as_index=False, dropna=False)[["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수"]]
        .sum(min_count=1)
        .fillna(0)
    )
    agg["CTR"] = safe_div(agg["총클릭수"], agg["노출수"])
    return agg



def build_lookup_map(df: pd.DataFrame, key: str) -> Dict[str, pd.Series]:
    deduped = df.drop_duplicates(subset=[key], keep="first")
    return {str(row[key]): row for _, row in deduped.iterrows() if str(row[key]).strip()}



def merge_campaign_data(ad_campaign_df: pd.DataFrame, conversion_df: pd.DataFrame) -> pd.DataFrame:
    exact_map = build_lookup_map(conversion_df, "캠페인명")
    normalized_map = build_lookup_map(conversion_df, "캠페인명_정규화")
    product_map = build_lookup_map(conversion_df, "상품명_추정값_정규화")

    records: List[Dict[str, object]] = []
    for _, row in ad_campaign_df.iterrows():
        matched = None
        match_type = "unmatched"
        if row["캠페인명"] in exact_map:
            matched = exact_map[row["캠페인명"]]
            match_type = "exact_match"
        elif row["캠페인명_정규화"] in normalized_map:
            matched = normalized_map[row["캠페인명_정규화"]]
            match_type = "normalized_match"
        elif row["상품명_추정값_정규화"] in product_map:
            matched = product_map[row["상품명_추정값_정규화"]]
            match_type = "product_token_match"

        record = row.to_dict()
        record["match_type"] = match_type
        record["구매데이터_캠페인명"] = matched["캠페인명"] if matched is not None else ""
        record["방문수"] = float(matched["방문수"]) if matched is not None else 0.0
        record["총구매수"] = float(matched["총구매수"]) if matched is not None else 0.0
        record["총매출액"] = float(matched["총매출액"]) if matched is not None else 0.0
        records.append(record)

    out = pd.DataFrame(records)
    out["구매전환율"] = safe_div(out["총구매수"], out["총클릭수"])
    out["클릭당 매출"] = safe_div(out["총매출액"], out["총클릭수"])
    out["CTR"] = safe_div(out["총클릭수"], out["노출수"])
    return out



def build_analysis_frames(ad_df: pd.DataFrame, conversion_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    daily_performance = (
        ad_df.groupby("일자", as_index=False)[["노출수", "총클릭수", "동영상조회수"]]
        .sum()
        .sort_values("일자")
    )
    daily_performance["CTR"] = safe_div(daily_performance["총클릭수"], daily_performance["노출수"])

    campaign_daily = (
        ad_df.groupby(["캠페인명", "일자"], as_index=False)[["노출수", "총클릭수", "동영상조회수"]]
        .sum()
        .sort_values(["캠페인명", "일자"])
    )
    campaign_daily["CTR"] = safe_div(campaign_daily["총클릭수"], campaign_daily["노출수"])

    campaign_agg = (
        ad_df.groupby([
            "캠페인명", "캠페인명_원본", "캠페인명_정규화", "상품명_추정값", "상품명_추정값_정규화",
            "광고상품명_정리", "구분", "프로모션명", "광고주", "업종"
        ], as_index=False)[["노출수", "클릭수", "컴패니언배너클릭수", "총클릭수", "동영상조회수"]]
        .sum()
    )
    campaign_performance = merge_campaign_data(campaign_agg, conversion_df)
    campaign_performance = campaign_performance.sort_values(["총매출액", "총클릭수"], ascending=[False, False]).reset_index(drop=True)

    summary = pd.DataFrame([
        {
            "전체 노출수": campaign_performance["노출수"].sum(),
            "전체 총 클릭수": campaign_performance["총클릭수"].sum(),
            "전체 CTR": safe_div(campaign_performance["총클릭수"].sum(), campaign_performance["노출수"].sum()),
            "전체 동영상조회수": campaign_performance["동영상조회수"].sum(),
            "전체 총구매수": campaign_performance["총구매수"].sum(),
            "전체 총매출액": campaign_performance["총매출액"].sum(),
            "전체 구매전환율": safe_div(campaign_performance["총구매수"].sum(), campaign_performance["총클릭수"].sum()),
            "매칭률": safe_div((campaign_performance["match_type"] != "unmatched").sum(), len(campaign_performance)),
        }
    ])

    return {
        "daily_performance": daily_performance,
        "campaign_daily": campaign_daily,
        "campaign_performance": campaign_performance,
        "summary": summary,
    }



def apply_filters(ad_df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    out = ad_df.copy()
    if filters.date_min is not None:
        out = out[out["일자"] >= filters.date_min]
    if filters.date_max is not None:
        out = out[out["일자"] <= filters.date_max]
    if filters.campaigns:
        out = out[out["캠페인명"].isin(filters.campaigns)]
    if filters.products:
        out = out[out["광고상품명_정리"].isin(filters.products)]
    if filters.categories:
        out = out[out["구분"].isin(filters.categories)]
    if filters.industries:
        out = out[out["업종"].isin(filters.industries)]
    return out


# ─────────────────────────────────────────────────────
# 6. 표현용 테이블 / 차트
# ─────────────────────────────────────────────────────
def build_display_tables(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    summary = frames["summary"].copy()
    summary_display = pd.DataFrame([
        {
            "전체 노출수": format_number(summary.loc[0, "전체 노출수"]),
            "전체 총 클릭수": format_number(summary.loc[0, "전체 총 클릭수"]),
            "전체 CTR": format_percent(summary.loc[0, "전체 CTR"]),
            "전체 동영상조회수": format_number(summary.loc[0, "전체 동영상조회수"]),
            "전체 총구매수": format_number(summary.loc[0, "전체 총구매수"]),
            "전체 총매출액": format_currency(summary.loc[0, "전체 총매출액"]),
            "전체 구매전환율": format_percent(summary.loc[0, "전체 구매전환율"]),
            "매칭률": format_percent(summary.loc[0, "매칭률"]),
        }
    ])

    daily = frames["daily_performance"].copy()
    daily_display = daily.copy()
    daily_display["일자"] = daily_display["일자"].dt.strftime("%Y-%m-%d")
    daily_display["노출수"] = daily_display["노출수"].map(format_number)
    daily_display["총클릭수"] = daily_display["총클릭수"].map(format_number)
    daily_display["CTR"] = daily_display["CTR"].map(format_percent)
    daily_display["동영상조회수"] = daily_display["동영상조회수"].map(format_number)
    daily_display = daily_display.rename(columns={"총클릭수": "총 클릭수"})

    campaign = frames["campaign_performance"].copy()
    campaign_display = campaign[[
        "캠페인명", "상품명_추정값", "match_type", "노출수", "총클릭수", "CTR", "동영상조회수",
        "방문수", "총구매수", "총매출액", "구매전환율", "클릭당 매출"
    ]].copy()
    campaign_display = campaign_display.rename(columns={"총클릭수": "총 클릭수"})
    for column in ["노출수", "총 클릭수", "동영상조회수", "방문수", "총구매수"]:
        campaign_display[column] = campaign_display[column].map(format_number)
    for column in ["CTR", "구매전환율"]:
        campaign_display[column] = campaign_display[column].map(format_percent)
    for column in ["총매출액", "클릭당 매출"]:
        campaign_display[column] = campaign_display[column].map(format_currency)

    return {
        "summary": summary_display,
        "daily": daily_display,
        "campaign": campaign_display,
    }



def build_excel_download(summary: pd.DataFrame, daily: pd.DataFrame, campaign: pd.DataFrame) -> bytes:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            daily.to_excel(writer, sheet_name="daily_performance", index=False)
            campaign.to_excel(writer, sheet_name="campaign_performance", index=False)
        buffer.seek(0)
        return buffer.read()
    except Exception as error:
        raise ValueError("엑셀 다운로드 파일 생성에 실패했습니다. openpyxl 설치 여부를 확인해주세요.") from error



def plot_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=18, color="#163B72")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=COLORWAY,
        font=dict(family="Pretendard, Apple SD Gothic Neo, sans-serif", color="#1F2937"),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(gridcolor="#E5EEF9", linecolor="#D5E3F7"),
        yaxis=dict(gridcolor="#E5EEF9", linecolor="#D5E3F7"),
    )
    return fig



def render_charts(daily_df: pd.DataFrame, campaign_df: pd.DataFrame, top_n: int = 10) -> None:
    sec("시각화")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            daily_df,
            x="일자",
            y=["노출수", "총클릭수"],
            markers=True,
            labels={"value": "수치", "variable": "지표"},
        )
        st.plotly_chart(plot_theme(fig, "일별 노출수 / 총 클릭수 추이"), use_container_width=True)

    with col2:
        fig = px.line(daily_df, x="일자", y="CTR", markers=True)
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(plot_theme(fig, "일별 CTR 추이"), use_container_width=True)

    top_clicks = campaign_df.sort_values("총클릭수", ascending=False).head(top_n)
    top_purchases = campaign_df.sort_values(["총구매수", "총클릭수"], ascending=[False, False]).head(top_n)
    top_cvr = campaign_df.sort_values(["구매전환율", "총클릭수"], ascending=[False, False]).head(top_n)
    top_revenue = campaign_df.sort_values(["총매출액", "총클릭수"], ascending=[False, False]).head(top_n)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        fig = px.bar(top_clicks, x="캠페인명", y="총클릭수", color_discrete_sequence=[COLORWAY[0]])
        st.plotly_chart(plot_theme(fig, f"캠페인별 총 클릭수 TOP{top_n}"), use_container_width=True)
    with row2_col2:
        fig = px.bar(top_purchases, x="캠페인명", y="총구매수", color_discrete_sequence=[COLORWAY[1]])
        st.plotly_chart(plot_theme(fig, f"캠페인별 총구매수 TOP{top_n}"), use_container_width=True)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        fig = px.bar(top_cvr, x="캠페인명", y="구매전환율", color_discrete_sequence=[COLORWAY[2]])
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(plot_theme(fig, f"캠페인별 구매전환율 TOP{top_n}"), use_container_width=True)
    with row3_col2:
        fig = px.bar(top_revenue, x="캠페인명", y="총매출액", color_discrete_sequence=[COLORWAY[0]])
        st.plotly_chart(plot_theme(fig, f"캠페인별 총매출액 TOP{top_n}"), use_container_width=True)


# ─────────────────────────────────────────────────────
# 7. 메인 앱
# ─────────────────────────────────────────────────────
def main() -> None:
    st.markdown(
        """
        <div class="report-header">
            <h1>📊 광고주 보고용 성과 분석</h1>
            <p>전환/매출 CSV + 일별 광고 XLS/XLSX 두 파일을 함께 업로드하면 광고주 보고용 표와 차트를 생성합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 📂 데이터 업로드")
        conversion_file = st.file_uploader("전환/매출 데이터 업로드 (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])
        ad_file = st.file_uploader("일별 광고 데이터 업로드 (XLS, XLSX)", type=["xls", "xlsx"])
        st.markdown("### ⚙️ 보고 설정")
        top_n = st.slider("차트 TOP N", min_value=5, max_value=20, value=10)

    if conversion_file is None or ad_file is None:
        info_box("전환/매출 csv/xls/xlsx 파일과 일별 광고 xls/xlsx 파일을 모두 업로드해주세요.")
        st.warning("두 파일이 모두 업로드되기 전에는 전처리와 분석을 실행하지 않습니다.")
        st.stop()

    try:
        with st.status("분석 준비 중...", expanded=True) as status:
            status.write("파일 로드 중")
            conversion_raw = load_conversion_file(conversion_file.getvalue(), conversion_file.name)
            ad_raw = load_ad_excel(ad_file.getvalue())

            status.write("광고 데이터 전처리 중")
            ad_df = preprocess_ad_daily(ad_raw)

            status.write("전환 데이터 전처리 중")
            conversion_df = preprocess_conversion_csv(conversion_raw)

            if ad_df.empty or conversion_df.empty:
                raise ValueError("업로드한 파일 중 하나가 비어 있어 분석을 진행할 수 없습니다.")

            with st.sidebar:
                st.markdown("### 🔎 필터")
                date_min = ad_df["일자"].min()
                date_max = ad_df["일자"].max()
                date_range = st.date_input(
                    "날짜 범위",
                    value=(date_min.date(), date_max.date()),
                    min_value=date_min.date(),
                    max_value=date_max.date(),
                )
                campaigns = sorted(ad_df["캠페인명"].dropna().unique().tolist())
                products = sorted(ad_df["광고상품명_정리"].dropna().astype(str).unique().tolist())
                categories = sorted(ad_df["구분"].dropna().astype(str).unique().tolist())
                industries = sorted(ad_df["업종"].dropna().astype(str).unique().tolist())
                selected_campaigns = st.multiselect("캠페인", campaigns, default=campaigns)
                selected_products = st.multiselect("광고상품", products, default=products)
                selected_categories = st.multiselect("구분", categories, default=categories)
                selected_industries = st.multiselect("업종", industries, default=industries)

            filters = Filters(
                date_min=pd.to_datetime(date_range[0]) if len(date_range) > 0 else None,
                date_max=pd.to_datetime(date_range[1]) if len(date_range) > 1 else None,
                campaigns=selected_campaigns,
                products=selected_products,
                categories=selected_categories,
                industries=selected_industries,
            )
            filtered_ad_df = apply_filters(ad_df, filters)
            ensure_non_empty(filtered_ad_df, "필터링된 광고 데이터")

            status.write("캠페인 매칭 중")
            frames = build_analysis_frames(filtered_ad_df, conversion_df)

            status.write("시각화 생성 중")
            status.update(label="분석 완료", state="complete", expanded=False)
    except ValueError as error:
        st.error(f"데이터 검증 오류: {error}")
        st.stop()
    except Exception as error:
        st.error(f"파일 처리 중 오류가 발생했습니다: {error}")
        st.stop()

    display_frames = build_display_tables(frames)
    summary = frames["summary"].iloc[0]
    campaign_perf = frames["campaign_performance"]
    daily_perf = frames["daily_performance"]

    info_box(
        f"분석 기간: <b>{daily_perf['일자'].min().date()}</b> ~ <b>{daily_perf['일자'].max().date()}</b> · "
        f"캠페인 {campaign_perf['캠페인명'].nunique():,}개 · 매칭률 {format_percent(summary['매칭률'])}"
    )

    sec("전체 Summary")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        kpi_card("전체 노출수", format_number(summary["전체 노출수"]))
    with kpi_cols[1]:
        kpi_card("전체 총 클릭수", format_number(summary["전체 총 클릭수"]))
    with kpi_cols[2]:
        kpi_card("전체 CTR", format_percent(summary["전체 CTR"]))
    with kpi_cols[3]:
        kpi_card("전체 구매전환율", format_percent(summary["전체 구매전환율"]))

    kpi_cols2 = st.columns(4)
    with kpi_cols2[0]:
        kpi_card("전체 동영상조회수", format_number(summary["전체 동영상조회수"]))
    with kpi_cols2[1]:
        kpi_card("전체 총구매수", format_number(summary["전체 총구매수"]))
    with kpi_cols2[2]:
        kpi_card("전체 총매출액", format_currency(summary["전체 총매출액"]))
    with kpi_cols2[3]:
        kpi_card("매칭률", format_percent(summary["매칭률"]))

    tabs = st.tabs(["요약", "캠페인별 성과", "일별 성과", "매칭 점검"])
    with tabs[0]:
        sec("Summary 테이블")
        st.dataframe(display_frames["summary"], use_container_width=True, hide_index=True)
        render_charts(daily_perf, campaign_perf, top_n=top_n)

    with tabs[1]:
        sec("캠페인별 성과 테이블")
        st.dataframe(display_frames["campaign"], use_container_width=True, hide_index=True)

    with tabs[2]:
        sec("일별 성과 테이블")
        st.dataframe(display_frames["daily"], use_container_width=True, hide_index=True)
        sec("캠페인별 일자 추이 샘플")
        campaign_daily = frames["campaign_daily"].copy()
        campaign_daily["일자"] = campaign_daily["일자"].dt.strftime("%Y-%m-%d")
        st.dataframe(campaign_daily.head(100), use_container_width=True, hide_index=True)

    with tabs[3]:
        sec("최종 캠페인 매칭 결과 샘플")
        match_sample = campaign_perf[["캠페인명", "구매데이터_캠페인명", "상품명_추정값", "match_type", "총클릭수", "총구매수", "총매출액"]].head(30).copy()
        match_sample = match_sample.rename(columns={"총클릭수": "총 클릭수"})
        st.dataframe(match_sample, use_container_width=True, hide_index=True)
        unmatched = campaign_perf[campaign_perf["match_type"] == "unmatched"][["캠페인명", "상품명_추정값", "총클릭수"]].copy()
        unmatched = unmatched.rename(columns={"총클릭수": "총 클릭수"})
        if unmatched.empty:
            st.success("모든 캠페인이 구매/매출 데이터와 매칭되었습니다.")
        else:
            st.warning(f"매칭되지 않은 캠페인 {len(unmatched):,}개가 있습니다. 아래 샘플을 확인해주세요.")
            st.dataframe(unmatched.head(30), use_container_width=True, hide_index=True)

    excel_bytes = build_excel_download(display_frames["summary"], display_frames["daily"], display_frames["campaign"])
    st.download_button(
        label="📥 결과 엑셀 다운로드",
        data=excel_bytes,
        file_name="advertiser_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
