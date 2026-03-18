from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 사용자 설정값
# ============================================================
@dataclass
class Config:
    csv_path: str = '캠페인_20260308-20260318_숲토어_kook428.csv'
    xls_path: str = 'soop_adaccount_report_260312_260318.xls'
    output_dir: str = 'output'
    output_excel: str = 'advertiser_report.xlsx'
    top_n: int = 10
    figure_dpi: int = 150


# ============================================================
# 공통 유틸
# ============================================================
def log(message: str) -> None:
    print(f'[LOG] {message}')


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif denominator == 0:
        denominator = np.nan
    result = numerator / denominator
    if isinstance(result, pd.Series):
        return result.fillna(0)
    return 0 if pd.isna(result) else result


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace(r'\s+', '', regex=True)
        .replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    )
    return pd.to_numeric(cleaned, errors='coerce').fillna(0)


def normalize_campaign_name(value: object) -> str:
    text = str(value or '').strip()
    lowered = text.lower()
    normalized = re.sub(r'[^0-9a-z가-힣]+', '', lowered)
    return normalized


def extract_product_token(value: object) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    candidates = re.split(r'[>/|\\_\-:\[\]\(\)]+', text)
    candidates = [item.strip() for item in candidates if item.strip()]
    if not candidates:
        return ''
    return candidates[-1]


DISPLAY_NAME_CANDIDATES = ['캠페인명', '캠페인', '광고명', '광고', 'Campaign', 'campaign']
DATE_CANDIDATES = ['일자', '날짜', 'date', 'Date']
IMP_CANDIDATES = ['노출수', '노출', 'impression', 'impressions']
CLICK_CANDIDATES = ['클릭수', '클릭', 'click', 'clicks']
COMPANION_CLICK_CANDIDATES = ['컴패니언배너클릭수', '컴패니언 클릭수', '배너클릭수', 'companion_clicks']
VIDEO_VIEW_CANDIDATES = ['동영상조회수', '비디오조회수', '영상조회수', 'video_views', '동영상 조회수']
VISIT_CANDIDATES = ['방문수', '방문', '세션수', '유입수']
PURCHASE_CANDIDATES = ['총구매수', '구매수', '구매', 'orders', '구매건수']
REVENUE_CANDIDATES = ['총매출액', '매출액', '매출', 'revenue', 'sales']


CANONICAL_COLUMNS = {
    '캠페인명': DISPLAY_NAME_CANDIDATES,
    '일자': DATE_CANDIDATES,
    '노출수': IMP_CANDIDATES,
    '클릭수': CLICK_CANDIDATES,
    '컴패니언배너클릭수': COMPANION_CLICK_CANDIDATES,
    '동영상조회수': VIDEO_VIEW_CANDIDATES,
    '방문수': VISIT_CANDIDATES,
    '총구매수': PURCHASE_CANDIDATES,
    '총매출액': REVENUE_CANDIDATES,
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for canonical, candidates in CANONICAL_COLUMNS.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = canonical
                break
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, required: Sequence[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f'{dataset_name} 파일에 필요한 컬럼이 없습니다: {missing}. '\
            f'현재 컬럼: {list(df.columns)}'
        )


# ============================================================
# 데이터 로딩/전처리
# ============================================================
def detect_excel_header_row(path: str, max_rows: int = 15) -> int:
    preview = pd.read_excel(path, header=None, nrows=max_rows)
    for idx, row in preview.iterrows():
        row_values = row.fillna('').astype(str).str.strip().tolist()
        joined = ' '.join(row_values)
        if any(keyword in joined for keyword in ['캠페인', '노출', '클릭', '일자', '날짜']):
            return idx
    return 0



def preprocess_campaign_columns(df: pd.DataFrame, campaign_column: str = '캠페인명') -> pd.DataFrame:
    out = df.copy()
    out['캠페인명_원본'] = out[campaign_column].astype(str).fillna('').str.strip()
    out['캠페인명'] = out['캠페인명_원본']
    out['캠페인명_정규화'] = out['캠페인명_원본'].apply(normalize_campaign_name)
    out['상품명_추정값'] = out['캠페인명_원본'].apply(extract_product_token)
    out['상품명_추정값_정규화'] = out['상품명_추정값'].apply(normalize_campaign_name)
    return out



def load_xls_daily_data(path: str) -> pd.DataFrame:
    log('XLS 일별 광고 데이터 로딩 중...')
    header_row = detect_excel_header_row(path)
    log(f'XLS 헤더 행 감지 완료: {header_row + 1}행')
    df = pd.read_excel(path, header=header_row)
    df.columns = [str(column).strip() for column in df.columns]
    df = standardize_columns(df)
    require_columns(df, ['캠페인명', '일자', '노출수', '클릭수'], 'XLS')

    if '컴패니언배너클릭수' not in df.columns:
        df['컴패니언배너클릭수'] = 0
    if '동영상조회수' not in df.columns:
        df['동영상조회수'] = 0

    numeric_columns = ['노출수', '클릭수', '컴패니언배너클릭수', '동영상조회수']
    for column in numeric_columns:
        df[column] = clean_numeric_series(df[column])

    df['일자'] = pd.to_datetime(df['일자'], errors='coerce')
    if df['일자'].isna().all():
        raise ValueError('XLS 파일의 날짜 컬럼을 datetime으로 변환하지 못했습니다.')

    df = preprocess_campaign_columns(df)
    df['총 클릭수'] = df['클릭수'] + df['컴패니언배너클릭수']
    df['CTR'] = safe_divide(df['총 클릭수'], df['노출수'])
    return df



def load_csv_conversion_data(path: str) -> pd.DataFrame:
    log('CSV 구매/매출 데이터 로딩 중...')
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    df = standardize_columns(df)
    require_columns(df, ['캠페인명', '방문수', '총구매수', '총매출액'], 'CSV')

    for column in ['방문수', '총구매수', '총매출액']:
        df[column] = clean_numeric_series(df[column])

    df = preprocess_campaign_columns(df)
    return df


# ============================================================
# 매칭 로직
# ============================================================
def build_lookup_map(df: pd.DataFrame, key_column: str) -> Dict[str, pd.Series]:
    deduped = df.drop_duplicates(subset=[key_column], keep='first')
    return {str(row[key_column]): row for _, row in deduped.iterrows() if str(row[key_column]).strip()}



def match_conversion_data(ad_agg: pd.DataFrame, conversion_df: pd.DataFrame) -> pd.DataFrame:
    log('캠페인 매칭 수행 중...')
    exact_map = build_lookup_map(conversion_df, '캠페인명')
    normalized_map = build_lookup_map(conversion_df, '캠페인명_정규화')
    product_map = build_lookup_map(conversion_df, '상품명_추정값_정규화')

    records: List[Dict[str, object]] = []
    for _, row in ad_agg.iterrows():
        matched = None
        match_type = 'unmatched'
        match_key = ''

        if row['캠페인명'] in exact_map:
            matched = exact_map[row['캠페인명']]
            match_type = 'exact_match'
            match_key = row['캠페인명']
        elif row['캠페인명_정규화'] in normalized_map:
            matched = normalized_map[row['캠페인명_정규화']]
            match_type = 'normalized_match'
            match_key = row['캠페인명_정규화']
        elif row['상품명_추정값_정규화'] in product_map:
            matched = product_map[row['상품명_추정값_정규화']]
            match_type = 'product_token_match'
            match_key = row['상품명_추정값_정규화']

        record = row.to_dict()
        record['match_type'] = match_type
        record['match_key'] = match_key
        if matched is not None:
            record['방문수'] = float(matched['방문수'])
            record['총구매수'] = float(matched['총구매수'])
            record['총매출액'] = float(matched['총매출액'])
            record['구매데이터_캠페인명'] = matched['캠페인명']
        else:
            record['방문수'] = 0.0
            record['총구매수'] = 0.0
            record['총매출액'] = 0.0
            record['구매데이터_캠페인명'] = ''
        records.append(record)

    matched_df = pd.DataFrame(records)
    matched_df['구매전환율'] = safe_divide(matched_df['총구매수'], matched_df['총 클릭수'])
    matched_df['클릭당 매출'] = safe_divide(matched_df['총매출액'], matched_df['총 클릭수'])
    matched_df['CTR'] = safe_divide(matched_df['총 클릭수'], matched_df['노출수'])
    return matched_df


# ============================================================
# 집계/리포트 생성
# ============================================================
def build_daily_tables(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log('일별/캠페인별 집계 생성 중...')
    daily_performance = (
        daily_df.groupby('일자', as_index=False)[['노출수', '총 클릭수', '동영상조회수']]
        .sum()
        .sort_values('일자')
    )
    daily_performance['CTR'] = safe_divide(daily_performance['총 클릭수'], daily_performance['노출수'])

    campaign_daily_trend = (
        daily_df.groupby(['캠페인명', '일자'], as_index=False)[['노출수', '총 클릭수', '동영상조회수']]
        .sum()
        .sort_values(['캠페인명', '일자'])
    )
    campaign_daily_trend['CTR'] = safe_divide(campaign_daily_trend['총 클릭수'], campaign_daily_trend['노출수'])

    campaign_totals = (
        daily_df.groupby(['캠페인명', '캠페인명_원본', '캠페인명_정규화', '상품명_추정값', '상품명_추정값_정규화'], as_index=False)
        [['노출수', '총 클릭수', '동영상조회수']]
        .sum()
    )
    campaign_totals['CTR'] = safe_divide(campaign_totals['총 클릭수'], campaign_totals['노출수'])
    return daily_performance, campaign_daily_trend, campaign_totals



def build_summary(campaign_perf: pd.DataFrame) -> pd.DataFrame:
    total_impressions = campaign_perf['노출수'].sum()
    total_clicks = campaign_perf['총 클릭수'].sum()
    total_video_views = campaign_perf['동영상조회수'].sum()
    total_purchases = campaign_perf['총구매수'].sum()
    total_revenue = campaign_perf['총매출액'].sum()
    matched_ratio = safe_divide((campaign_perf['match_type'] != 'unmatched').sum(), len(campaign_perf))

    summary = pd.DataFrame([
        {
            '전체 노출수': total_impressions,
            '전체 총 클릭수': total_clicks,
            '전체 CTR': safe_divide(total_clicks, total_impressions),
            '전체 동영상조회수': total_video_views,
            '전체 총구매수': total_purchases,
            '전체 총매출액': total_revenue,
            '전체 구매전환율': safe_divide(total_purchases, total_clicks),
            '매칭률': matched_ratio,
        }
    ])
    return summary



def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if '일자' in out.columns:
        out['일자'] = pd.to_datetime(out['일자']).dt.strftime('%Y-%m-%d')
    for column in ['CTR', '구매전환율', '전체 CTR', '전체 구매전환율', '매칭률']:
        if column in out.columns:
            out[f'{column}_표시'] = (out[column] * 100).round(2).map(lambda x: f'{x:.2f}%')
    for column in ['총매출액', '클릭당 매출', '전체 총매출액']:
        if column in out.columns:
            out[f'{column}_표시'] = out[column].round(0).map(lambda x: f'{x:,.0f}')
    for column in ['노출수', '총 클릭수', '동영상조회수', '방문수', '총구매수', '전체 노출수', '전체 총 클릭수', '전체 동영상조회수', '전체 총구매수']:
        if column in out.columns:
            out[f'{column}_표시'] = out[column].round(0).map(lambda x: f'{x:,.0f}')
    return out


# ============================================================
# 시각화
# ============================================================
def setup_korean_font() -> None:
    preferred_fonts = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']
    available = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams['font.family'] = font_name
            break
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'



def apply_chart_style(ax: plt.Axes, title: str, ylabel: str = '') -> None:
    ax.set_title(title, fontsize=14, fontweight='bold', color='#1F3C88', pad=14)
    ax.set_ylabel(ylabel, color='#335C99')
    ax.grid(axis='y', linestyle='--', alpha=0.25, color='#6FA8DC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D8E6F3')
    ax.spines['bottom'].set_color('#D8E6F3')
    ax.tick_params(colors='#3B4F6B')



def save_line_chart(df: pd.DataFrame, x: str, y_columns: List[str], labels: List[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['#1F77D0', '#6BA5FF', '#174A7E']
    for index, column in enumerate(y_columns):
        ax.plot(df[x], df[column], marker='o', linewidth=2.2, label=labels[index], color=colors[index % len(colors)])
    apply_chart_style(ax, title)
    ax.legend(frameon=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def save_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, output_path: Path, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(df[x], df[y], color='#2F80ED', edgecolor='#1C5FB8', alpha=0.92)
    apply_chart_style(ax, title)
    plt.xticks(rotation=35, ha='right')
    for bar, value in zip(bars, df[y]):
        label = f'{value * 100:.2f}%' if percent else f'{value:,.0f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=9, color='#1C355E')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def create_charts(daily_performance: pd.DataFrame, campaign_perf: pd.DataFrame, output_dir: Path, top_n: int) -> List[Path]:
    log('차트 생성 중...')
    setup_korean_font()
    chart_paths: List[Path] = []

    top_by_clicks = campaign_perf.sort_values('총 클릭수', ascending=False).head(top_n)
    top_by_revenue = campaign_perf.sort_values('총매출액', ascending=False).head(top_n)
    top_by_purchase_rate = campaign_perf.sort_values(['구매전환율', '총 클릭수'], ascending=[False, False]).head(top_n)

    path = output_dir / 'daily_impressions_clicks.png'
    save_line_chart(daily_performance, '일자', ['노출수', '총 클릭수'], ['노출수', '총 클릭수'], '일별 노출수 / 총 클릭수 추이', path)
    chart_paths.append(path)

    path = output_dir / 'daily_ctr.png'
    save_line_chart(daily_performance, '일자', ['CTR'], ['CTR'], '일별 CTR 추이', path)
    chart_paths.append(path)

    path = output_dir / 'campaign_clicks_top10.png'
    save_bar_chart(top_by_clicks, '캠페인명', '총 클릭수', f'캠페인별 총 클릭수 TOP{top_n}', path)
    chart_paths.append(path)

    path = output_dir / 'campaign_purchases_top10.png'
    save_bar_chart(top_by_revenue.sort_values('총구매수', ascending=False).head(top_n), '캠페인명', '총구매수', f'캠페인별 총구매수 TOP{top_n}', path)
    chart_paths.append(path)

    path = output_dir / 'campaign_conversion_rate_top10.png'
    save_bar_chart(top_by_purchase_rate, '캠페인명', '구매전환율', f'캠페인별 구매전환율 TOP{top_n}', path, percent=True)
    chart_paths.append(path)

    path = output_dir / 'campaign_revenue_top10.png'
    save_bar_chart(top_by_revenue, '캠페인명', '총매출액', f'캠페인별 총매출액 TOP{top_n}', path)
    chart_paths.append(path)
    return chart_paths


# ============================================================
# 저장/출력
# ============================================================
def export_to_excel(summary: pd.DataFrame, daily_performance: pd.DataFrame, campaign_performance: pd.DataFrame, output_path: Path) -> None:
    log(f'엑셀 저장 중... -> {output_path}')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        add_display_columns(summary).to_excel(writer, sheet_name='summary', index=False)
        add_display_columns(daily_performance).to_excel(writer, sheet_name='daily_performance', index=False)
        add_display_columns(campaign_performance).to_excel(writer, sheet_name='campaign_performance', index=False)



def print_report_samples(summary: pd.DataFrame, daily_performance: pd.DataFrame, campaign_performance: pd.DataFrame) -> None:
    log('summary 테이블 미리보기')
    print(add_display_columns(summary).to_string(index=False))
    log('일별 성과 테이블 미리보기')
    print(add_display_columns(daily_performance.head(10)).to_string(index=False))
    log('캠페인별 성과 테이블 상위 10개 미리보기')
    preview_columns = ['캠페인명', '상품명_추정값', 'match_type', '노출수', '총 클릭수', 'CTR', '방문수', '총구매수', '총매출액', '구매전환율']
    print(add_display_columns(campaign_performance[preview_columns].head(10)).to_string(index=False))
    log('최종 캠페인 매칭 결과 샘플 10개')
    sample_columns = ['캠페인명', '구매데이터_캠페인명', '상품명_추정값', 'match_type', 'match_key']
    print(campaign_performance[sample_columns].head(10).to_string(index=False))


# ============================================================
# 메인 실행 함수
# ============================================================
def run_analysis(config: Config) -> Dict[str, pd.DataFrame]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_df = load_xls_daily_data(config.xls_path)
    conversion_df = load_csv_conversion_data(config.csv_path)

    daily_performance, campaign_daily_trend, campaign_totals = build_daily_tables(daily_df)
    campaign_performance = match_conversion_data(campaign_totals, conversion_df)
    campaign_performance = campaign_performance.sort_values(['총매출액', '총 클릭수'], ascending=[False, False]).reset_index(drop=True)
    summary = build_summary(campaign_performance)

    create_charts(daily_performance, campaign_performance, output_dir, config.top_n)
    export_to_excel(summary, daily_performance, campaign_performance, output_dir / config.output_excel)
    print_report_samples(summary, daily_performance, campaign_performance)

    log(f'결과 파일 저장 경로: {output_dir.resolve()}')
    return {
        'summary': summary,
        'daily_performance': daily_performance,
        'campaign_performance': campaign_performance,
        'campaign_daily_trend': campaign_daily_trend,
    }



def main() -> None:
    config = Config()
    try:
        run_analysis(config)
        log('분석이 정상적으로 완료되었습니다.')
    except FileNotFoundError as error:
        print(f'[ERROR] 파일을 찾을 수 없습니다: {error}')
    except ValueError as error:
        print(f'[ERROR] 데이터 검증 실패: {error}')
    except Exception as error:
        print(f'[ERROR] 예상치 못한 오류가 발생했습니다: {error}')
        raise


if __name__ == '__main__':
    main()
