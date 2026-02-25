## 광고 캠페인 리포트 대시보드

업로드한 XLSX/CSV 데이터를 기반으로 광고 캠페인을 일자별/캠페인별로 자동 요약하고 시각화하는 Streamlit 앱입니다.

### 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 지원 기능

- 필수 컬럼 검증 및 숫자/날짜 자동 정제
- 캠페인별 집계: 노출, 클릭, 조회, CTR/VTR, 비용 지표(eCPM/CPC/CPV)
- 일자별 집계 테이블 + 추이 그래프
- 회사/캠페인/기간 필터
- 캠페인 요약 CSV 다운로드

### 입력 데이터 필수 컬럼

`광고계정`, `회사`, `캠페인명`, `일자`, `노출수`, `클릭수`, `컴패니언배너클릭수`, `동영상조회수`, `CTR`, `CTR(전체)`, `VTR`, `광고비`, `eCPM`, `CPC`, `CPV`
