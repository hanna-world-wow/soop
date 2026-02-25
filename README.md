# 광고 캠페인 통합 리포트 (Streamlit)

광고 리포트(`.xlsx`, `.xls`, `.csv`)를 업로드하면 캠페인/일자별 성과를 집계하고,
차트/랭킹/상세 테이블을 확인한 뒤 CSV/PDF로 내보낼 수 있는 Streamlit 앱입니다.

## 포함 기능
- 파일 업로드 및 컬럼 정규화
- 캠페인명 규칙 기반 광고주/상품/기기 파싱
- KPI 재계산 (`CTR`, `VTR`, `CPC`, `CPV`, `eCPM`)
- 성과 추이 차트, TOP 5 랭킹, 상세 데이터 표시
- CSV 및 광고주용 PDF 다운로드

## 로컬 실행
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 기본적으로 아래 주소로 접속합니다.
- http://localhost:8501

## 배포 시 참고
- PDF 차트 이미지 렌더링을 위해 `kaleido`가 필요합니다.
- 엑셀 입력을 위해 `openpyxl`/`xlrd`가 필요합니다.
