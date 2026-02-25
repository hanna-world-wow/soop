"""초보자용 실행 스크립트.

- `python run_dashboard.py` 한 줄로 Streamlit 대시보드를 실행합니다.
- 필수 패키지가 없으면 설치 명령을 안내합니다.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

REQUIRED_MODULES = ["streamlit", "pandas", "plotly", "openpyxl"]


def _missing_modules() -> list[str]:
    missing: list[str] = []
    for module in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def main() -> int:
    missing = _missing_modules()
    if missing:
        print("[실행 전 준비 필요] 아래 패키지가 설치되지 않았습니다:")
        print("- " + ", ".join(missing))
        print("\n아래 명령을 먼저 실행해주세요:")
        print("pip install -r requirements.txt")
        return 1

    print("광고 캠페인 대시보드를 실행합니다...")
    print("브라우저가 자동으로 열리지 않으면 http://localhost:8501 접속")

    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
