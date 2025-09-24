# Dockerfile (repo 루트)
FROM python:3.11-slim

# 1) 가벼운 기본 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) 의존성 설치: pyproject/uv.lock 우선, 없으면 requirements.txt
#   - 현재 repo에 pyproject.toml, uv.lock, requirements.txt 모두 있으므로
#     아래 로직이 자동으로 pyproject→uv.lock 기반 설치를 시도합니다.
COPY pyproject.toml uv.lock* requirements.txt* ./
RUN python -m pip install --upgrade pip && \
    if [ -f "pyproject.toml" ]; then \
        pip install uv && uv pip install --system . ; \
    elif [ -f "requirements.txt" ]; then \
        pip install -r requirements.txt ; \
    fi

# 3) 앱 소스 복사 (api 폴더만 있으면 충분)
COPY api ./api

# 4) 런타임 설정
ENV PORT=8000
EXPOSE 8000

# 5) FastAPI 실행 (A안: /api/* 경로)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
