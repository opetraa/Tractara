FROM python:3.11-slim-bullseye

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# 소스 코드 복사
COPY src ./src
COPY data ./data
COPY models ./models

# DVC 설정 복사
COPY .dvc ./.dvc
COPY dvc.yaml ./dvc.yaml
COPY params.yaml ./params.yaml

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 애플리케이션 실행
EXPOSE 8000
CMD ["uvicorn", "src.clara_ssot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
