# 기본 커밋 메시지
M ?= chore: sync changes

.PHONY: help install test lint format clean run run-ui docker-build docker-up docker-down \
        dvc-repro dvc-push dvc-pull dvc-status git-push git-pull \
        dvc-add-push sync-all sync-rules ingest
FILE ?=

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean cache files"
	@echo "  make run         - Run API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up   - Start services with docker-compose"
	@echo ""
	@echo "  make git-push m=\"...\" - Add, commit, and push all changes"
	@echo "  make git-pull      - Pull changes from remote"
	@echo "  make dvc-repro   - Reproduce DVC pipeline"
	@echo "  make dvc-add-push file=\"...\" - Add a large file to DVC and push"
	@echo ""
	@echo "  make sync-all m=\"...\" - (RECOMMENDED) Add all changes to DVC/Git and push"
	@echo "  make sync-rules       - AI 규칙 동기화 (.ai-rules/ → CLAUDE.md, GEMINI.md)"

install:
	poetry install

test:
	poetry run pytest tests/ -v --cov=src/clara_ssot

lint:
	poetry run pylint src/
	poetry run mypy src/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +

run:
	poetry run uvicorn src.clara_ssot.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	@echo "API running on: http://127.0.0.1:8000/docs"
	@echo "   (Swagger UI 실행)"
	poetry run uvicorn src.clara_ssot.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t clara-ssot:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

dvc-repro:
	dvc repro

dvc-push:
	dvc push

dvc-pull:
	dvc pull

dvc-status:
	dvc status

# 사용법: make dvc-add-push file="file1.pptx file2.xlsx"
dvc-add-push:
	@dvc add $(file)
	@dvc push
	@git add $(patsubst %,%.dvc,$(file))
	@echo "✅ DVC에 파일 추가 및 푸시 완료. 이제 git-push를 실행하세요."

# sync-all: DVC와 Git의 모든 변경사항을 자동으로 감지하고 푸시합니다.
# 1. 새로 추가된 대용량 파일(pptx, xlsx 등)을 자동으로 `dvc add` 합니다.
# 2. DVC로 관리되는 모든 데이터를 원격 스토리지로 푸시합니다.
# 3. Git의 모든 변경사항을 원격 저장소로 푸시합니다.
sync-all:
	@echo "🔄 [1/3] 새로운 대용량 파일을 찾아 DVC에 추가합니다..."
	@find . -type f \( -name "*.pptx" -o -name "*.xlsx" -o -name "*.csv" -o -name "*.pdf" \) | while read file; do \
		if [ ! -f "$$file.dvc" ]; then \
			echo "  -> 새로운 파일 발견: $$file. DVC에 추가합니다."; \
			dvc add "$$file"; \
		fi \
	done
	@echo "🔄 [2/3] DVC 원격 스토리지로 모든 데이터를 푸시합니다..."
	@dvc push -a
	@echo "🔄 [3/3] Git 변경사항을 원격 저장소로 푸시합니다..."
	@make git-push m='$(M)'
	@echo "🚀 모든 동기화가 완료되었습니다!"

sync-rules:
	@echo "🔄 AI 규칙 파일 동기화 중..."
	@python3 scripts/sync_ai_rules.py
	@echo "✅ CLAUDE.md, GEMINI.md 업데이트 완료"

git-push:
	@git add .
	@git commit -m '$(M)'
	@git push
	@echo "✅ 모든 변경사항이 원격 저장소로 푸시되었습니다."

git-pull:
	@git pull
	@echo "✅ 원격 저장소의 최신 변경사항을 가져왔습니다."

# 1. 뒤에 오는 인자들을 변수로 캡처합니다.
# 'ingest'를 제외한 나머지 단어들을 가져옵니다.
ARGS = $(filter-out $@,$(MAKECMDGOALS))

# Makefile

.PHONY: ingest

ingest:
	@echo "📂 Starting bulk ingestion from /data folder..."
	@poetry run python3 src/clara_ssot/scripts/ingest_bulk.py

# 2. 뒤에 오는 인자들이 '존재하지 않는 명령어'라고 에러가 나지 않게 방지합니다.
%:
	@:

.PHONY: stop
stop:
	@echo "Stopping uvicorn server..."
	-pkill -f uvicorn
	@echo "Server stopped."
# 서버를 끈다.