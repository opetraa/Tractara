# CLARA-SSoT

**CLARA-SSoT**는 규제가 강한 도메인(원자력, 의료 등)에서 쓸 수 있는
데이터 중심 AI 아키텍처입니다.

**PDF → DOC/TERM Baseline JSON → Landing/SSoT 저장**까지의
엔드투엔드 **Ingestion 파이프라인 + FastAPI 서버**를 제공합니다.

---

## 주요 아이디어

- **SSoT (Single Source of Truth)**
  - 문서(DOC)와 용어(TERM)를 **JSON 스키마**로 정의하고,
  - 이 스키마를 통과한 것만 SSoT에 저장합니다.

- **Ingestion 파이프라인**
  1. PDF 파싱 — Docling → PyMuPDF → Gemini Vision 하이브리드 전략
  2. 섹션 분류 + DCMES 메타데이터 추출 (Track A 결정론적 + Track B LLM 병렬)
  3. DOC Baseline 생성 + 스키마 검증
  4. Landing Zone 저장
  5. TERM 후보 추출/정규화/병합 (Gemini LLM + `instructor`)
  6. 승격 가능한 TERM만 골라 SSoT 저장

- **LLM-friendly 에러 모델**
  - JSON Schema 검증 결과를 **ProblemDetails + MachineReadableError** (RFC 7807) 형태로 반환
  - LLM이 누락 필드와 수정 방법을 바로 파악할 수 있도록 구조화

---

## AI 규칙 관리

Claude Code, Gemini CLI 등 여러 AI 도구에 동일한 프로젝트 규칙을 유지하기 위해
**마스터 파일 → 도구별 파일 자동 생성** 방식을 사용합니다.

```text
.ai-rules/
├── main.md              ← 공통 규칙 (여기만 편집)
├── claude/
│   └── overrides.md     ← Claude Code 전용 추가
└── gemini/
    └── overrides.md     ← Gemini CLI 전용 추가

CLAUDE.md                ← 자동 생성 (직접 편집 금지)
GEMINI.md                ← 자동 생성 (직접 편집 금지)
```

규칙 변경 시: `.ai-rules/main.md` 수정 → `make sync-rules` 실행

---

## 디렉터리 구조

```text
CLARA-SSoT/
├── .ai-rules/                  # AI 도구 공통 규칙 마스터
├── .github/workflows/          # CI/CD 파이프라인
├── .pre-commit-config.yaml     # pre-commit 훅 (format, lint, rules sync)
├── docs/
│   ├── ARCHITECTURE.md         # 모듈 구조 및 데이터 흐름
│   └── DESIGN.md               # 설계 목표 및 결정
├── scripts/
│   └── sync_ai_rules.py        # AI 규칙 동기화 스크립트
├── src/
│   └── clara_ssot/
│       ├── api/
│       │   ├── main.py         # FastAPI 앱 진입점
│       │   └── pipeline.py     # Ingestion 파이프라인 오케스트레이터
│       ├── parsing/
│       │   ├── pdf_parser.py           # 하이브리드 PDF 파서 (Docling/PyMuPDF/Gemini)
│       │   ├── section_classifier.py   # 섹션 분류 (제목, 본문, 표 등)
│       │   └── metadata_extractor.py   # DCMES 메타데이터 추출
│       ├── normalization/
│       │   ├── doc_mapper.py   # 파싱 결과 → DOC Baseline JSON
│       │   └── term_mapper.py  # 파싱 결과 → TERM 후보 (Gemini LLM)
│       ├── landing/
│       │   └── landing_repository.py
│       ├── validation/
│       │   ├── json_schema_validator.py
│       │   └── term_validator.py
│       ├── curation/
│       │   └── term_curation_service.py
│       ├── ssot/
│       │   ├── doc_ssot_repository.py
│       │   ├── term_ssot_repository.py
│       │   └── build_repository.py
│       ├── schemas/
│       │   ├── DOC_baseline_schema.json
│       │   └── TERM_baseline_schema.json
│       ├── scripts/
│       │   └── ingest_bulk.py  # 벌크 PDF 인제스트
│       ├── problem_details.py  # RFC 7807 에러 모델
│       ├── logging_setup.py
│       └── tracing.py          # W3C Trace Context
├── src/data/
│   ├── landing/
│   │   ├── docs/               # Landing Zone DOC JSON
│   │   └── terms/              # Landing Zone TERM 후보 JSON
│   └── ssot/
│       ├── docs/               # SSoT DOC
│       └── terms/              # SSoT TERM
├── tests/
│   ├── test_smoke.py
│   └── test_parsing_strategy.py
├── mcp_server.py               # FastMCP 서버 (PPTX, Excel, Git 도구)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── dvc.yaml                    # DVC 파이프라인
├── params.yaml                 # DVC 파라미터
├── CLAUDE.md                   # 자동 생성 (Claude Code 규칙)
└── GEMINI.md                   # 자동 생성 (Gemini CLI 규칙)
```

---

## 주요 의존성

| 용도 | 라이브러리 |
|---|---|
| API 서버 | FastAPI, Uvicorn |
| 데이터 검증 | Pydantic v2, jsonschema |
| PDF 파싱 (기본) | Docling |
| PDF 파싱 (폴백) | PyMuPDF, pdfplumber |
| LLM 추출 | google-generativeai (Gemini), anthropic |
| 구조화 출력 | instructor |
| 데이터 버저닝 | DVC (S3, GDrive 지원) |
| 코드 품질 | black, isort, pylint, mypy, pytest |

---

## 커맨드

```bash
make install      # 의존성 설치 (poetry install)
make format       # black + isort
make lint         # pylint + mypy
make test         # pytest + coverage
make run          # FastAPI 서버 실행 (port 8000)
make ingest       # 벌크 PDF 인제스트
make docker-up    # Docker Compose 실행
make sync-all     # DVC + Git 전체 동기화
make sync-rules   # AI 규칙 동기화 (.ai-rules/ → CLAUDE.md, GEMINI.md)
```

---

## 빠른 시작

```bash
# 1. 의존성 설치
make install

# 2. 환경 변수 설정
cp .env.example .env
# .env에 GEMINI_API_KEY 입력

# 3. 서버 실행
make run
# → http://localhost:8000/docs

# 4. PDF 인제스트
curl -X POST "http://localhost:8000/ingest" -F "file=@document.pdf"
```

### Docker 사용

```bash
make docker-up    # clara-ssot-api 컨테이너 실행 (port 8000)
```

---

## 참고 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 모듈 구조, 데이터 흐름, 저장소 구조
- [DESIGN.md](docs/DESIGN.md) — 설계 목표, 기능 요구사항, 설계 결정
