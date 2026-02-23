<!-- ⚠️  자동 생성 파일 — 직접 편집하지 마세요.
     원본: .ai-rules/main.md + .ai-rules/gemini/overrides.md
     수정 방법: 해당 파일을 편집한 뒤 `make sync-rules` 실행 -->

# 프로젝트 개요
규제 도메인(원자력, 의료 등)용 데이터 중심 AI 아키텍처. PDF → JSON SSoT 파이프라인 + FastAPI 서버.

# 워크플로우 원칙
- IMPORTANT: 코드 생성 전 반드시 자연어로 의사결정 논리를 먼저 작성하고 확인받을 것
- 한 번에 하나의 함수/모듈만 구현. 모놀리식 출력 금지
- 새 기능은 반드시 테스트와 함께 작성
- 아키텍처 변경 전 반드시 질문할 것
- 코드 변경 완료 후 아래 문서 동기화 표를 참조해 관련 docs/ 업데이트 여부 확인

| 변경 대상 | 업데이트 대상 |
|---|---|
| 모듈 추가/삭제, 데이터 흐름 변경 | `docs/ARCHITECTURE.md` |
| 기능 추가/제거, 설계 목표 변경 | `docs/DESIGN.md` |
| 커맨드/의존성 변경 | `.ai-rules/main.md # 커맨드` |
| AI 규칙 변경 | `.ai-rules/main.md` 편집 후 `make sync-rules` 실행 |

# 코드 스타일
- Python 3.11+
- 포매터: black (line-length 88), isort (profile=black)
- 린터: pylint, mypy
- 타입 힌트 필수 (mypy strict 기준)

# 커맨드
```bash
make install      # 의존성 설치 (poetry install)
make test         # 테스트 실행 (pytest + coverage)
make lint         # pylint + mypy
make format       # black + isort
make run          # FastAPI 서버 실행 (port 8000)
make ingest       # 벌크 PDF 인제스트
make docker-up    # Docker Compose 실행
make sync-all     # DVC + Git 전체 동기화 (권장)
make sync-rules   # AI 규칙 동기화 (.ai-rules/ → CLAUDE.md, GEMINI.md)
```

# 참고 문서
docs/ARCHITECTURE.md 와 docs/DESIGN.md를 항상 참고하세요.
