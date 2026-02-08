# CLARA-SSoT

**CLARA-SSoT**는 규제가 강한 도메인(원자력, 의료 등)에서 쓸 수 있는
데이터 중심 AI 아키텍처입니다.

현재 버전은 **PDF 1개 → DOC/TERM Baseline JSON → Landing/SSoT 저장**까지의
엔드투엔드 **Ingestion 파이프라인 + FastAPI 서버**를 제공하는 프로토타입입니다.

---

## 주요 아이디어

- **SSoT (Single Source of Truth)**
  - 문서(DOC)와 용어(TERM)를 **JSON 스키마**로 정의하고,
  - 이 스키마를 통과한 것만 SSoT에 저장합니다.

- **Ingestion 파이프라인**
  1. PDF 파싱 (현재는 더미)
  2. DOC Baseline 생성 + 스키마 검증
  3. Landing Zone 저장
  4. TERM 후보 추출/정규화/병합
  5. 승격 가능한 TERM만 골라 SSoT 저장

- **LLM-friendly 에러 모델**
  - JSON Schema 검증 결과를 **ProblemDetails + MachineReadableError** 형태로 반환
  - LLM이 뭘 채워야 하는지 학습/보정하기 쉽게 설계

---

## 디렉터리 구조 (요약)

```text
CLARA-SSoT-main/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── start-swagger.bat     # SwaggerUI 창과 VS code 실행
├── data/
│   ├── landing/          # Ingestion 결과 원본(로우) JSON 저장
│   └── ssot/
│       ├── docs/         # DOC SSoT
│       └── terms/        # TERM SSoT
├── models/               # (향후) 모델/체크포인트 위치
├── src/
│   └── clara_ssot/
│       ├── api/          # FastAPI 엔드포인트, 파이프라인 엔트리
│       ├── parsing/      # PDF 파서 (현재 더미)
│       ├── normalization/# DOC/TERM Baseline 매핑
│       ├── landing/      # Landing Zone 저장소
│       ├── ssot/         # SSoT 저장소
│       ├── validation/   # JSON Schema / TERM validator
│       ├── curation/     # TERM 후보 병합/큐레이션
│       ├── schemas/      # DOC/TERM JSON Schema
│       ├── logging_setup.py
│       ├── tracing.py
│       └── problem_details.py
└── tests/
    └── test_smoke.py

