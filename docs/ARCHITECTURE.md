# ARCHITECTURE

## 모듈 구조

```
src/clara_ssot/
├── api/
│   ├── main.py                  # FastAPI 앱 진입점, 라우터 등록
│   └── pipeline.py              # Ingestion 파이프라인 오케스트레이터
├── parsing/
│   ├── pdf_parser.py            # PDF → 원시 텍스트/섹션 추출
│   ├── section_classifier.py   # 섹션 분류 (제목, 본문, 표 등)
│   └── metadata_extractor.py   # 표지/서문 메타데이터 추출 (Track A 규칙 + Track B LLM 병렬)
├── normalization/
│   ├── doc_mapper.py            # 원시 파싱 결과 → DOC Baseline JSON
│   └── term_mapper.py           # 원시 파싱 결과 → TERM 후보 리스트
├── landing/
│   └── landing_repository.py   # Landing Zone 저장 (로우 JSON)
├── validation/
│   ├── json_schema_validator.py # DOC/TERM JSON Schema 검증
│   └── term_validator.py        # TERM 승격 가능 여부 검증
├── curation/
│   └── term_curation_service.py # TERM 후보 병합 및 중복 제거
├── ssot/
│   ├── doc_ssot_repository.py   # DOC SSoT 저장/조회
│   ├── term_ssot_repository.py  # TERM SSoT 저장/조회
│   └── build_repository.py      # SSoT 빌드/재구성 유틸
├── schemas/
│   ├── DOC_baseline_schema.json # DOC JSON Schema 정의
│   └── TERM_baseline_schema.json# TERM JSON Schema 정의
├── logging_setup.py             # 로깅 설정
├── tracing.py                   # 분산 트레이싱
└── problem_details.py           # LLM-friendly 에러 모델
```

## 데이터 흐름

```
PDF 파일 입력
    │
    ▼
[parsing] pdf_parser.py + section_classifier.py
    │  원시 텍스트, 섹션 구조
    │
    ├──→ [metadata_extractor] (doc_mapper 내부에서 호출)
    │      Track A (결정론적): title 스코어링, identifier 정규식
    │      Track B (LLM/Gemini): creator, publisher, date 등 병렬 추출
    │      → ExtractedMetadata
    ▼
[normalization] doc_mapper.py → DOC Baseline JSON (메타데이터 포함)
               term_mapper.py  → TERM 후보 리스트
    │
    ▼
[validation] json_schema_validator.py
    │  스키마 통과한 것만 진행
    │  실패 시 → ProblemDetails + MachineReadableError 반환
    ▼
[landing] landing_repository.py
    │  Landing Zone에 원본 JSON 저장
    ▼
[curation] term_curation_service.py
    │  TERM 후보 병합, 중복 제거
    ▼
[validation] term_validator.py
    │  승격 가능한 TERM만 선별
    ▼
[ssot] doc_ssot_repository.py  → src/data/ssot/docs/
       term_ssot_repository.py → src/data/ssot/terms/
```

## 저장소 구조

```
src/data/
├── landing/docs/   # Landing Zone: 스키마 검증 통과한 원본 DOC JSON
└── ssot/
    ├── docs/       # SSoT DOC: 승격된 최종 문서
    └── terms/      # SSoT TERM: 승격된 최종 용어
```

## 에러 모델

검증 실패 시 `problem_details.py`의 `ProblemDetails` + `MachineReadableError` 형태로 반환.
LLM이 누락 필드와 수정 방법을 파악할 수 있도록 구조화된 JSON 에러를 제공.
