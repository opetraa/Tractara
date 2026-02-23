# DESIGN

## 목표
규제가 강한 도메인(원자력, 의료 등)에서 문서 데이터를 신뢰할 수 있는 단일 진실 공급원(SSoT)으로 관리한다.
LLM이 소비하기 좋은 구조화된 JSON으로 변환하고, 검증 실패 시 LLM이 스스로 보정할 수 있는 에러 정보를 제공한다.

## 현재 프로토타입 범위
- PDF 1개 → DOC/TERM Baseline JSON → Landing/SSoT 저장까지의 엔드투엔드 Ingestion 파이프라인
- FastAPI 서버로 파이프라인 트리거 및 SSoT 조회 제공

## 기능 요구사항

### Ingestion 파이프라인
- PDF를 파싱해 섹션 구조와 텍스트를 추출한다
- 추출 결과를 DOC Baseline JSON Schema에 맞게 정규화한다
- TERM 후보를 추출하고 TERM Baseline JSON Schema에 맞게 정규화한다
- JSON Schema 검증을 통과한 것만 Landing Zone에 저장한다
- TERM 후보를 병합/중복 제거 후 승격 가능한 것만 SSoT에 저장한다

### API 서버
- PDF 업로드 및 Ingestion 트리거 엔드포인트
- SSoT DOC/TERM 조회 엔드포인트
- 검증 실패 시 `ProblemDetails` + `MachineReadableError` JSON 반환

### 에러 모델 요구사항
- LLM이 어떤 필드가 누락/잘못됐는지 파악할 수 있어야 한다
- 사람이 읽기 쉽고 기계가 파싱하기 쉬운 구조여야 한다

## 설계 결정

| 결정 | 이유 |
|---|---|
| JSON Schema 기반 검증 | 규제 도메인 특성상 스키마 엄격성 필요 |
| Landing Zone 분리 | 원본 보존 + SSoT 품질 보장 |
| TERM 승격 단계 분리 | 자동 추출과 신뢰 가능한 SSoT 사이의 품질 게이트 |
| ProblemDetails 에러 모델 | LLM 기반 자동 보정 루프 설계 고려 |
| DVC + Git 이중 관리 | 대용량 데이터(PDF, 모델)는 DVC, 코드/스키마는 Git |

## 미결 과제 (TODO)
- PDF 파서 실제 구현 (현재 더미)
- TERM 승격 기준 정형화
- LLM 기반 자동 보정 루프 설계
- 멀티 도큐먼트 충돌 해소 전략
