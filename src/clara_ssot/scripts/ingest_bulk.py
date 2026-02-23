#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# 1. 프로젝트 루트 경로 설정 및 sys.path 추가 (Imports 전에 수행해야 함)
# 이 파일은 src/clara_ssot/scripts/ingest_bulk.py 에 위치함
current_file = Path(__file__).resolve()
# src/clara_ssot/scripts/ -> src/clara_ssot/ -> src/ -> root (CLARA-SSoT)
project_root = current_file.parents[3]
sys.path.append(str(project_root))

# 2. 프로젝트 모듈 임포트 (sys.path 설정 후)
try:
    from src.clara_ssot.validation.json_schema_validator import schema_registry
    from src.clara_ssot.logging_setup import configure_logging
    from src.clara_ssot.api.pipeline import ingest_single_document
except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    print(f"   Current sys.path: {sys.path}")
    sys.exit(1)

# .env 로드
load_dotenv(override=True)

logger = logging.getLogger("bulk_ingest")


def main():
    # 1. 로깅 및 스키마 초기화
    configure_logging()
    schema_registry.load()

    # 2. 데이터 디렉토리 설정
    # 사용자가 지정한 경로: /workspaces/CLARA-SSoT/data
    # 로컬 개발 환경 호환성을 위해 프로젝트 루트 기준 data 폴더도 확인
    target_dir = Path("/workspaces/CLARA-SSoT/data")
    if not target_dir.exists():
        target_dir = project_root / "data"

    if not target_dir.exists():
        logger.error(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {target_dir}")
        logger.error("프로젝트 루트에 'data' 폴더를 생성하고 PDF 파일을 넣어주세요.")
        sys.exit(1)

    # 3. PDF 파일 탐색
    pdf_files = list(target_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️  {target_dir} 디렉토리에 PDF 파일이 없습니다.")
        return

    logger.info(f"🚀 일괄 수집 시작: {target_dir} 내 {len(pdf_files)}개 PDF 파일")

    # 4. 파일별 수집 실행
    success_count = 0
    fail_count = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] 처리 중: {pdf_path.name} ...")
        try:
            # 파이프라인 실행
            result = ingest_single_document(pdf_path)

            doc_id = result.get("documentId", "Unknown ID")
            term_count = result.get("promotedTermCount", 0)

            logger.info(
                f"✅ 성공: {pdf_path.name} (DocID: {doc_id}, Terms: {term_count})")
            success_count += 1

        except Exception as e:
            logger.error(f"❌ 실패: {pdf_path.name}")
            logger.error(f"   이유: {str(e)}")
            fail_count += 1

    # 5. 최종 리포트
    logger.info("=" * 60)
    logger.info(f"📊 일괄 수집 완료 리포트")
    logger.info(f"   - 총 파일 수 : {len(pdf_files)}")
    logger.info(f"   - 성공       : {success_count}")
    logger.info(f"   - 실패       : {fail_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
