# src/clara_ssot/api/main.py
from .pipeline import ingest_single_document
from ..validation.json_schema_validator import (
    SchemaValidationException,
    schema_registry,
)
from ..tracing import get_trace_id, new_child_span
from ..problem_details import MachineReadableError, ProblemDetails
from ..logging_setup import configure_logging
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi import FastAPI, File, UploadFile
import shutil
import tempfile
import traceback
from pathlib import Path
import logging

from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드


logger = logging.getLogger(__name__)
app = FastAPI(title="CLARA-SSoT Ingestion API")


@app.on_event("startup")
async def startup_event():
    configure_logging()
    schema_registry.load()


@app.exception_handler(SchemaValidationException)
async def schema_validation_exception_handler(_, exc: SchemaValidationException):
    # json_schema_validator에서 이미 ProblemDetails를 만들어 넣었으므로
    # 여기서는 그대로 반환만 해주면 된다.
    return JSONResponse(status_code=exc.problem.status, content=exc.problem.dict())


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    # 0) PDF 여부 체크
    if file.content_type != "application/pdf":
        trace_id = get_trace_id()
        problem = ProblemDetails(
            type="https://clara-ssot.org/problems/unsupported-media-type",
            title="Only PDF files are supported",
            status=415,
            detail=f"Received content-type: {file.content_type}",
            instance="/ingest",
            errors=[
                MachineReadableError(
                    code="UNSUPPORTED_MEDIA_TYPE",
                    target="file",
                    detail="Expected 'application/pdf' upload.",
                    meta={
                        "expected": "application/pdf",
                        "actual": file.content_type,
                        "errorClass": "user_input",
                    },
                )
            ],
            trace_id=trace_id,
            span_id=new_child_span(),
        )
        return JSONResponse(status_code=415, content=problem.dict())

    trace_id = get_trace_id()
    span_id = new_child_span()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            logger.info(f"Starting ingestion for file: {file.filename}")
            result = ingest_single_document(tmp_path)
            logger.info(f"Ingestion completed for file: {file.filename}")
        except SchemaValidationException as exc:
            # ✅ 스키마 에러는 전역 핸들러에게 맡긴다
            raise exc
        except Exception as e:
            # ✅ 나머지는 "internal-error"로 래핑하되, errors[]도 채워서 LLM-friendly 하게
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            problem = ProblemDetails(
                type="https://clara-ssot.org/problems/internal-error",
                title="Unexpected error during ingestion",
                status=500,
                detail=str(e),
                instance="/ingest",
                errors=[
                    MachineReadableError(
                        code="UNHANDLED_EXCEPTION",
                        target="clara_ssot.api.pipeline.ingest_single_document",
                        detail="Unhandled exception occurred during ingestion pipeline.",
                        meta={
                            "exceptionType": type(e).__name__,
                            "traceback": tb_str,
                            "errorClass": "system_bug",
                        },
                    )
                ],
                trace_id=trace_id,
                span_id=span_id,
            )
            return JSONResponse(status_code=500, content=problem.dict())

    # 정상 케이스 응답: 파이프라인 결과 + trace/span
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        **result,
    }
