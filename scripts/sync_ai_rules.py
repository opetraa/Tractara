"""AI 규칙 동기화 스크립트.

.ai-rules/main.md + .ai-rules/{tool}/overrides.md → CLAUDE.md, GEMINI.md

stdlib만 사용 — poetry 환경 불필요.
사용법:
    python3 scripts/sync_ai_rules.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

TOOLS: dict[str, str] = {
    "claude": "CLAUDE.md",
    "gemini": "GEMINI.md",
}

HEADER_TEMPLATE = """\
<!-- ⚠️  자동 생성 파일 — 직접 편집하지 마세요.
     원본: .ai-rules/main.md + .ai-rules/{tool}/overrides.md
     수정 방법: 해당 파일을 편집한 뒤 `make sync-rules` 실행 -->

"""


def sync() -> None:
    main_path = ROOT / ".ai-rules" / "main.md"
    if not main_path.exists():
        print(f"ERROR: {main_path} not found", file=sys.stderr)
        sys.exit(1)

    main_content = main_path.read_text(encoding="utf-8")
    changed: list[str] = []

    for tool, output_filename in TOOLS.items():
        overrides_path = ROOT / ".ai-rules" / tool / "overrides.md"
        if not overrides_path.exists():
            print(f"ERROR: {overrides_path} not found", file=sys.stderr)
            sys.exit(1)

        overrides_content = overrides_path.read_text(encoding="utf-8")
        header = HEADER_TEMPLATE.replace("{tool}", tool)
        new_content = header + main_content.rstrip("\n") + "\n\n" + overrides_content

        output_path = ROOT / output_filename
        if output_path.exists() and output_path.read_text(encoding="utf-8") == new_content:
            print(f"  [skip] {output_filename} — 변경 없음")
            continue

        output_path.write_text(new_content, encoding="utf-8")
        changed.append(output_filename)
        print(f"  [ok]   {output_filename} — 업데이트됨")

    if changed:
        print(f"\n생성된 파일: {', '.join(changed)}")
        print("git add 후 커밋하세요.")
        # pre-commit 환경에서 파일이 변경됐으면 훅 실패로 표시
        if "--check" in sys.argv:
            sys.exit(1)
    else:
        print("\n모든 파일이 최신 상태입니다.")


if __name__ == "__main__":
    sync()
