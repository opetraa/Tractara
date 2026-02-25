# tests/test_term_types.py
"""
TERM 3단 분리 아키텍처 단위 테스트.

검증 대상:
- TermType Enum 값
- TermClassReference / TermRelReference / TermRuleReference 패턴 검증
- _normalize_term_id 정규화 로직
- build_term_baseline_candidates 출력 포맷
- term_curation_service 그룹핑 키 변경
"""
import pytest
from pydantic import ValidationError

from clara_ssot.models.term_types import (
    TermClassReference,
    TermRelReference,
    TermRuleReference,
    TermType,
)
from clara_ssot.normalization.term_mapper import (
    TermCandidate,
    _normalize_term_id,
    build_term_baseline_candidates,
)
from clara_ssot.curation.term_curation_service import merge_term_candidates


# ── TermType Enum ────────────────────────────────────────────────────────────

def test_term_type_values():
    assert TermType.CLASS.value == "TERM-CLASS"
    assert TermType.REL.value == "TERM-REL"
    assert TermType.RULE.value == "TERM-RULE"


def test_term_type_is_str():
    """JSON 직렬화 시 문자열로 출력되어야 한다."""
    assert isinstance(TermType.CLASS, str)


# ── Typed Reference 패턴 검증 ────────────────────────────────────────────────

class TestTermClassReference:
    def test_valid(self):
        ref = TermClassReference(term_id="term:class:operating_temperature")
        assert ref.term_id == "term:class:operating_temperature"

    def test_invalid_rel_prefix(self):
        with pytest.raises(ValidationError):
            TermClassReference(term_id="term:rel:exceeds_threshold")

    def test_invalid_rule_prefix(self):
        with pytest.raises(ValidationError):
            TermClassReference(term_id="term:rule:some_rule")

    def test_invalid_old_format(self):
        with pytest.raises(ValidationError):
            TermClassReference(term_id="term:amp")


class TestTermRelReference:
    def test_valid(self):
        ref = TermRelReference(term_id="term:rel:exceeds_threshold")
        assert ref.term_id == "term:rel:exceeds_threshold"

    def test_invalid_class_prefix(self):
        with pytest.raises(ValidationError):
            TermRelReference(term_id="term:class:operating_temperature")


class TestTermRuleReference:
    def test_valid(self):
        ref = TermRuleReference(term_id="term:rule:amp_inspection_interval_rule")
        assert ref.term_id == "term:rule:amp_inspection_interval_rule"

    def test_invalid_class_prefix(self):
        with pytest.raises(ValidationError):
            TermRuleReference(term_id="term:class:operating_temperature")


# ── _normalize_term_id ───────────────────────────────────────────────────────

class TestNormalizeTermId:
    def test_headword_en_takes_priority(self):
        result = _normalize_term_id("Operating Temperature", "OT", TermType.CLASS)
        assert result == "term:class:operating_temperature"

    def test_spaces_to_underscore(self):
        result = _normalize_term_id("Stress Corrosion Cracking", "SCC", TermType.CLASS)
        assert result == "term:class:stress_corrosion_cracking"

    def test_rel_prefix(self):
        result = _normalize_term_id("Exceeds Threshold", "exceeds", TermType.REL)
        assert result == "term:rel:exceeds_threshold"

    def test_rule_prefix(self):
        result = _normalize_term_id("AMP Inspection Interval Rule", "", TermType.RULE)
        assert result == "term:rule:amp_inspection_interval_rule"

    def test_fallback_to_term_when_headword_en_empty(self):
        result = _normalize_term_id("", "AMP", TermType.CLASS)
        assert result == "term:class:amp"

    def test_fallback_to_term_when_headword_en_none(self):
        result = _normalize_term_id(None, "Fatigue", TermType.CLASS)
        assert result == "term:class:fatigue"

    def test_non_ascii_headword_falls_back_via_normalization(self):
        """한글 headword는 ASCII 필터 후 비어있으면 'unknown'으로."""
        result = _normalize_term_id("피로", "", TermType.CLASS)
        # 한글만 있으면 ASCII 필터 후 normalized가 비어 "unknown"
        assert result == "term:class:unknown"

    def test_mixed_ascii_korean(self):
        """영문+한글 혼합이면 영문 부분만 남는다."""
        result = _normalize_term_id("AMP 경년열화", "AMP", TermType.CLASS)
        assert result == "term:class:amp"

    def test_special_chars_removed(self):
        # 하이픈(-) → 공백 → 언더스코어, 괄호 제거 후 공백 → 언더스코어
        result = _normalize_term_id("Stress-Corrosion Cracking (SCC)", "SCC", TermType.CLASS)
        assert result == "term:class:stress_corrosion_cracking_scc"

    def test_no_uuid_in_output(self):
        """출력에 UUID 형식이 없어야 한다."""
        result = _normalize_term_id("Operating Temperature", "OT", TermType.CLASS)
        import re
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-"
        assert not re.search(uuid_pattern, result)


# ── build_term_baseline_candidates ──────────────────────────────────────────

class TestBuildTermBaselineCandidates:
    def _make_candidate(self, headword_en: str = "Aging Management Program", term: str = "AMP") -> TermCandidate:
        return TermCandidate(
            term=term,
            headword_en=headword_en,
            headword_ko="경년열화 관리 프로그램",
            definition_en="A program to manage aging effects.",
            definition_ko="경년열화 관리 프로그램.",
            domain=["nuclear"],
            context="AMP is required.",
            term_type=TermType.CLASS,
        )

    def test_term_id_format(self):
        result = build_term_baseline_candidates("DOC-001", [self._make_candidate()])
        assert len(result) == 1
        assert result[0]["termId"] == "term:class:aging_management_program"

    def test_term_type_field_present(self):
        result = build_term_baseline_candidates("DOC-001", [self._make_candidate()])
        assert result[0]["termType"] == "TERM-CLASS"

    def test_no_uuid_in_term_id(self):
        """termId에 UUID 형식이 없어야 한다."""
        import re
        result = build_term_baseline_candidates("DOC-001", [self._make_candidate()])
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-"
        assert not re.search(uuid_pattern, result[0]["termId"])

    def test_provenance_doc_id(self):
        result = build_term_baseline_candidates("DOC-XYZ", [self._make_candidate()])
        assert result[0]["provenance"]["sources"][0]["docId"] == "DOC-XYZ"

    def test_status_is_candidate(self):
        result = build_term_baseline_candidates("DOC-001", [self._make_candidate()])
        assert result[0]["status"] == "candidate"


# ── merge_term_candidates 그룹핑 키 ─────────────────────────────────────────

class TestMergeTermCandidates:
    def test_same_term_same_type_merges(self):
        """같은 term + 같은 termType → 병합."""
        candidates = [
            {"term": "AMP", "termType": "TERM-CLASS", "termId": "term:class:amp",
             "definition_en": "[PENDING]", "definition_ko": ""},
            {"term": "AMP", "termType": "TERM-CLASS", "termId": "term:class:amp",
             "definition_en": "Aging Management Program.", "definition_ko": "경년열화"},
        ]
        merged = merge_term_candidates(candidates)
        assert len(merged) == 1
        assert merged[0]["definition_en"] == "Aging Management Program."

    def test_same_term_different_type_not_merged(self):
        """같은 term이지만 termType이 다르면 별개 엔트리로 유지."""
        candidates = [
            {"term": "requires", "termType": "TERM-CLASS", "termId": "term:class:requires",
             "definition_en": "A concept.", "definition_ko": ""},
            {"term": "requires", "termType": "TERM-REL", "termId": "term:rel:requires",
             "definition_en": "A relation.", "definition_ko": ""},
        ]
        merged = merge_term_candidates(candidates)
        assert len(merged) == 2

    def test_different_terms_not_merged(self):
        """다른 term → 각각 별개 엔트리."""
        candidates = [
            {"term": "AMP", "termType": "TERM-CLASS", "termId": "term:class:amp",
             "definition_en": "...", "definition_ko": ""},
            {"term": "SCC", "termType": "TERM-CLASS", "termId": "term:class:scc",
             "definition_en": "...", "definition_ko": ""},
        ]
        merged = merge_term_candidates(candidates)
        assert len(merged) == 2
