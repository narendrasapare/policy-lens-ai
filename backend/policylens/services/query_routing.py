import re
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    FACTUAL = "factual"
    SUMMARY = "summary"
    MULTI_PART = "multi_part"


@dataclass(frozen=True)
class QueryPlan:
    query_type: QueryType
    sub_questions: tuple[str, ...]


class QueryRouter:
    _FOLLOW_UP = re.compile(
        r"\s*,?\s+and\s+(?=(?:what|how|who|when|where|why|which|"
        r"does|do|is|are|can|should|will|may|explain|describe)\b)",
        flags=re.IGNORECASE,
    )

    _COMPARISON = re.compile(
        r"^(?:compare|contrast)\s+(?:the\s+)?(.+?)\s+"
        r"(?:with|versus|vs\.?|and)\s+(?:the\s+)?(.+?)[?.!]*$",
        flags=re.IGNORECASE,
    )

    _SUMMARY = re.compile(
        r"^(?:(?:please|can you|could you)\s+)?"
        r"(?:summari[sz]e|outline|give\s+me\s+(?:a\s+|an\s+)?"
        r"(?:summary|overview)|provide\s+(?:a\s+|an\s+)?"
        r"(?:summary|overview))\b",
        flags=re.IGNORECASE,
    )

    def route(self, query: str) -> QueryPlan:
        cleaned = " ".join(query.split())

        if not cleaned:
            raise ValueError("Question cannot be blank.")

        clauses = self._FOLLOW_UP.split(cleaned)

        if len(clauses) > 1:
            return QueryPlan(
                query_type=QueryType.MULTI_PART,
                sub_questions=tuple(
                    self._as_question(clause) for clause in clauses
                ),
            )

        comparison = self._COMPARISON.fullmatch(cleaned)

        if comparison is not None:
            left, right = comparison.groups()
            return QueryPlan(
                query_type=QueryType.MULTI_PART,
                sub_questions=(
                    f"What does the policy say about {left.strip()}?",
                    f"What does the policy say about {right.strip()}?",
                ),
            )

        if self._SUMMARY.match(cleaned):
            return QueryPlan(QueryType.SUMMARY, (cleaned,))

        return QueryPlan(QueryType.FACTUAL, (cleaned,))

    @staticmethod
    def _as_question(clause: str) -> str:
        return clause.strip(" ,.;?!") + "?"