from pydantic import BaseModel


class DuplicateDetectionResult(BaseModel):
    """Image deduplication analysis result."""

    hash_value: str
