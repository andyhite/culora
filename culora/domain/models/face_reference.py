"""Reference image models for identity matching."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from culora.domain.models.face import FaceDetection


@dataclass
class ReferenceEmbedding:
    """Face embedding from a reference image.

    Contains the face embedding vector along with metadata about the
    source face detection used to create the embedding.
    """

    embedding: np.ndarray
    face_detection: FaceDetection
    source_image: Path
    confidence_score: float

    def __post_init__(self) -> None:
        """Validate embedding after initialization."""
        if len(self.embedding.shape) != 1:
            raise ValueError("Embedding must be a 1D numpy array")
        if self.confidence_score < 0.0 or self.confidence_score > 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


class ReferenceProcessingResult(BaseModel):
    """Result of processing a reference image."""

    success: bool = Field(description="Whether processing succeeded")
    embeddings: list[ReferenceEmbedding] = Field(
        default_factory=list, description="Extracted face embeddings"
    )
    error: str | None = Field(
        default=None, description="Error message if processing failed"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ReferenceImage(BaseModel):
    """Reference image with extracted face embeddings.

    Represents a single reference image that has been processed to extract
    face embeddings for identity matching.
    """

    image_path: Path = Field(description="Path to the reference image")
    embeddings: list[ReferenceEmbedding] = Field(
        default_factory=list, description="Face embeddings extracted from this image"
    )
    processed_at: datetime = Field(
        default_factory=datetime.now, description="When the image was processed"
    )
    processing_success: bool = Field(
        default=True, description="Whether processing succeeded"
    )
    error_message: str | None = Field(
        default=None, description="Error message if processing failed"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @field_serializer("image_path")
    def serialize_path(self, path: Path) -> str:
        """Serialize Path to string."""
        return str(path)

    @property
    def has_embeddings(self) -> bool:
        """Check if this reference image has any embeddings."""
        return len(self.embeddings) > 0

    @property
    def primary_embedding(self) -> ReferenceEmbedding | None:
        """Get the primary (highest confidence) embedding."""
        if not self.embeddings:
            return None
        return max(self.embeddings, key=lambda e: e.confidence_score)


class ReferenceSet(BaseModel):
    """Collection of reference images for identity matching.

    Contains multiple reference images and provides methods for
    identity matching against face embeddings.
    """

    images: list[ReferenceImage] = Field(
        default_factory=list, description="Reference images in this set"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When this set was created"
    )
    name: str | None = Field(
        default=None, description="Optional name for this reference set"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def total_embeddings(self) -> int:
        """Get total number of embeddings across all images."""
        return sum(len(img.embeddings) for img in self.images)

    @property
    def valid_images(self) -> list[ReferenceImage]:
        """Get only images that were processed successfully."""
        return [
            img for img in self.images if img.processing_success and img.has_embeddings
        ]

    def get_all_embeddings(self) -> list[ReferenceEmbedding]:
        """Get all embeddings from all reference images."""
        embeddings = []
        for image in self.valid_images:
            embeddings.extend(image.embeddings)
        return embeddings

    def add_reference_image(self, reference_image: ReferenceImage) -> None:
        """Add a reference image to this set."""
        self.images.append(reference_image)


class SimilarityMatch(BaseModel):
    """Result of matching a face against reference embeddings.

    Contains similarity scores and information about the best matching
    reference embedding.
    """

    face_detection: FaceDetection = Field(description="The face that was matched")
    similarities: list[float] = Field(
        description="Similarity scores to each reference embedding"
    )
    best_similarity: float = Field(description="Highest similarity score")
    best_reference_index: int | None = Field(
        default=None, description="Index of best matching reference embedding"
    )
    meets_threshold: bool = Field(
        description="Whether best similarity meets the threshold"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class ReferenceMatchResult(BaseModel):
    """Result of matching faces against a reference set.

    Contains match results for all faces detected in an image
    and indicates which face (if any) should be considered primary.
    """

    image_path: Path = Field(description="Path to the image that was matched")
    success: bool = Field(description="Whether matching succeeded")
    matches: list[SimilarityMatch] = Field(
        default_factory=list, description="Match results for each detected face"
    )
    primary_face_index: int | None = Field(
        default=None, description="Index of the primary (best matching) face"
    )
    processing_duration: float = Field(
        default=0.0, description="Time taken for matching in seconds"
    )
    error: str | None = Field(
        default=None, description="Error message if matching failed"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @field_serializer("image_path")
    def serialize_path(self, path: Path) -> str:
        """Serialize Path to string."""
        return str(path)

    @property
    def has_matches(self) -> bool:
        """Check if any faces met the similarity threshold."""
        return any(match.meets_threshold for match in self.matches)

    @property
    def primary_match(self) -> SimilarityMatch | None:
        """Get the primary match if available."""
        if self.primary_face_index is not None and self.primary_face_index < len(
            self.matches
        ):
            return self.matches[self.primary_face_index]
        return None
