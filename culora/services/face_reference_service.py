"""Reference service for identity matching and face selection."""

import json
import time
from pathlib import Path

from structlog import get_logger

from culora.core.exceptions import CuLoRAError
from culora.domain import CuLoRAConfig, FaceDetection
from culora.domain.models.face_reference import (
    ReferenceEmbedding,
    ReferenceImage,
    ReferenceMatchResult,
    ReferenceProcessingResult,
    ReferenceSet,
    SimilarityMatch,
)
from culora.services.face_analysis_service import get_face_analysis_service
from culora.services.image_service import get_image_service
from culora.utils.similarity import cosine_similarity

logger = get_logger(__name__)


class FaceReferenceServiceError(CuLoRAError):
    """Face reference service specific errors."""

    pass


class FaceReferenceService:
    """Service for reference image processing and identity matching."""

    def __init__(self, config: CuLoRAConfig) -> None:
        """Initialize reference service.

        Args:
            config: Application configuration
        """
        self.config = config
        self.face_config = config.faces

        logger.info(
            "FaceReferenceService initialized",
            similarity_threshold=self.face_config.reference_similarity_threshold,
        )

    def process_reference_image(self, image_path: Path) -> ReferenceProcessingResult:
        """Process a single reference image to extract face embeddings.

        Args:
            image_path: Path to the reference image

        Returns:
            Reference processing result with embeddings or error
        """
        try:
            # Load and analyze the image
            image_service = get_image_service()
            face_service = get_face_analysis_service()

            # Load the image
            image_result = image_service.load_image(image_path)
            if not image_result.success:
                return ReferenceProcessingResult(
                    success=False, error=f"Failed to load image: {image_result.error}"
                )

            # Analyze faces
            face_result = face_service.analyze_image(image_result)
            if not face_result.success:
                return ReferenceProcessingResult(
                    success=False, error=f"Failed to analyze faces: {face_result.error}"
                )

            if not face_result.faces:
                return ReferenceProcessingResult(
                    success=False, error="No faces detected in reference image"
                )

            # Create reference embeddings
            embeddings = []
            for face in face_result.faces:
                if face.embedding is not None:
                    ref_embedding = ReferenceEmbedding(
                        embedding=face.embedding,
                        face_detection=face,
                        source_image=image_path,
                        confidence_score=face.confidence,
                    )
                    embeddings.append(ref_embedding)

            if not embeddings:
                return ReferenceProcessingResult(
                    success=False,
                    error="No valid face embeddings found in reference image",
                )

            return ReferenceProcessingResult(success=True, embeddings=embeddings)

        except Exception as e:
            logger.error(
                "Reference processing failed", image_path=str(image_path), error=str(e)
            )
            return ReferenceProcessingResult(
                success=False, error=f"Processing failed: {e}"
            )

    def create_reference_set_from_images(self, image_paths: list[Path]) -> ReferenceSet:
        """Create a reference set from a list of image files.

        Args:
            image_paths: List of paths to reference images

        Returns:
            Reference set with processed images
        """
        reference_set = ReferenceSet()

        logger.info(
            "Processing reference images from file list",
            count=len(image_paths),
        )

        for image_path in image_paths:
            try:
                # Process each image
                processing_result = self.process_reference_image(image_path)

                # Create reference image
                ref_image = ReferenceImage(
                    image_path=image_path,
                    embeddings=(
                        processing_result.embeddings
                        if processing_result.success
                        else []
                    ),
                    processing_success=processing_result.success,
                    error_message=processing_result.error,
                )

                reference_set.add_reference_image(ref_image)

            except Exception as e:
                logger.error(
                    "Failed to process reference image",
                    image_file=str(image_path),
                    error=str(e),
                )
                # Add failed reference
                ref_image = ReferenceImage(
                    image_path=image_path,
                    processing_success=False,
                    error_message=str(e),
                )
                reference_set.add_reference_image(ref_image)

        logger.info(
            "Reference set created from image list",
            total_images=len(reference_set.images),
            valid_images=len(reference_set.valid_images),
            total_embeddings=reference_set.total_embeddings,
        )

        return reference_set

    def create_reference_set_from_directory(self, directory: Path) -> ReferenceSet:
        """Create a reference set from all images in a directory.

        Args:
            directory: Directory containing reference images

        Returns:
            Reference set with processed images
        """
        reference_set = ReferenceSet()

        if not directory.exists() or not directory.is_dir():
            logger.warning("Directory does not exist", directory=str(directory))
            return reference_set

        # Get all image files
        image_service = get_image_service()
        supported_formats = image_service.get_supported_formats()

        image_files: list[Path] = []
        for format_ext in supported_formats:
            image_files.extend(directory.glob(f"*{format_ext}"))
            image_files.extend(directory.glob(f"*{format_ext.upper()}"))

        logger.info(
            "Processing reference images",
            directory=str(directory),
            count=len(image_files),
        )

        for image_file in image_files:
            try:
                # Process each image
                processing_result = self.process_reference_image(image_file)

                # Create reference image
                ref_image = ReferenceImage(
                    image_path=image_file,
                    embeddings=(
                        processing_result.embeddings
                        if processing_result.success
                        else []
                    ),
                    processing_success=processing_result.success,
                    error_message=processing_result.error,
                )

                reference_set.add_reference_image(ref_image)

            except Exception as e:
                logger.error(
                    "Failed to process reference image",
                    image_file=str(image_file),
                    error=str(e),
                )
                # Add failed reference
                ref_image = ReferenceImage(
                    image_path=image_file,
                    processing_success=False,
                    error_message=str(e),
                )
                reference_set.add_reference_image(ref_image)

        logger.info(
            "Reference set created",
            total_images=len(reference_set.images),
            valid_images=len(reference_set.valid_images),
            total_embeddings=reference_set.total_embeddings,
        )

        return reference_set

    def save_reference_set(
        self, reference_set: ReferenceSet, output_path: Path
    ) -> None:
        """Save a reference set to a JSON file.

        Args:
            reference_set: Reference set to save
            output_path: Path to save the JSON file
        """
        try:
            # Convert to JSON-serializable format
            data = reference_set.model_dump(mode="json")

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON file
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("Reference set saved", output_path=str(output_path))

        except Exception as e:
            logger.error(
                "Failed to save reference set",
                output_path=str(output_path),
                error=str(e),
            )
            raise FaceReferenceServiceError(f"Failed to save reference set: {e}") from e

    def load_reference_set(self, input_path: Path) -> ReferenceSet:
        """Load a reference set from a JSON file.

        Args:
            input_path: Path to the JSON file

        Returns:
            Loaded reference set

        Raises:
            FaceReferenceServiceError: If loading fails
        """
        try:
            if not input_path.exists():
                raise FaceReferenceServiceError(
                    f"Reference set file does not exist: {input_path}"
                )

            with open(input_path) as f:
                data = json.load(f)

            # Parse into reference set
            reference_set = ReferenceSet.model_validate(data)

            logger.info(
                "Reference set loaded",
                input_path=str(input_path),
                total_images=len(reference_set.images),
                valid_images=len(reference_set.valid_images),
            )

            return reference_set

        except Exception as e:
            logger.error(
                "Failed to load reference set", input_path=str(input_path), error=str(e)
            )
            raise FaceReferenceServiceError(f"Failed to load reference set: {e}") from e

    def match_faces_to_reference(
        self, image_path: Path, reference_set: ReferenceSet, threshold: float
    ) -> ReferenceMatchResult:
        """Match faces in an image against a reference set.

        Args:
            image_path: Path to image to analyze
            reference_set: Reference set to match against
            threshold: Similarity threshold for matches

        Returns:
            Reference match result with all face matches
        """
        start_time = time.time()

        try:
            # Load and analyze the image
            image_service = get_image_service()
            face_service = get_face_analysis_service()

            image_result = image_service.load_image(image_path)
            if not image_result.success:
                return ReferenceMatchResult(
                    image_path=image_path,
                    success=False,
                    error=f"Failed to load image: {image_result.error}",
                    processing_duration=time.time() - start_time,
                )

            face_result = face_service.analyze_image(image_result)
            if not face_result.success:
                return ReferenceMatchResult(
                    image_path=image_path,
                    success=False,
                    error=f"Failed to analyze faces: {face_result.error}",
                    processing_duration=time.time() - start_time,
                )

            if not face_result.faces:
                return ReferenceMatchResult(
                    image_path=image_path,
                    success=True,
                    processing_duration=time.time() - start_time,
                )

            # Get reference embeddings
            ref_embeddings = reference_set.get_all_embeddings()
            if not ref_embeddings:
                return ReferenceMatchResult(
                    image_path=image_path,
                    success=False,
                    error="Reference set contains no valid embeddings",
                    processing_duration=time.time() - start_time,
                )

            # Match each face
            matches = []
            best_match_index = None
            best_similarity = 0.0

            for i, face in enumerate(face_result.faces):
                if face.embedding is None:
                    # Create a no-match result for faces without embeddings
                    match = SimilarityMatch(
                        face_detection=face,
                        similarities=[],
                        best_similarity=0.0,
                        best_reference_index=None,
                        meets_threshold=False,
                    )
                else:
                    # Calculate similarities to all reference embeddings
                    similarities = []
                    for ref_emb in ref_embeddings:
                        sim = cosine_similarity(face.embedding, ref_emb.embedding)
                        similarities.append(sim)

                    # Find best match
                    if similarities:
                        best_sim = max(similarities)
                        best_ref_idx = similarities.index(best_sim)
                        meets_threshold = best_sim >= threshold

                        match = SimilarityMatch(
                            face_detection=face,
                            similarities=similarities,
                            best_similarity=best_sim,
                            best_reference_index=best_ref_idx,
                            meets_threshold=meets_threshold,
                        )

                        # Track overall best match
                        if meets_threshold and best_sim > best_similarity:
                            best_similarity = best_sim
                            best_match_index = i
                    else:
                        match = SimilarityMatch(
                            face_detection=face,
                            similarities=[],
                            best_similarity=0.0,
                            best_reference_index=None,
                            meets_threshold=False,
                        )

                matches.append(match)

            return ReferenceMatchResult(
                image_path=image_path,
                success=True,
                matches=matches,
                primary_face_index=best_match_index,
                processing_duration=time.time() - start_time,
            )

        except Exception as e:
            logger.error(
                "Face matching failed", image_path=str(image_path), error=str(e)
            )
            return ReferenceMatchResult(
                image_path=image_path,
                success=False,
                error=str(e),
                processing_duration=time.time() - start_time,
            )

    def select_primary_face(
        self, faces: list[FaceDetection], reference_set: ReferenceSet | None = None
    ) -> FaceDetection | None:
        """Select the primary face from a list using reference matching.

        Args:
            faces: List of detected faces
            reference_set: Optional reference set for matching

        Returns:
            Primary face or None if no suitable face found
        """
        if not faces:
            return None

        if len(faces) == 1:
            return faces[0]

        # If no reference set, fall back to largest face
        if reference_set is None or not reference_set.get_all_embeddings():
            logger.debug("No reference set available, using largest face fallback")
            return max(faces, key=lambda f: f.face_area_ratio)

        try:
            # Try reference matching
            ref_embeddings = reference_set.get_all_embeddings()
            best_face = None
            best_similarity = 0.0

            for face in faces:
                if face.embedding is None:
                    continue

                # Calculate average similarity to all reference embeddings
                similarities = [
                    cosine_similarity(face.embedding, ref_emb.embedding)
                    for ref_emb in ref_embeddings
                ]

                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)

                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_face = face

            # If reference matching found a good match, use it
            if (
                best_face is not None
                and best_similarity >= self.face_config.reference_similarity_threshold
            ):
                logger.debug(
                    "Selected face using reference matching", similarity=best_similarity
                )
                return best_face

            # Fall back to largest face if enabled
            if self.face_config.use_reference_fallback:
                logger.debug(
                    "Reference matching below threshold, falling back to largest face"
                )
                return max(faces, key=lambda f: f.face_area_ratio)

            # No fallback, return None
            logger.debug("Reference matching below threshold, no fallback enabled")
            return None

        except Exception as e:
            logger.warning(
                "Reference matching failed, using largest face fallback", error=str(e)
            )
            return max(faces, key=lambda f: f.face_area_ratio)


# Global service instance
_face_reference_service: FaceReferenceService | None = None


def initialize_face_reference_service(config: CuLoRAConfig) -> FaceReferenceService:
    """Initialize global FaceReferenceService instance.

    Args:
        config: Application configuration

    Returns:
        Initialized FaceReferenceService instance
    """
    global _face_reference_service
    _face_reference_service = FaceReferenceService(config)
    logger.info("Global FaceReferenceService initialized")
    return _face_reference_service


def get_face_reference_service() -> FaceReferenceService:
    """Get global FaceReferenceService instance.

    Returns:
        Global FaceReferenceService instance
    """
    global _face_reference_service
    if _face_reference_service is None:
        from culora.services import get_config_service

        config_service = get_config_service()

        # Load default config if not already loaded
        try:
            config = config_service.get_config()
        except Exception:
            config = config_service.load_config()

        _face_reference_service = FaceReferenceService(config)
        logger.info("Global FaceReferenceService initialized")

    return _face_reference_service
