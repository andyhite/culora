"""Tests for face reference service."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from culora.domain import CuLoRAConfig
from culora.domain.models.face import FaceDetection
from culora.domain.models.face_reference import ReferenceSet
from culora.services.face_reference_service import (
    FaceReferenceService,
    FaceReferenceServiceError,
)


class TestFaceReferenceService:
    """Tests for FaceReferenceService."""

    @pytest.fixture
    def config(self) -> CuLoRAConfig:
        """Create test configuration."""
        return CuLoRAConfig()

    @pytest.fixture
    def service(self, config: CuLoRAConfig) -> FaceReferenceService:
        """Create face reference service instance."""
        return FaceReferenceService(config)

    def test_service_initialization(self, service: FaceReferenceService) -> None:
        """Test service initialization."""
        assert service.config is not None
        assert service.face_config is not None

    def test_select_primary_face_empty_list(
        self, service: FaceReferenceService
    ) -> None:
        """Test primary face selection with empty face list."""
        result = service.select_primary_face([])
        assert result is None

    def test_select_primary_face_single_face(
        self, service: FaceReferenceService
    ) -> None:
        """Test primary face selection with single face."""
        face = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=None,
            face_area_ratio=0.1,
        )

        result = service.select_primary_face([face])
        assert result == face

    def test_select_primary_face_no_reference_fallback(
        self, service: FaceReferenceService
    ) -> None:
        """Test primary face selection falls back to largest face."""
        face1 = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=None,
            face_area_ratio=0.1,
        )
        face2 = FaceDetection(
            bbox=(50.0, 60.0, 80.0, 90.0),
            confidence=0.90,
            landmarks=None,
            embedding=None,
            face_area_ratio=0.2,  # Larger face
        )

        result = service.select_primary_face([face1, face2])
        assert result == face2  # Should select larger face

    @patch("culora.services.face_reference_service.get_face_analysis_service")
    @patch("culora.services.face_reference_service.get_image_service")
    def test_process_reference_image_success(
        self,
        mock_image_service: Mock,
        mock_face_service: Mock,
        service: FaceReferenceService,
    ) -> None:
        """Test successful reference image processing."""
        # Mock image loading
        mock_image_result = Mock()
        mock_image_result.success = True
        mock_image_service.return_value.load_image.return_value = mock_image_result

        # Mock face analysis
        mock_face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            face_area_ratio=0.1,
        )

        mock_face_result = Mock()
        mock_face_result.success = True
        mock_face_result.faces = [mock_face_detection]
        mock_face_service.return_value.analyze_image.return_value = mock_face_result

        # Process image
        result = service.process_reference_image(Path("/test/image.jpg"))

        assert result.success is True
        assert len(result.embeddings) == 1
        assert result.error is None

    @patch("culora.services.face_reference_service.get_image_service")
    def test_process_reference_image_load_failure(
        self,
        mock_image_service: Mock,
        service: FaceReferenceService,
    ) -> None:
        """Test reference image processing with load failure."""
        # Mock image loading failure
        mock_image_result = Mock()
        mock_image_result.success = False
        mock_image_result.error = "Failed to load"
        mock_image_service.return_value.load_image.return_value = mock_image_result

        # Process image
        result = service.process_reference_image(Path("/test/image.jpg"))

        assert result.success is False
        assert result.error and "Failed to load image" in result.error

    def test_create_reference_set_from_nonexistent_directory(
        self, service: FaceReferenceService
    ) -> None:
        """Test creating reference set from non-existent directory."""
        result = service.create_reference_set_from_directory(Path("/nonexistent"))

        assert isinstance(result, ReferenceSet)
        assert len(result.images) == 0

    def test_save_and_load_reference_set(
        self, service: FaceReferenceService, tmp_path: Path
    ) -> None:
        """Test saving and loading reference set."""
        # Create a reference set
        ref_set = ReferenceSet()

        # Save it
        output_file = tmp_path / "test_refs.json"
        service.save_reference_set(ref_set, output_file)

        # Check file exists
        assert output_file.exists()

        # Load it back
        loaded_set = service.load_reference_set(output_file)

        assert isinstance(loaded_set, ReferenceSet)
        assert len(loaded_set.images) == 0

    def test_load_nonexistent_reference_set(
        self, service: FaceReferenceService
    ) -> None:
        """Test loading non-existent reference set."""
        with pytest.raises(FaceReferenceServiceError):
            service.load_reference_set(Path("/nonexistent.json"))

    @patch("culora.services.face_reference_service.get_face_analysis_service")
    @patch("culora.services.face_reference_service.get_image_service")
    def test_match_faces_to_reference_success(
        self,
        mock_image_service: Mock,
        mock_face_service: Mock,
        service: FaceReferenceService,
    ) -> None:
        """Test successful face matching to reference."""
        # Mock image loading
        mock_image_result = Mock()
        mock_image_result.success = True
        mock_image_service.return_value.load_image.return_value = mock_image_result

        # Mock face analysis
        mock_face_detection = FaceDetection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.95,
            landmarks=None,
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            face_area_ratio=0.1,
        )

        mock_face_result = Mock()
        mock_face_result.success = True
        mock_face_result.faces = [mock_face_detection]
        mock_face_service.return_value.analyze_image.return_value = mock_face_result

        # Create reference set with embeddings
        ref_set = ReferenceSet()

        # Match faces - this will fail due to no reference embeddings
        result = service.match_faces_to_reference(
            Path("/test/image.jpg"), ref_set, threshold=0.7
        )

        # Should fail due to no valid embeddings in reference set
        assert result.success is False
        assert result.error and "no valid embeddings" in result.error
        assert result.image_path == Path("/test/image.jpg")
        assert result.processing_duration >= 0
