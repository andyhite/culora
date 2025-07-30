"""Tests for FaceAnalysisService."""

from datetime import datetime
from pathlib import Path

import pytest

from culora.domain import FaceAnalysisConfig, ImageLoadResult, ImageMetadata
from culora.services.face_analysis_service import (
    FaceAnalysisService,
    FaceAnalysisServiceError,
    get_face_analysis_service,
    initialize_face_analysis_service,
)
from tests.helpers import ConfigBuilder


class TestFaceAnalysisService:
    """Test FaceAnalysisService functionality."""

    def test_face_analysis_service_initialization(self) -> None:
        """Test FaceAnalysisService initialization."""
        config = ConfigBuilder().build()
        service = FaceAnalysisService(config)

        assert service.config == config
        assert service.face_config == config.faces
        assert isinstance(service.face_config, FaceAnalysisConfig)
        assert service._model is None
        assert service._device_context is None

    def test_get_model_info_not_initialized(self) -> None:
        """Test getting model info when not initialized."""
        config = ConfigBuilder().build()
        service = FaceAnalysisService(config)

        info = service.get_model_info()

        assert info["status"] == "not_initialized"
        assert info["model_name"] == "buffalo_l"
        assert "cache_dir" in info

    def test_analyze_image_failed_image_load(self) -> None:
        """Test analysis with failed image load."""
        config = ConfigBuilder().build()
        service = FaceAnalysisService(config)

        # Create failed image load result
        image_result = ImageLoadResult(
            success=False,
            image=None,
            metadata=ImageMetadata(
                path=Path("/nonexistent/image.jpg"),
                format="JPEG",
                width=0,
                height=0,
                file_size=0,
                created_at=datetime.fromtimestamp(0.0),
                modified_at=datetime.fromtimestamp(0.0),
                is_valid=False,
                error_message="File not found",
            ),
            error="File not found",
            error_code="FILE_NOT_FOUND",
        )

        result = service.analyze_image(image_result)

        assert result.success is False
        assert result.face_count == 0
        assert result.error_code == "IMAGE_LOAD_FAILED"
        assert result.error and "Image loading failed" in result.error


class TestFaceAnalysisServiceGlobalInstance:
    """Test global FaceAnalysisService instance management."""

    def test_initialize_face_analysis_service(self) -> None:
        """Test initializing global face analysis service."""
        config = ConfigBuilder().build()

        service = initialize_face_analysis_service(config)

        assert isinstance(service, FaceAnalysisService)
        assert service.config == config

    def test_get_face_analysis_service_after_initialization(self) -> None:
        """Test getting face analysis service after initialization."""
        config = ConfigBuilder().build()

        # Initialize first
        original_service = initialize_face_analysis_service(config)

        # Get the service
        retrieved_service = get_face_analysis_service()

        assert retrieved_service is original_service

    def test_get_face_analysis_service_not_initialized(self) -> None:
        """Test getting face analysis service before initialization."""
        # Reset global service
        import culora.services.face_analysis_service

        culora.services.face_analysis_service._face_analysis_service = None

        with pytest.raises(FaceAnalysisServiceError) as exc_info:
            get_face_analysis_service()

        assert "FaceAnalysisService not initialized" in str(exc_info.value)
