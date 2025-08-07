"""Tests for CompositionService."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from culora.domain import CuLoRAConfig
from culora.domain.models.composition import (
    BackgroundComplexity,
    CameraAngle,
    CompositionAnalysis,
    FacialExpression,
    LightingQuality,
    SceneType,
    ShotType,
)
from culora.services.composition_service import (
    CompositionService,
    CompositionServiceError,
    get_composition_service,
)
from tests.mocks.vision_language_mocks import (
    MOCK_RESPONSES,
    MockTokenizer,
    MockVisionLanguageModel,
)


class TestCompositionService:
    """Test cases for CompositionService."""

    @pytest.fixture
    def config(self) -> CuLoRAConfig:
        """Create test configuration."""
        return CuLoRAConfig()

    @pytest.fixture
    def service(self, config: CuLoRAConfig) -> CompositionService:
        """Create composition service."""
        return CompositionService(config)

    @pytest.fixture
    def test_image(self, temp_dir: Path) -> tuple[Image.Image, Path]:
        """Create test image."""
        image = Image.new("RGB", (800, 600), color="red")
        image_path = temp_dir / "test_image.jpg"
        image.save(image_path)
        return image, image_path

    def test_composition_service_initialization(self, config: CuLoRAConfig) -> None:
        """Test composition service initialization."""
        service = CompositionService(config)
        assert service.config == config
        assert service.composition_config == config.composition
        assert service._model is None
        assert service._tokenizer is None

    @patch("culora.services.composition_service.AutoModelForCausalLM")
    @patch("culora.services.composition_service.AutoTokenizer")
    def test_model_loading_success(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        service: CompositionService,
    ) -> None:
        """Test successful model loading."""
        # Setup mocks
        mock_tokenizer = MockTokenizer()
        mock_model = MockVisionLanguageModel()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Load model
        service._ensure_model_loaded()

        # Verify model and tokenizer are loaded
        assert service._model is not None
        assert service._tokenizer is not None
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("culora.services.composition_service.AutoModelForCausalLM")
    @patch("culora.services.composition_service.AutoTokenizer")
    def test_model_loading_failure(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        service: CompositionService,
    ) -> None:
        """Test model loading failure."""
        # Setup mock to raise exception
        mock_model_class.from_pretrained.side_effect = Exception("Model loading failed")

        # Verify exception is raised
        with pytest.raises(CompositionServiceError):
            service._ensure_model_loaded()

    def test_image_preparation_no_resize_needed(
        self, service: CompositionService, test_image: tuple[Image.Image, Path]
    ) -> None:
        """Test image preparation when no resize is needed."""
        image, _ = test_image
        prepared = service._prepare_image_for_analysis(image)

        # Image should be unchanged
        assert prepared.size == image.size

    def test_image_preparation_with_resize(self, service: CompositionService) -> None:
        """Test image preparation with resize."""
        # Create large image that needs resizing
        large_image = Image.new("RGB", (2048, 1536), color="blue")
        prepared = service._prepare_image_for_analysis(large_image)

        # Image should be resized while maintaining aspect ratio
        max_width, max_height = service.composition_config.max_image_size
        assert prepared.size[0] <= max_width
        assert prepared.size[1] <= max_height

        # Aspect ratio should be preserved
        original_ratio = large_image.size[0] / large_image.size[1]
        new_ratio = prepared.size[0] / prepared.size[1]
        assert abs(original_ratio - new_ratio) < 0.01

    @patch("culora.services.composition_service.AutoModelForCausalLM")
    @patch("culora.services.composition_service.AutoTokenizer")
    def test_analyze_image_success(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        service: CompositionService,
        test_image: tuple[Image.Image, Path],
    ) -> None:
        """Test successful image analysis."""
        image, image_path = test_image

        # Setup mocks
        mock_tokenizer = MockTokenizer()
        mock_model = MockVisionLanguageModel(
            {"professional_headshot": MOCK_RESPONSES["professional_headshot"]}
        )
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Analyze image
        result = service.analyze_image(image, image_path)

        # Verify result
        assert result.success is True
        assert result.path == image_path
        assert result.analysis is not None
        assert result.analysis_duration is not None
        assert result.analysis_duration > 0
        assert result.model_response is not None

        # Verify analysis content
        analysis = result.analysis
        assert analysis.shot_type == ShotType.HEADSHOT
        assert analysis.scene_type == SceneType.STUDIO
        assert analysis.lighting_quality == LightingQuality.EXCELLENT
        assert analysis.background_complexity == BackgroundComplexity.SIMPLE
        assert analysis.facial_expression == FacialExpression.CONFIDENT
        assert analysis.camera_angle == CameraAngle.EYE_LEVEL
        assert analysis.confidence_score == 0.95

    @patch("culora.services.composition_service.AutoModelForCausalLM")
    @patch("culora.services.composition_service.AutoTokenizer")
    def test_analyze_image_with_parsing_error(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        service: CompositionService,
        test_image: tuple[Image.Image, Path],
    ) -> None:
        """Test image analysis with response parsing error."""
        image, image_path = test_image

        # Setup mocks with invalid response
        mock_tokenizer = MockTokenizer()
        mock_model = MockVisionLanguageModel(
            {"parsing_error": MOCK_RESPONSES["parsing_error"]}
        )
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Analyze image
        result = service.analyze_image(image, image_path)

        # Should use fallback parsing
        assert result.success is True
        assert result.analysis is not None
        # Fallback should set confidence to 0.5
        assert result.analysis.confidence_score == 0.5

    @patch("culora.services.composition_service.AutoModelForCausalLM")
    @patch("culora.services.composition_service.AutoTokenizer")
    def test_analyze_image_model_failure(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        service: CompositionService,
        test_image: tuple[Image.Image, Path],
    ) -> None:
        """Test image analysis with model failure."""
        image, image_path = test_image

        # Setup mocks to fail
        mock_model_class.from_pretrained.side_effect = Exception("Model loading failed")

        # Model loading should fail, causing analysis to fail
        result = service.analyze_image(image, image_path)

        # Verify failure
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "COMPOSITION_ANALYSIS_FAILED"
        assert result.analysis is None

    def test_parse_enum_field_valid_value(self, service: CompositionService) -> None:
        """Test enum field parsing with valid value."""
        result = service._parse_enum_field("closeup", ShotType, ShotType.UNKNOWN)
        assert result == ShotType.CLOSEUP

    def test_parse_enum_field_invalid_value(self, service: CompositionService) -> None:
        """Test enum field parsing with invalid value."""
        result = service._parse_enum_field("invalid", ShotType, ShotType.UNKNOWN)
        assert result == ShotType.UNKNOWN

    def test_parse_enum_field_none_value(self, service: CompositionService) -> None:
        """Test enum field parsing with None value."""
        result = service._parse_enum_field(None, ShotType, ShotType.UNKNOWN)
        assert result == ShotType.UNKNOWN

    def test_parse_enum_field_case_insensitive(
        self, service: CompositionService
    ) -> None:
        """Test enum field parsing is case insensitive."""
        result = service._parse_enum_field("CLOSEUP", ShotType, ShotType.UNKNOWN)
        assert result == ShotType.CLOSEUP

        result = service._parse_enum_field("CloseUp", ShotType, ShotType.UNKNOWN)
        assert result == ShotType.CLOSEUP

    def test_fallback_parse_response_shot_types(
        self, service: CompositionService
    ) -> None:
        """Test fallback response parsing for shot types."""
        response = "This image shows a closeup view of the subject"
        data = service._fallback_parse_response(response)
        assert data["shot_type"] == "closeup"

    def test_fallback_parse_response_scene_types(
        self, service: CompositionService
    ) -> None:
        """Test fallback response parsing for scene types."""
        response = "The subject is photographed outdoors in natural light"
        data = service._fallback_parse_response(response)
        assert data["scene_type"] == "outdoor"

    def test_fallback_parse_response_defaults(
        self, service: CompositionService
    ) -> None:
        """Test fallback response parsing sets defaults."""
        response = "Generic image description without specific keywords"
        data = service._fallback_parse_response(response)
        assert data["confidence"] == 0.5
        assert "description" in data

    @patch("culora.services.composition_service.AutoModelForCausalLM")
    @patch("culora.services.composition_service.AutoTokenizer")
    def test_analyze_batch_success(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        service: CompositionService,
        temp_dir: Path,
    ) -> None:
        """Test successful batch analysis."""
        # Create test images
        images_and_paths = []
        for i in range(3):
            image = Image.new("RGB", (400, 300), color=["red", "green", "blue"][i])
            path = temp_dir / f"test_{i}.jpg"
            image.save(path)
            images_and_paths.append((image, path))

        # Setup mocks
        mock_tokenizer = MockTokenizer()
        mock_model = MockVisionLanguageModel()  # Uses default responses
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Analyze batch
        result = service.analyze_batch(images_and_paths)

        # Verify batch result
        assert len(result.results) == 3
        assert result.successful_analyses == 3
        assert result.failed_analyses == 0
        assert result.total_duration > 0
        assert result.images_per_second > 0

        # Verify distributions
        assert len(result.shot_type_distribution) > 0
        assert len(result.scene_type_distribution) > 0
        assert result.mean_confidence > 0

    def test_analyze_batch_empty_list(self, service: CompositionService) -> None:
        """Test batch analysis with empty list."""
        result = service.analyze_batch([])

        assert len(result.results) == 0
        assert result.successful_analyses == 0
        assert result.failed_analyses == 0
        assert result.images_per_second == 0

    def test_parse_response_valid_json(self, service: CompositionService) -> None:
        """Test response parsing with valid JSON."""
        response = MOCK_RESPONSES["professional_headshot"]
        analysis = service._parse_response(response)

        assert isinstance(analysis, CompositionAnalysis)
        assert analysis.shot_type == ShotType.HEADSHOT
        assert analysis.confidence_score == 0.95

    def test_parse_response_malformed_json(self, service: CompositionService) -> None:
        """Test response parsing with malformed JSON."""
        response = "This contains closeup and outdoor but no valid JSON"
        analysis = service._parse_response(response)

        # Should use fallback parsing
        assert isinstance(analysis, CompositionAnalysis)
        assert analysis.shot_type == ShotType.CLOSEUP  # From fallback
        assert analysis.confidence_score == 0.5  # Fallback default

    def test_batch_statistics_calculation(self, service: CompositionService) -> None:
        """Test batch statistics calculation."""
        # Create mock results with different compositions
        from culora.domain.models.composition import CompositionResult

        results = []
        analyses = [
            CompositionAnalysis(
                shot_type=ShotType.CLOSEUP,
                scene_type=SceneType.INDOOR,
                lighting_quality=LightingQuality.EXCELLENT,
                background_complexity=BackgroundComplexity.SIMPLE,
                facial_expression=FacialExpression.CONFIDENT,
                camera_angle=CameraAngle.EYE_LEVEL,
                confidence_score=0.9,
                raw_description="Test 1",
            ),
            CompositionAnalysis(
                shot_type=ShotType.MEDIUM_SHOT,
                scene_type=SceneType.OUTDOOR,
                lighting_quality=LightingQuality.NATURAL,
                background_complexity=BackgroundComplexity.MODERATE,
                facial_expression=FacialExpression.RELAXED,
                camera_angle=CameraAngle.LOW_ANGLE,
                confidence_score=0.8,
                raw_description="Test 2",
            ),
        ]

        for i, analysis in enumerate(analyses):
            results.append(
                CompositionResult(
                    path=Path(f"test_{i}.jpg"),
                    success=True,
                    analysis=analysis,
                    analysis_duration=0.5,
                )
            )

        # Add one failed result
        results.append(
            CompositionResult(
                path=Path("failed.jpg"),
                success=False,
                error="Test error",
            )
        )

        successful_results = [r for r in results if r.success and r.analysis]

        batch_result = service._calculate_batch_statistics(
            results, successful_results, 2.0
        )

        # Verify statistics
        assert batch_result.successful_analyses == 2
        assert batch_result.failed_analyses == 1
        assert batch_result.total_duration == 2.0
        assert batch_result.images_per_second == 1.5  # 3 images / 2 seconds

        # Verify distributions
        assert batch_result.shot_type_distribution[ShotType.CLOSEUP] == 1
        assert batch_result.shot_type_distribution[ShotType.MEDIUM_SHOT] == 1
        assert batch_result.scene_type_distribution[SceneType.INDOOR] == 1
        assert batch_result.scene_type_distribution[SceneType.OUTDOOR] == 1

        # Verify confidence statistics (using approximate equality for floating point)
        assert abs(batch_result.mean_confidence - 0.85) < 0.001  # (0.9 + 0.8) / 2


class TestCompositionServiceGlobal:
    """Test cases for global composition service functions."""

    def test_get_composition_service_singleton(self) -> None:
        """Test that get_composition_service returns singleton."""
        config = CuLoRAConfig()
        service1 = get_composition_service(config)
        service2 = get_composition_service()

        assert service1 is service2

        # Reset global state for other tests
        import culora.services.composition_service

        culora.services.composition_service._composition_service = None

    def test_get_composition_service_with_config(self) -> None:
        """Test get_composition_service with explicit config."""
        config = CuLoRAConfig()
        service = get_composition_service(config)

        assert isinstance(service, CompositionService)
        assert service.config == config

        # Reset global state
        import culora.services.composition_service

        culora.services.composition_service._composition_service = None
