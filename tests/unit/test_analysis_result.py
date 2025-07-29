"""Unit tests for AnalysisResult model."""

from culora.config import AnalysisStage
from culora.models.analysis_result import AnalysisResult
from culora.models.duplicate_detection_result import DuplicateDetectionResult
from culora.models.face_detection_result import Face, FaceDetectionResult
from culora.models.image_quality_result import ImageQualityResult


class TestAnalysisResult:
    """Tests for AnalysisResult model functionality."""

    def test_get_method_with_quality(self) -> None:
        """Test get method with quality stage."""
        analysis = AnalysisResult()
        quality_result = ImageQualityResult(
            sharpness_score=200.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=0.8,
        )
        analysis.set_quality(quality_result)

        result = analysis.get(AnalysisStage.QUALITY)
        assert result is quality_result

    def test_get_method_with_face(self) -> None:
        """Test get method with face stage."""
        analysis = AnalysisResult()
        faces = [Face(bounding_box=(10, 10, 50, 50), confidence=0.8)]
        face_result = FaceDetectionResult(
            faces=faces,
            model_used="test_model",
            device_used="cpu",
        )
        analysis.set_face(face_result)

        result = analysis.get(AnalysisStage.FACE)
        assert result is face_result

    def test_get_method_with_deduplication(self) -> None:
        """Test get method with deduplication stage."""
        analysis = AnalysisResult()
        dedup_result = DuplicateDetectionResult(hash_value="abc123")
        analysis.set_deduplication(dedup_result)

        result = analysis.get(AnalysisStage.DEDUPLICATION)
        assert result is dedup_result

    def test_get_method_returns_none_for_unset_stage(self) -> None:
        """Test get method returns None for unset stages."""
        analysis = AnalysisResult()

        assert analysis.get(AnalysisStage.QUALITY) is None
        assert analysis.get(AnalysisStage.FACE) is None
        assert analysis.get(AnalysisStage.DEDUPLICATION) is None

    def test_has_stage_method(self) -> None:
        """Test has_stage method."""
        analysis = AnalysisResult()

        # Initially no stages
        assert not analysis.has_stage(AnalysisStage.QUALITY)
        assert not analysis.has_stage(AnalysisStage.FACE)
        assert not analysis.has_stage(AnalysisStage.DEDUPLICATION)

        # Add a quality result
        quality_result = ImageQualityResult(
            sharpness_score=200.0,
            brightness_score=120.0,
            contrast_score=60.0,
            composite_score=0.8,
        )
        analysis.set_quality(quality_result)

        assert analysis.has_stage(AnalysisStage.QUALITY)
        assert not analysis.has_stage(AnalysisStage.FACE)
        assert not analysis.has_stage(AnalysisStage.DEDUPLICATION)

    def test_contains_method(self) -> None:
        """Test __contains__ method (in operator)."""
        analysis = AnalysisResult()

        # Initially no stages
        assert AnalysisStage.QUALITY not in analysis
        assert AnalysisStage.FACE not in analysis
        assert AnalysisStage.DEDUPLICATION not in analysis

        # Add a face result
        faces = [Face(bounding_box=(10, 10, 50, 50), confidence=0.8)]
        face_result = FaceDetectionResult(
            faces=faces,
            model_used="test_model",
            device_used="cpu",
        )
        analysis.set_face(face_result)

        assert AnalysisStage.QUALITY not in analysis
        assert AnalysisStage.FACE in analysis
        assert AnalysisStage.DEDUPLICATION not in analysis
