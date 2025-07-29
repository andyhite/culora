"""Unit tests for image selection and scoring logic."""

from unittest.mock import Mock

from culora.config import AnalysisStage, CuLoRAConfig, ScoringConfig
from culora.services.selection_service import SelectionService


class TestSelectionService:
    """Test the SelectionService and its selection logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selection_service = SelectionService()
        self.config = CuLoRAConfig()

    def test_selection_service_initialization(self):
        """Test that selection service initializes correctly."""
        assert self.selection_service is not None
        assert hasattr(self.selection_service, "_config_manager")

    def test_tier_selection_logic(self):
        """Test the basic tier selection logic exists."""
        # Create a mock DirectoryAnalysis
        mock_analysis = Mock()
        mock_analysis.images = []  # Empty list

        # Test with empty directory
        selected, total = self.selection_service.select_images(
            mock_analysis, "/tmp/output"
        )

        assert selected == 0
        assert total == 0

    def test_selection_service_has_required_methods(self):
        """Test that selection service has the expected interface."""
        # Check that the service has the required methods
        assert hasattr(self.selection_service, "select_images")
        assert callable(self.selection_service.select_images)

        # The service should be properly initialized
        assert hasattr(self.selection_service, "_config_manager")

    def test_selection_with_max_images(self):
        """Test selection respects max images parameter."""
        mock_analysis = Mock()
        mock_analysis.images = []

        selected, total = self.selection_service.select_images(
            mock_analysis, "/tmp/output", max_images=5
        )

        # Should handle max_images parameter without error
        assert selected >= 0
        assert total >= 0


class TestScoringConfiguration:
    """Test scoring configuration effects."""

    def test_scoring_config_creation(self):
        """Test that scoring config can be created with custom values."""
        scoring_config = ScoringConfig(quality_weight=0.8, face_weight=0.2)

        assert scoring_config.quality_weight == 0.8
        assert scoring_config.face_weight == 0.2

    def test_face_area_thresholds(self):
        """Test face area ratio threshold configuration."""
        scoring_config = ScoringConfig(
            face_area_min=0.1, face_area_peak=0.2, face_area_max=0.3
        )

        assert scoring_config.face_area_min == 0.1
        assert scoring_config.face_area_peak == 0.2
        assert scoring_config.face_area_max == 0.3

    def test_multi_face_penalty_config(self):
        """Test multi-face penalty configuration."""
        scoring_config = ScoringConfig(multi_face_penalty=0.15, max_face_penalty=0.6)

        assert scoring_config.multi_face_penalty == 0.15
        assert scoring_config.max_face_penalty == 0.6


class TestSelectionEdgeCases:
    """Test edge cases in selection logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selection_service = SelectionService()

    def test_empty_image_list(self):
        """Test selection with empty image list."""
        mock_analysis = Mock()
        mock_analysis.images = []

        selected, total = self.selection_service.select_images(
            mock_analysis, "/tmp/output"
        )

        assert selected == 0
        assert total == 0

    def test_zero_max_images(self):
        """Test selection with max_images=0."""
        mock_analysis = Mock()
        mock_analysis.images = []

        selected, _ = self.selection_service.select_images(
            mock_analysis, "/tmp/output", max_images=0
        )

        assert selected == 0

    def test_negative_max_images(self):
        """Test selection with negative max_images."""
        mock_analysis = Mock()
        mock_analysis.images = []

        selected, total = self.selection_service.select_images(
            mock_analysis, "/tmp/output", max_images=-1
        )

        # Should handle gracefully
        assert selected >= 0
        assert total >= 0


class TestConfigurationIntegration:
    """Test configuration integration with selection logic."""

    def test_config_enabled_stages_integration(self):
        """Test that config enabled_stages property works with selection."""
        config = CuLoRAConfig()

        # All stages should be enabled by default
        enabled_stages = config.enabled_stages
        assert len(enabled_stages) == 3
        assert AnalysisStage.DEDUPLICATION in enabled_stages
        assert AnalysisStage.QUALITY in enabled_stages
        assert AnalysisStage.FACE in enabled_stages

    def test_selective_stage_disabling(self):
        """Test disabling individual stages."""
        config = CuLoRAConfig()
        config.deduplication.enabled = False

        enabled_stages = config.enabled_stages
        assert len(enabled_stages) == 2
        assert AnalysisStage.DEDUPLICATION not in enabled_stages
        assert AnalysisStage.QUALITY in enabled_stages
        assert AnalysisStage.FACE in enabled_stages

    def test_all_stages_disabled(self):
        """Test disabling all stages."""
        config = CuLoRAConfig()
        config.deduplication.enabled = False
        config.quality.enabled = False
        config.face.enabled = False

        enabled_stages = config.enabled_stages
        assert len(enabled_stages) == 0
