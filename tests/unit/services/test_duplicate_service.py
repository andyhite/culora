"""Tests for duplicate detection service - properly typed version."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from culora.domain.models.duplicate import (
    DuplicateAnalysis,
    DuplicateConfig,
    DuplicateGroup,
    DuplicateMatch,
    DuplicateRemovalStrategy,
    DuplicateThreshold,
    HashAlgorithm,
    ImageHash,
)
from culora.services.duplicate_service import DuplicateService, get_duplicate_service


class TestDuplicateService:
    """Test cases for DuplicateService."""

    @pytest.fixture
    def mock_config(self) -> DuplicateConfig:
        """Create mock duplicate detection configuration."""
        threshold = DuplicateThreshold(
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
            similarity_threshold=10,
            group_threshold=5,
        )
        return DuplicateConfig(
            threshold=threshold,
            removal_strategy=DuplicateRemovalStrategy.KEEP_HIGHEST_QUALITY,
        )

    @pytest.fixture
    def duplicate_service(self, mock_config: DuplicateConfig) -> DuplicateService:
        """Create DuplicateService instance."""
        return DuplicateService(mock_config)

    @pytest.fixture
    def sample_image_paths(self, tmp_path: Path) -> list[Path]:
        """Create sample image paths."""
        paths = []
        for i in range(5):
            path = tmp_path / f"image_{i}.jpg"
            path.touch()
            paths.append(path)
        return paths

    @patch("culora.services.duplicate_service.Image.open")
    @patch("culora.services.duplicate_service.imagehash.phash")
    def test_calculate_hash_success(
        self,
        mock_phash: Mock,
        mock_image_open: Mock,
        duplicate_service: DuplicateService,
        sample_image_paths: list[Path],
    ) -> None:
        """Test successful hash calculation."""
        # Setup mocks
        mock_image = Mock(spec=Image.Image)
        mock_image.mode = "RGB"
        mock_image_open.return_value.__enter__.return_value = mock_image
        mock_hash_obj = Mock()
        # Use setattr to avoid mypy method assignment error
        mock_hash_obj.__str__ = Mock(return_value="abcd1234efgh5678")  # type: ignore[method-assign]
        mock_phash.return_value = mock_hash_obj

        # Test hash calculation
        image_path = sample_image_paths[0]
        result = duplicate_service.calculate_hash(image_path)

        # Verify result
        assert isinstance(result, ImageHash)
        assert result.image_path == image_path
        assert result.hash_value == "abcd1234efgh5678"
        assert result.hash_algorithm == HashAlgorithm.PERCEPTUAL
        assert result.hash_size == 64  # 16 hex chars * 4 bits

    def test_get_hash_function(self, mock_config: DuplicateConfig) -> None:
        """Test hash function selection."""
        # Test different algorithms
        algorithms = [
            (HashAlgorithm.AVERAGE, "average_hash"),
            (HashAlgorithm.PERCEPTUAL, "phash"),
            (HashAlgorithm.DIFFERENCE, "dhash"),
            (HashAlgorithm.WAVELET, "whash"),
        ]

        for algorithm, expected_func in algorithms:
            mock_config.threshold.hash_algorithm = algorithm
            service = DuplicateService(mock_config)
            hash_func = service._get_hash_function()
            assert hash_func.__name__ == expected_func

    def test_find_duplicates_simple(self, duplicate_service: DuplicateService) -> None:
        """Test duplicate finding with simple mock setup."""
        # Create simple mock hashes
        mock_hashes = [
            ImageHash(
                image_path=Path("image1.jpg"),
                hash_value="0000000000000000",
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
                hash_size=64,
            ),
            ImageHash(
                image_path=Path("image2.jpg"),
                hash_value="0000000000000001",
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
                hash_size=64,
            ),
        ]

        with patch(
            "culora.services.duplicate_service.imagehash.hex_to_hash"
        ) as mock_hex_to_hash:
            # Create mock hash objects that return expected distances
            mock_hash1 = Mock()
            mock_hash2 = Mock()
            mock_hash1.__sub__ = Mock(return_value=1)  # Distance of 1

            def mock_hex_side_effect(hex_str: str) -> Mock:
                if hex_str == "0000000000000000":
                    return mock_hash1
                else:
                    return mock_hash2

            mock_hex_to_hash.side_effect = mock_hex_side_effect

            matches = duplicate_service.find_duplicates(mock_hashes)

            # Should find 1 match within threshold
            assert len(matches) == 1
            assert matches[0].hamming_distance == 1
            assert matches[0].is_near_duplicate
            assert not matches[0].is_exact_duplicate

    def test_group_duplicates_empty(self, duplicate_service: DuplicateService) -> None:
        """Test grouping with no matches."""
        groups = duplicate_service.group_duplicates([])
        assert groups == []

    @patch.object(DuplicateService, "calculate_batch_hashes")
    @patch.object(DuplicateService, "find_duplicates")
    @patch.object(DuplicateService, "group_duplicates")
    def test_analyze_duplicates_complete_workflow(
        self,
        mock_group_duplicates: Mock,
        mock_find_duplicates: Mock,
        mock_calculate_batch_hashes: Mock,
        duplicate_service: DuplicateService,
        sample_image_paths: list[Path],
    ) -> None:
        """Test complete duplicate analysis workflow."""
        # Setup mocks
        mock_hashes = [
            ImageHash(
                image_path=path,
                hash_value=f"hash_{i}",
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
                hash_size=64,
            )
            for i, path in enumerate(sample_image_paths)
        ]
        mock_calculate_batch_hashes.return_value = mock_hashes

        mock_matches = [
            DuplicateMatch(
                image1_path=sample_image_paths[0],
                image2_path=sample_image_paths[1],
                hamming_distance=0,  # Exact duplicate
                similarity_score=1.0,
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
            )
        ]
        mock_find_duplicates.return_value = mock_matches

        mock_groups = [
            DuplicateGroup(
                group_id="group_0001",
                image_paths=[sample_image_paths[0], sample_image_paths[1]],
                max_distance=0,
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
            )
        ]
        mock_group_duplicates.return_value = mock_groups

        # Test analysis
        result = duplicate_service.analyze_duplicates(sample_image_paths)

        # Verify result
        assert isinstance(result, DuplicateAnalysis)
        assert result.total_images == 5
        assert result.total_hashes == 5
        assert result.total_matches == 1
        assert result.total_groups == 1
        assert result.exact_duplicates == 1
        assert result.near_duplicates == 1  # Exact duplicates are also near duplicates
        assert len(result.unique_images) == 3  # 5 - 2 (in group)


class TestDuplicateServiceGlobal:
    """Test cases for global duplicate service management."""

    def test_get_duplicate_service_singleton(self) -> None:
        """Test singleton behavior of get_duplicate_service."""
        service1 = get_duplicate_service()
        service2 = get_duplicate_service()

        assert service1 is service2
        assert isinstance(service1, DuplicateService)


class TestDuplicateModels:
    """Test cases for duplicate detection domain models."""

    def test_duplicate_threshold_validation(self) -> None:
        """Test duplicate threshold validation."""
        # Valid threshold
        threshold = DuplicateThreshold(similarity_threshold=10, group_threshold=5)
        assert threshold.similarity_threshold == 10
        assert threshold.group_threshold == 5

        # Invalid threshold (group > similarity)
        with pytest.raises(ValueError, match="Group threshold .* must be <="):
            DuplicateThreshold(similarity_threshold=5, group_threshold=10)

    def test_image_hash_computed_fields(self) -> None:
        """Test ImageHash computed fields."""
        image_hash = ImageHash(
            image_path=Path("test.jpg"),
            hash_value="abcd1234",  # 8 hex chars = 32 bits
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
            hash_size=32,
        )

        assert image_hash.hash_bits == 32

    def test_duplicate_match_computed_fields(self) -> None:
        """Test DuplicateMatch computed fields."""
        # Exact duplicate
        exact_match = DuplicateMatch(
            image1_path=Path("img1.jpg"),
            image2_path=Path("img2.jpg"),
            hamming_distance=0,
            similarity_score=1.0,
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
        )
        assert exact_match.is_exact_duplicate
        assert exact_match.is_near_duplicate

        # Near duplicate
        near_match = DuplicateMatch(
            image1_path=Path("img1.jpg"),
            image2_path=Path("img2.jpg"),
            hamming_distance=3,
            similarity_score=0.95,
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
        )
        assert not near_match.is_exact_duplicate
        assert near_match.is_near_duplicate

        # Similar but not near
        similar_match = DuplicateMatch(
            image1_path=Path("img1.jpg"),
            image2_path=Path("img2.jpg"),
            hamming_distance=8,
            similarity_score=0.87,
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
        )
        assert not similar_match.is_exact_duplicate
        assert not similar_match.is_near_duplicate

    def test_duplicate_group_computed_fields(self) -> None:
        """Test DuplicateGroup computed fields."""
        paths = [Path(f"img{i}.jpg") for i in range(3)]
        group = DuplicateGroup(
            group_id="test_group",
            image_paths=paths,
            max_distance=5,
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
        )

        assert group.image_count == 3
        assert not group.has_representative

        # Select representative
        quality_scores = {str(paths[1]): 0.9, str(paths[0]): 0.7, str(paths[2]): 0.8}
        representative = group.select_representative(quality_scores)

        assert representative == paths[1]  # Highest quality
        assert group.has_representative
        assert group.representative_path == paths[1]  # type: ignore[unreachable]

    def test_duplicate_analysis_computed_fields(self) -> None:
        """Test DuplicateAnalysis computed fields."""
        unique_images = [Path(f"unique{i}.jpg") for i in range(3)]
        groups = [
            DuplicateGroup(
                group_id="group1",
                image_paths=[Path("dup1.jpg"), Path("dup2.jpg")],
                representative_path=Path("dup1.jpg"),
                max_distance=3,
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
            ),
            DuplicateGroup(
                group_id="group2",
                image_paths=[Path("dup3.jpg"), Path("dup4.jpg")],
                representative_path=Path("dup3.jpg"),
                max_distance=4,
                hash_algorithm=HashAlgorithm.PERCEPTUAL,
            ),
        ]

        analysis = DuplicateAnalysis(
            total_images=7,  # 3 unique + 4 in groups
            total_hashes=7,
            total_matches=2,
            total_groups=2,
            exact_duplicates=1,
            near_duplicates=2,
            hash_algorithm=HashAlgorithm.PERCEPTUAL,
            threshold_config=DuplicateThreshold(),
            duplicate_groups=groups,
            unique_images=unique_images,
        )

        # Test computed fields
        assert analysis.duplicate_rate == pytest.approx(57.1, abs=0.1)  # 4/7 * 100
        assert analysis.reduction_rate == pytest.approx(28.6, abs=0.1)  # (7-5)/7 * 100
        assert analysis.images_after_deduplication == 5  # 3 unique + 2 representatives

        # Test get_representative_images
        representatives = analysis.get_representative_images()
        assert len(representatives) == 5
        assert set(representatives) == {
            Path("unique0.jpg"),
            Path("unique1.jpg"),
            Path("unique2.jpg"),
            Path("dup1.jpg"),
            Path("dup3.jpg"),
        }
