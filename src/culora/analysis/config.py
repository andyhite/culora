"""Analysis stage configuration definitions for CuLoRA."""

from culora.models.analysis import AnalysisStage, StageConfig


def get_deduplication_config() -> StageConfig:
    """Get current deduplication stage configuration."""
    return StageConfig(
        stage=AnalysisStage.DEDUPLICATION,
        config={"algorithm": "dhash", "hash_size": "8", "threshold": "2"},
        version="1.0",
    )


def get_quality_config() -> StageConfig:
    """Get current quality assessment stage configuration."""
    return StageConfig(
        stage=AnalysisStage.QUALITY,
        config={
            "sharpness_threshold": "150",
            "brightness_min": "60",
            "brightness_max": "200",
            "contrast_threshold": "40",
        },
        version="1.0",
    )


def get_face_config() -> StageConfig:
    """Get current face detection stage configuration."""
    return StageConfig(
        stage=AnalysisStage.FACE,
        config={
            "confidence_threshold": "0.5",
            "model_name": "yolo11n.pt",
            "max_detections": "10",
            "iou_threshold": "0.5",
            "use_half_precision": "true",
            "device": "auto",
        },
        version="2.0",
    )


def get_all_stage_configs() -> dict[AnalysisStage, StageConfig]:
    """Get all available stage configurations."""
    return {
        AnalysisStage.DEDUPLICATION: get_deduplication_config(),
        AnalysisStage.QUALITY: get_quality_config(),
        AnalysisStage.FACE: get_face_config(),
    }


def get_enabled_stage_configs(enabled_stages: list[AnalysisStage]) -> list[StageConfig]:
    """Get configurations for enabled stages only."""
    all_configs = get_all_stage_configs()
    return [all_configs[stage] for stage in enabled_stages if stage in all_configs]
