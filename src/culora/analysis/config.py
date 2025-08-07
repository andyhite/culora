"""Analysis stage configuration definitions for CuLoRA."""

from typing import Any

from culora.models.analysis import AnalysisStage, StageConfig

# Data-driven stage configuration definitions
STAGE_CONFIGS: dict[AnalysisStage, dict[str, Any]] = {
    AnalysisStage.DEDUPLICATION: {
        "config": {"algorithm": "dhash", "hash_size": "8", "threshold": "2"},
        "version": "1.0",
    },
    AnalysisStage.QUALITY: {
        "config": {
            "sharpness_threshold": "150",
            "brightness_min": "60",
            "brightness_max": "200",
            "contrast_threshold": "40",
        },
        "version": "1.0",
    },
    AnalysisStage.FACE: {
        "config": {
            "confidence_threshold": "0.5",
            "model_name": "yolo11n.pt",
            "max_detections": "10",
            "iou_threshold": "0.5",
            "use_half_precision": "true",
            "device": "auto",
        },
        "version": "2.0",
    },
}


def get_stage_config(stage: AnalysisStage) -> StageConfig:
    """Get configuration for a specific stage.

    Args:
        stage: The analysis stage to get configuration for.

    Returns:
        StageConfig for the specified stage.

    Raises:
        KeyError: If the stage is not configured.
    """
    config_data = STAGE_CONFIGS[stage]
    return StageConfig(
        stage=stage,
        config=config_data["config"],
        version=config_data["version"],
    )


def get_all_stage_configs() -> dict[AnalysisStage, StageConfig]:
    """Get all available stage configurations."""
    return {stage: get_stage_config(stage) for stage in STAGE_CONFIGS}


def get_enabled_stage_configs(enabled_stages: list[AnalysisStage]) -> list[StageConfig]:
    """Get configurations for enabled stages only."""
    return [
        get_stage_config(stage) for stage in enabled_stages if stage in STAGE_CONFIGS
    ]
