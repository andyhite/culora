"""Composition analysis configuration."""

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class CompositionConfig(BaseModel):
    """Configuration for composition analysis using vision-language models."""

    # Model settings
    model_name: str = Field(
        default="vikhyatk/moondream2",
        description="Hugging Face model identifier for vision-language model",
    )
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home()
        / "Library"
        / "Application Support"
        / "culora"
        / "composition_models",
        description="Directory to cache composition analysis models",
    )
    device_preference: str = Field(
        default="auto", description="Device preference: 'auto', 'cuda', 'mps', or 'cpu'"
    )

    # Analysis settings
    enable_shot_type_analysis: bool = Field(
        default=True, description="Enable shot type classification"
    )
    enable_scene_analysis: bool = Field(
        default=True, description="Enable scene type analysis"
    )
    enable_lighting_analysis: bool = Field(
        default=True, description="Enable lighting quality assessment"
    )
    enable_background_analysis: bool = Field(
        default=True, description="Enable background complexity analysis"
    )
    enable_expression_analysis: bool = Field(
        default=True, description="Enable facial expression analysis"
    )
    enable_angle_analysis: bool = Field(
        default=True, description="Enable camera angle analysis"
    )

    # Confidence and quality settings
    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for accepting analysis results",
    )
    enable_confidence_scoring: bool = Field(
        default=True, description="Enable confidence scoring for analysis results"
    )

    # Performance settings
    batch_size: int = Field(
        default=4, ge=1, le=16, description="Batch size for model inference"
    )
    max_image_size: tuple[int, int] = Field(
        default=(1024, 1024),
        description="Maximum image size for analysis (width, height)",
    )
    enable_model_caching: bool = Field(
        default=True, description="Cache loaded models in memory"
    )
    memory_optimization: bool = Field(
        default=True, description="Enable memory optimization techniques"
    )

    # Response parsing settings
    max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum retries for failed analyses"
    )
    response_timeout: float = Field(
        default=30.0, ge=5.0, description="Timeout for model responses in seconds"
    )
    enable_fallback_parsing: bool = Field(
        default=True,
        description="Enable fallback parsing when structured response fails",
    )

    # Prompt engineering settings
    use_structured_prompts: bool = Field(
        default=True, description="Use structured prompts for consistent outputs"
    )
    prompt_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation (lower = more consistent)",
    )
    enable_example_prompts: bool = Field(
        default=True, description="Include examples in prompts for better consistency"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders: ClassVar[dict[type, Any]] = {Path: str}
        use_enum_values = True


# Composition analysis prompts
COMPOSITION_ANALYSIS_PROMPT = """Analyze this image and classify its composition characteristics. Be specific and choose the most appropriate category for each aspect.

SHOT TYPE - Choose the framing that best describes how the subject is positioned:
• extreme_closeup - Very tight crop on face/details
• closeup - Face and shoulders visible
• medium_closeup - Head to chest visible
• medium_shot - Waist up
• medium_long_shot - Knees up
• long_shot - Full body with some background
• extreme_long_shot - Subject small in frame
• portrait - Traditional portrait framing
• headshot - Professional headshot style
• full_body - Complete body visible

SCENE TYPE - Identify the environment:
• indoor - Inside a building or enclosed space
• outdoor - Outside in open air
• studio - Professional photography studio setting
• natural - Natural outdoor environment
• urban - City or urban environment
• interior - Indoor architectural or designed space

LIGHTING QUALITY - Assess the lighting:
• excellent - Professional, well-balanced lighting
• good - Clear, adequate lighting
• fair - Decent but not optimal lighting
• poor - Insufficient or problematic lighting
• harsh - Strong, direct lighting creating harsh shadows
• soft - Gentle, diffused lighting
• dramatic - Moody lighting with strong contrasts
• natural - Natural sunlight or daylight
• artificial - Indoor/electric lighting

BACKGROUND COMPLEXITY - Evaluate background detail:
• simple - Plain, minimal background
• moderate - Some detail but not distracting
• complex - Detailed background with many elements
• cluttered - Busy, messy background
• clean - Tidy, organized background
• bokeh - Blurred/out-of-focus background

FACIAL EXPRESSION (if person visible):
• neutral - No strong expression
• happy - Positive, joyful expression
• serious - Formal, stern expression
• smiling - Warm, friendly smile
• laughing - Active laughter or joy
• contemplative - Thoughtful, reflective
• confident - Self-assured appearance
• relaxed - Calm, at ease
• intense - Strong, focused expression

CAMERA ANGLE (if determinable):
• eye_level - Camera at subject's eye level
• low_angle - Camera below subject looking up
• high_angle - Camera above subject looking down
• dutch_angle - Tilted camera angle
• overhead - Camera directly above
• straight_on - Direct frontal view

Respond ONLY with valid JSON using actual category names (not placeholder text):
{
  "shot_type": "closeup",
  "scene_type": "indoor",
  "lighting_quality": "good",
  "background_complexity": "simple",
  "facial_expression": "neutral",
  "camera_angle": "eye_level",
  "confidence": 0.85,
  "description": "A closeup portrait of a person with neutral expression in indoor lighting"
}"""

COMPOSITION_EXAMPLES: list[dict[str, Any]] = [
    {
        "description": "Professional headshot with clean background",
        "expected_response": {
            "shot_type": "headshot",
            "scene_type": "studio",
            "lighting_quality": "excellent",
            "background_complexity": "simple",
            "facial_expression": "confident",
            "camera_angle": "eye_level",
            "confidence": 0.95,
            "description": "Professional headshot with clean studio lighting",
        },
    },
    {
        "description": "Full body outdoor portrait",
        "expected_response": {
            "shot_type": "full_body",
            "scene_type": "outdoor",
            "lighting_quality": "natural",
            "background_complexity": "moderate",
            "facial_expression": "relaxed",
            "camera_angle": "eye_level",
            "confidence": 0.88,
            "description": "Full body portrait in natural outdoor setting",
        },
    },
    {
        "description": "Close-up with dramatic lighting",
        "expected_response": {
            "shot_type": "closeup",
            "scene_type": "indoor",
            "lighting_quality": "dramatic",
            "background_complexity": "simple",
            "facial_expression": "intense",
            "camera_angle": "low_angle",
            "confidence": 0.82,
            "description": "Dramatic close-up portrait with moody lighting",
        },
    },
]
