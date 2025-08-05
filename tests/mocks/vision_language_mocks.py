"""Mock implementations for vision-language models used in testing."""

from typing import Any
from unittest.mock import MagicMock

import torch
from PIL import Image


class MockVisionLanguageModel:
    """Mock vision-language model for testing composition analysis."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        """Initialize mock model with predefined responses.

        Args:
            responses: Optional mapping of image descriptions to model responses
        """
        self.responses = responses or {}
        self.default_response = """{
    "shot_type": "medium_shot",
    "scene_type": "outdoor",
    "lighting_quality": "natural",
    "background_complexity": "moderate",
    "facial_expression": "relaxed",
    "camera_angle": "eye_level",
    "confidence": 0.85,
    "description": "Mock analysis of the provided image"
}"""
        self.call_count = 0
        self.last_prompt: str | None = None
        self.last_image: Image.Image | None = None

    def encode_image(self, image: Image.Image) -> Any:
        """Mock image encoding."""
        self.last_image = image
        return {"encoded": "mock_encoding"}

    def answer_question(self, encoded_image: Any, prompt: str, tokenizer: Any) -> str:
        """Mock question answering."""
        self.call_count += 1
        self.last_prompt = prompt

        # Return specific response if only one is provided (for targeted tests)
        if len(self.responses) == 1:
            return next(iter(self.responses.values()))

        # Return specific response if image description matches
        for description, response in self.responses.items():
            if description in prompt.lower():
                return response

        # Return response based on call count for deterministic testing
        responses = [
            """{
    "shot_type": "closeup",
    "scene_type": "indoor",
    "lighting_quality": "excellent",
    "background_complexity": "simple",
    "facial_expression": "confident",
    "camera_angle": "eye_level",
    "confidence": 0.92,
    "description": "Professional headshot with clean background"
}""",
            """{
    "shot_type": "full_body",
    "scene_type": "outdoor",
    "lighting_quality": "natural",
    "background_complexity": "complex",
    "facial_expression": "happy",
    "camera_angle": "low_angle",
    "confidence": 0.78,
    "description": "Full body outdoor portrait with natural lighting"
}""",
            """{
    "shot_type": "medium_shot",
    "scene_type": "studio",
    "lighting_quality": "dramatic",
    "background_complexity": "simple",
    "facial_expression": "serious",
    "camera_angle": "high_angle",
    "confidence": 0.88,
    "description": "Studio portrait with dramatic lighting"
}""",
        ]

        if self.call_count <= len(responses):
            return responses[self.call_count - 1]

        return self.default_response

    def to(self, device: torch.device) -> "MockVisionLanguageModel":
        """Mock device movement."""
        return self

    def eval(self) -> "MockVisionLanguageModel":
        """Mock evaluation mode."""
        return self


class MockTokenizer:
    """Mock tokenizer for vision-language models."""

    def __init__(self) -> None:
        """Initialize mock tokenizer."""
        self.call_count = 0

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> "MockTokenizer":
        """Mock tokenizer creation."""
        return cls()


def create_mock_vision_language_model() -> MagicMock:
    """Create a mock vision-language model for testing.

    Returns:
        Configured MagicMock that behaves like a vision-language model
    """
    mock_model = MagicMock()
    mock_instance = MockVisionLanguageModel()

    # Configure the mock to return our mock instance
    mock_model.from_pretrained.return_value = mock_instance
    mock_model.return_value = mock_instance

    return mock_model


def create_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for testing.

    Returns:
        Configured MagicMock that behaves like a tokenizer
    """
    mock_tokenizer = MagicMock()
    mock_instance = MockTokenizer()

    # Configure the mock to return our mock instance
    mock_tokenizer.from_pretrained.return_value = mock_instance
    mock_tokenizer.return_value = mock_instance

    return mock_tokenizer


# Common test responses for different scenarios
MOCK_RESPONSES = {
    "professional_headshot": """{
    "shot_type": "headshot",
    "scene_type": "studio",
    "lighting_quality": "excellent",
    "background_complexity": "simple",
    "facial_expression": "confident",
    "camera_angle": "eye_level",
    "confidence": 0.95,
    "description": "Professional headshot with clean studio lighting"
}""",
    "outdoor_portrait": """{
    "shot_type": "medium_shot",
    "scene_type": "outdoor",
    "lighting_quality": "natural",
    "background_complexity": "moderate",
    "facial_expression": "relaxed",
    "camera_angle": "eye_level",
    "confidence": 0.82,
    "description": "Outdoor portrait in natural lighting"
}""",
    "dramatic_closeup": """{
    "shot_type": "closeup",
    "scene_type": "indoor",
    "lighting_quality": "dramatic",
    "background_complexity": "simple",
    "facial_expression": "intense",
    "camera_angle": "low_angle",
    "confidence": 0.89,
    "description": "Dramatic close-up with moody lighting"
}""",
    "parsing_error": "This is not valid JSON and should trigger fallback parsing",
    "low_confidence": """{
    "shot_type": "unknown",
    "scene_type": "unknown",
    "lighting_quality": "unknown",
    "background_complexity": "unknown",
    "facial_expression": null,
    "camera_angle": null,
    "confidence": 0.3,
    "description": "Unclear image with low confidence analysis"
}""",
}
