"""File utilities for testing."""

import json
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch


class TempFileHelper:
    """Helper for creating temporary files and directories."""

    @staticmethod
    @contextmanager
    def create_config_file(
        config_data: dict[str, Any], suffix: str = ".json"
    ) -> Generator[Path, None, None]:
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            if suffix == ".json":
                json.dump(config_data, f, indent=2)
            elif suffix in [".yaml", ".yml"]:
                import yaml

                yaml.dump(config_data, f, default_flow_style=False)
            else:
                f.write(str(config_data))

            temp_path = Path(f.name)

        try:
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    @contextmanager
    def create_temp_dir() -> Generator[Path, None, None]:
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @staticmethod
    @contextmanager
    def create_temp_file(suffix: str = ".txt") -> Generator[Path, None, None]:
        """Create a temporary file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            temp_path = Path(f.name)

        try:
            yield temp_path
        finally:
            temp_path.unlink(missing_ok=True)


def patch_environment(**env_vars: str) -> Any:
    """Create a context manager to patch environment variables."""
    import os

    return patch.dict(os.environ, env_vars)
