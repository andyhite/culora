"""Basic CLI integration tests for CuLoRA."""

import tempfile
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from culora.main import app


class TestBasicCLI:
    """Basic CLI functionality tests."""

    def test_help_command(self):
        """Test that help command works."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "analyze" in result.output

    def test_version_command(self):
        """Test that version command works."""
        runner = CliRunner()
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0

    def test_config_help(self):
        """Test config command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "show" in result.output

    def test_analyze_help(self):
        """Test analyze command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "input-dir" in result.output or "Directory" in result.output

    def test_analyze_nonexistent_directory(self):
        """Test analyze command with non-existent directory."""
        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "/nonexistent/directory"])

        # Should fail gracefully
        assert result.exit_code != 0

    def test_analyze_empty_directory(self):
        """Test analyze command with empty directory."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(app, ["analyze", temp_dir])

            # Should handle empty directory gracefully (empty table)
            assert result.exit_code == 0
            assert "Image Analysis Results" in result.output

    def test_analyze_with_real_images(self):
        """Test analyze command with actual image files."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a few test images
            for i in range(2):
                img = Image.new("RGB", (100, 100), color=(255, 0, 0))
                img.save(temp_path / f"test_{i}.jpg")

            result = runner.invoke(app, ["analyze", temp_dir])

            # Should complete successfully with real images
            assert result.exit_code == 0
            assert "test_0.jpg" in result.output or "test_1.jpg" in result.output
