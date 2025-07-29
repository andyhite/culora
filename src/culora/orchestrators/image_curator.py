"""Image curator orchestrator for CuLoRA."""

from culora.managers.config_manager import ConfigManager
from culora.models.directory_analysis import DirectoryAnalysis
from culora.services.selection_service import SelectionService
from culora.utils.console import get_console

console = get_console()


class ImageCurator:
    """Orchestrator for curating images using selection services."""

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        selection_service: SelectionService | None = None,
    ) -> None:
        """Initialize the image curator.

        Args:
            config_manager: Configuration manager instance. If None, uses singleton.
            selection_service: Selection service. If None, creates new instance.
        """
        self._config_manager = config_manager or ConfigManager.get_instance()
        self._selection_service = selection_service or SelectionService(
            self._config_manager
        )

    def select_images(
        self,
        analysis: DirectoryAnalysis,
        output_dir: str,
        draw_boxes: bool = False,
        dry_run: bool = False,
        max_images: int | None = None,
    ) -> tuple[int, int]:
        """Select and copy curated images.

        Args:
            analysis: Directory analysis results.
            output_dir: Directory to copy selected images to.
            draw_boxes: Whether to draw bounding boxes on faces.
            dry_run: Whether to perform a dry run.
            max_images: Maximum number of images to select (top N by score).

        Returns:
            Tuple of (selected_count, total_count).
        """
        return self._selection_service.select_images(
            analysis=analysis,
            output_dir=output_dir,
            draw_boxes=draw_boxes,
            dry_run=dry_run,
            max_images=max_images,
        )
