"""Mock implementations for PIQ (PyTorch Image Quality) library."""

from unittest.mock import MagicMock

import torch


class PIQMocks:
    """Mock implementations for PIQ functionality."""

    @staticmethod
    def create_brisque_mock(return_value: float = 25.5) -> MagicMock:
        """Create a mock for PIQ BRISQUE function.

        Args:
            return_value: BRISQUE score to return (default: 25.5, good quality)

        Returns:
            Mock function that returns a tensor with the specified value
        """
        mock_brisque = MagicMock()
        mock_tensor = torch.tensor(return_value)
        mock_brisque.return_value = mock_tensor
        return mock_brisque

    @staticmethod
    def create_failing_brisque_mock(
        error_message: str = "BRISQUE calculation failed",
    ) -> MagicMock:
        """Create a mock for PIQ BRISQUE function that raises an exception.

        Args:
            error_message: Error message to use in exception

        Returns:
            Mock function that raises an exception when called
        """
        mock_brisque = MagicMock()
        mock_brisque.side_effect = RuntimeError(error_message)
        return mock_brisque

    @staticmethod
    def create_variable_brisque_mock(scores: list[float]) -> MagicMock:
        """Create a mock that returns different BRISQUE scores on successive calls.

        Args:
            scores: List of BRISQUE scores to return in order

        Returns:
            Mock function that cycles through the provided scores
        """
        mock_brisque = MagicMock()
        tensors = [torch.tensor(score) for score in scores]
        mock_brisque.side_effect = tensors
        return mock_brisque

    @staticmethod
    def create_piq_module_mock(brisque_return_value: float = 25.5) -> MagicMock:
        """Create a complete mock of the PIQ module.

        Args:
            brisque_return_value: Default BRISQUE score to return

        Returns:
            Mock PIQ module with BRISQUE function
        """
        mock_piq = MagicMock()
        mock_piq.brisque = PIQMocks.create_brisque_mock(brisque_return_value)
        return mock_piq
