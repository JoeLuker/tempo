"""Unit tests for RetroactiveRemovalCoordinator."""

import pytest
from unittest.mock import Mock

from src.domain.services.retroactive_removal_coordinator import RetroactiveRemovalCoordinator


class TestRetroactiveRemovalCoordinator:
    """Tests for RetroactiveRemovalCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator instance."""
        return RetroactiveRemovalCoordinator(debug_mode=False)

    @pytest.fixture
    def sample_token_sets(self):
        """Create sample token sets for testing."""
        return {
            0: [(10, 0.8), (11, 0.6)],  # Step 0: two tokens
            1: [(20, 0.9)],              # Step 1: one token
            2: [(30, 0.7), (31, 0.5)]   # Step 2: two tokens
        }

    def test_apply_retroactive_removal_with_new_method(self, coordinator, sample_token_sets):
        """Test retroactive removal using retroactively_remove method."""
        # Create mock remover with new method
        mock_remover = Mock()
        # Mock returns only surviving tokens (removed token 11 from step 0)
        surviving_sets = {
            0: [(10, 0.8)],  # Removed (11, 0.6)
            1: [(20, 0.9)],
            2: [(30, 0.7), (31, 0.5)]
        }
        mock_remover.retroactively_remove.return_value = surviving_sets

        result = coordinator.apply_retroactive_removal(
            remover=mock_remover,
            prompt_length=5,
            all_token_sets=sample_token_sets,
            current_step=2
        )

        # Should have called the method
        mock_remover.retroactively_remove.assert_called_once_with(
            prompt_length=5,
            all_parallel_tokens=sample_token_sets,
            step=2
        )

        # Should return surviving sets
        assert result == surviving_sets

    def test_apply_retroactive_removal_with_old_method(self, coordinator, sample_token_sets):
        """Test retroactive removal using retroactively_prune method (legacy)."""
        # Create mock remover with only old method
        mock_remover = Mock(spec=[])  # Start with no methods
        mock_remover.retroactively_prune = Mock()
        surviving_sets = {
            0: [(10, 0.8)],
            1: [(20, 0.9)],
            2: [(30, 0.7)]
        }
        mock_remover.retroactively_prune.return_value = surviving_sets

        result = coordinator.apply_retroactive_removal(
            remover=mock_remover,
            prompt_length=5,
            all_token_sets=sample_token_sets,
            current_step=2
        )

        # Should have called the old method
        mock_remover.retroactively_prune.assert_called_once()

        # Should return surviving sets
        assert result == surviving_sets

    def test_apply_retroactive_removal_no_method(self, coordinator, sample_token_sets):
        """Test that remover without required methods returns empty dict."""
        mock_remover = Mock(spec=[])  # No retroactive removal methods

        result = coordinator.apply_retroactive_removal(
            remover=mock_remover,
            prompt_length=5,
            all_token_sets=sample_token_sets,
            current_step=2
        )

        # Should return empty dict
        assert result == {}

    def test_apply_retroactive_removal_with_update_step(self, coordinator, sample_token_sets):
        """Test that coordinator calls update_step if available."""
        mock_remover = Mock()
        mock_remover.update_step = Mock()
        mock_remover.retroactively_remove.return_value = sample_token_sets

        coordinator.apply_retroactive_removal(
            remover=mock_remover,
            prompt_length=5,
            all_token_sets=sample_token_sets,
            current_step=3
        )

        # Should have called update_step
        mock_remover.update_step.assert_called_once_with(3)

    def test_apply_retroactive_removal_error_handling(self, coordinator, sample_token_sets):
        """Test that errors are caught and empty dict returned."""
        mock_remover = Mock()
        mock_remover.retroactively_remove.side_effect = Exception("Test error")

        result = coordinator.apply_retroactive_removal(
            remover=mock_remover,
            prompt_length=5,
            all_token_sets=sample_token_sets,
            current_step=2
        )

        # Should return empty dict on error
        assert result == {}

    def test_count_removed_tokens_basic(self, coordinator):
        """Test counting removed tokens."""
        original_sets = {
            0: [(10, 0.8), (11, 0.6)],
            1: [(20, 0.9), (21, 0.7)]
        }
        surviving_sets = {
            0: [(10, 0.8)],       # Removed 11
            1: [(20, 0.9)]        # Removed 21
        }

        count = coordinator._count_removed_tokens(original_sets, surviving_sets, 2)

        assert count == 2

    def test_count_removed_tokens_no_removals(self, coordinator):
        """Test counting when no tokens removed."""
        original_sets = {
            0: [(10, 0.8)],
            1: [(20, 0.9)]
        }
        surviving_sets = original_sets.copy()

        count = coordinator._count_removed_tokens(original_sets, surviving_sets, 2)

        assert count == 0

    def test_count_removed_tokens_empty_sets(self, coordinator):
        """Test counting with empty sets."""
        count = coordinator._count_removed_tokens({}, {}, 2)

        assert count == 0

    def test_count_removed_tokens_partial_steps(self, coordinator):
        """Test counting removals only up to specified step."""
        original_sets = {
            0: [(10, 0.8), (11, 0.6)],
            1: [(20, 0.9), (21, 0.7)],
            2: [(30, 0.8), (31, 0.6)]
        }
        surviving_sets = {
            0: [(10, 0.8)],
            1: [(20, 0.9)],
            2: [(30, 0.8)]
        }

        # Count only up to step 2 (not including step 2)
        count = coordinator._count_removed_tokens(original_sets, surviving_sets, 2)

        # Should count removals from steps 0 and 1, but not step 2
        assert count == 2
