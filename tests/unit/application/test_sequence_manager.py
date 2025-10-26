"""Unit tests for SequenceManager (application layer)."""

import pytest
from unittest.mock import Mock

from src.application.services.sequence_manager import SequenceManager, SequenceMetrics


class TestSequenceManager:
    """Tests for SequenceManager."""

    @pytest.fixture
    def manager(self):
        """Create sequence manager instance."""
        return SequenceManager(debug_mode=False)

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.metrics.sequence_length == 0
        assert manager.metrics.initial_prompt_length == 0
        assert manager.metrics.step_count == 0
        assert manager.callbacks == []

    def test_initialize_with_prompt_length(self, manager):
        """Test initializing with prompt length."""
        manager.initialize(10)

        assert manager.metrics.initial_prompt_length == 10
        assert manager.metrics.sequence_length == 0
        assert manager.metrics.step_count == 0

    def test_initialize_with_negative_prompt_length(self, manager):
        """Test that negative prompt length raises error."""
        with pytest.raises(ValueError):
            manager.initialize(-5)

    def test_update_with_new_tokens(self, manager):
        """Test updating sequence with new tokens."""
        manager.initialize(10)

        result = manager.update(3)

        assert result is True
        assert manager.metrics.sequence_length == 3
        assert manager.metrics.step_count == 1
        assert len(manager.metrics.sequence_length_history) == 1
        assert manager.metrics.sequence_length_history[0] == 3

    def test_update_with_zero_tokens(self, manager):
        """Test that updating with zero tokens returns False."""
        manager.initialize(10)

        result = manager.update(0)

        assert result is False
        assert manager.metrics.sequence_length == 0
        assert manager.metrics.step_count == 0

    def test_update_with_negative_tokens(self, manager):
        """Test that updating with negative tokens returns False."""
        manager.initialize(10)

        result = manager.update(-5)

        assert result is False
        assert manager.metrics.sequence_length == 0

    def test_multiple_updates(self, manager):
        """Test multiple sequential updates."""
        manager.initialize(10)

        manager.update(2)
        manager.update(3)
        manager.update(1)

        assert manager.metrics.sequence_length == 6
        assert manager.metrics.step_count == 3
        assert manager.metrics.sequence_length_history == [2, 5, 6]

    def test_get_current_length(self, manager):
        """Test getting current sequence length."""
        manager.initialize(10)
        manager.update(5)

        assert manager.get_current_length() == 5

    def test_get_total_length(self, manager):
        """Test getting total length including prompt."""
        manager.initialize(10)
        manager.update(5)

        assert manager.get_total_length() == 15

    def test_get_step_count(self, manager):
        """Test getting step count."""
        manager.initialize(10)
        manager.update(2)
        manager.update(3)

        assert manager.get_step_count() == 2

    def test_add_callback(self, manager):
        """Test adding a callback."""
        callback = Mock()

        manager.add_callback(callback)

        assert len(manager.callbacks) == 1
        assert callback in manager.callbacks

    def test_callback_invocation(self, manager):
        """Test that callbacks are invoked on update."""
        callback = Mock()
        manager.add_callback(callback)
        manager.initialize(10)

        manager.update(3)

        callback.assert_called_once_with(3, 1, 10)

    def test_multiple_callbacks(self, manager):
        """Test multiple callbacks are all invoked."""
        callback1 = Mock()
        callback2 = Mock()

        manager.add_callback(callback1)
        manager.add_callback(callback2)
        manager.initialize(10)

        manager.update(5)

        callback1.assert_called_once_with(5, 1, 10)
        callback2.assert_called_once_with(5, 1, 10)

    def test_remove_callback(self, manager):
        """Test removing a callback."""
        callback = Mock()
        manager.add_callback(callback)

        result = manager.remove_callback(callback)

        assert result is True
        assert len(manager.callbacks) == 0

    def test_remove_nonexistent_callback(self, manager):
        """Test removing a callback that doesn't exist."""
        callback = Mock()

        result = manager.remove_callback(callback)

        assert result is False

    def test_get_metrics(self, manager):
        """Test getting metrics."""
        manager.initialize(10)
        manager.update(3)
        manager.update(2)

        metrics = manager.get_metrics()

        assert isinstance(metrics, SequenceMetrics)
        assert metrics.sequence_length == 5
        assert metrics.initial_prompt_length == 10
        assert metrics.step_count == 2
        assert metrics.total_length == 15

    def test_reset(self, manager):
        """Test resetting manager state."""
        manager.initialize(10)
        manager.update(5)
        callback = Mock()
        manager.add_callback(callback)

        manager.reset()

        assert manager.metrics.sequence_length == 0
        assert manager.metrics.initial_prompt_length == 0
        assert manager.metrics.step_count == 0
        # Note: callbacks are NOT cleared on reset
        assert len(manager.callbacks) == 1

    def test_validate_state(self, manager):
        """Test state validation."""
        manager.initialize(10)
        manager.update(5)

        assert manager.validate_state() is True

    def test_metrics_average_tokens_per_step(self, manager):
        """Test average tokens per step calculation."""
        manager.initialize(10)
        manager.update(4)
        manager.update(2)
        manager.update(3)

        metrics = manager.get_metrics()
        # Total 9 tokens over 3 steps = 3.0 average
        assert metrics.average_tokens_per_step == 3.0

    def test_metrics_average_tokens_per_step_zero_steps(self, manager):
        """Test average tokens per step with zero steps."""
        manager.initialize(10)

        metrics = manager.get_metrics()
        assert metrics.average_tokens_per_step == 0.0
