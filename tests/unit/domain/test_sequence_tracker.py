"""Unit tests for SequenceTracker domain service."""

import pytest
from src.domain.services.sequence_tracker import SequenceTracker


class TestSequenceTracker:
    """Tests for SequenceTracker."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = SequenceTracker(debug_mode=False)
        tracker.initialize(10)

        assert tracker.initial_prompt_length == 10
        assert tracker.sequence_length == 0
        assert tracker.step_count == 0

    def test_update_sequence_length(self):
        """Test updating sequence length."""
        tracker = SequenceTracker(debug_mode=False)
        tracker.initialize(5)

        tracker.update_sequence_length(1)
        assert tracker.sequence_length == 1
        assert tracker.step_count == 1

        tracker.update_sequence_length(3)
        assert tracker.sequence_length == 3
        assert tracker.step_count == 2

    def test_sequence_length_history(self):
        """Test sequence length history tracking."""
        tracker = SequenceTracker(debug_mode=False)
        tracker.initialize(5)

        tracker.update_sequence_length(1)
        tracker.update_sequence_length(2)
        tracker.update_sequence_length(5)

        assert tracker.sequence_length_history == [1, 2, 5]

    def test_callback_invocation(self):
        """Test callback is invoked on sequence update."""
        tracker = SequenceTracker(debug_mode=False)
        tracker.initialize(10)

        callback_calls = []

        def test_callback(seq_len, step, prompt_len):
            callback_calls.append((seq_len, step, prompt_len))

        tracker.update_sequence_length(5, callback=test_callback)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (5, 1, 10)

    def test_get_metrics(self):
        """Test getting sequence metrics."""
        tracker = SequenceTracker(debug_mode=False)
        tracker.initialize(10)

        tracker.update_sequence_length(5)
        tracker.update_sequence_length(10)

        metrics = tracker.get_metrics()

        assert metrics['sequence_length'] == 10
        assert metrics['initial_prompt_length'] == 10
        assert metrics['step_count'] == 2
        assert metrics['total_length'] == 20
        assert len(metrics['sequence_length_history']) == 2

    def test_no_update_on_shorter_length(self):
        """Test that shorter lengths don't update."""
        tracker = SequenceTracker(debug_mode=False)
        tracker.initialize(5)

        tracker.update_sequence_length(10)
        assert tracker.sequence_length == 10

        tracker.update_sequence_length(5)  # Should not update
        assert tracker.sequence_length == 10  # Still 10
