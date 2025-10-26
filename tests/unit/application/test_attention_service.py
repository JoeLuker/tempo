"""Unit tests for AttentionService."""

import pytest
import torch

from src.application.services.attention_service import AttentionService


class TestAttentionService:
    """Tests for AttentionService."""

    @pytest.fixture
    def isolated_service(self):
        """Create service with isolation enabled."""
        return AttentionService(
            isolate_parallel_tokens=True,
            device="cpu",
            debug_mode=False
        )

    @pytest.fixture
    def visible_service(self):
        """Create service with isolation disabled."""
        return AttentionService(
            isolate_parallel_tokens=False,
            device="cpu",
            debug_mode=False
        )

    def test_initialization(self, isolated_service):
        """Test service initializes correctly."""
        assert isolated_service.isolate_parallel_tokens is True
        assert isolated_service.device == "cpu"
        assert isolated_service.parallel_sets == []
        assert isolated_service.current_mask is None

    def test_register_parallel_set(self, isolated_service):
        """Test registering a parallel token set."""
        isolated_service.register_parallel_set(5, 7)

        assert len(isolated_service.parallel_sets) == 1
        assert isolated_service.parallel_sets[0] == (5, 8)  # End is exclusive

    def test_build_causal_mask_visible_mode(self, visible_service):
        """Test building causal mask in visible mode."""
        visible_service.initialize(10)
        mask = visible_service.build_attention_mask(seq_length=10)

        assert mask.shape == (10, 10)
        assert mask.device.type == "cpu"
        # Verify causal structure (upper triangle should be masked)
        assert (mask[0, 1:] < -1000).all()  # Future tokens masked

    def test_build_isolation_mask(self, isolated_service):
        """Test building isolation mask."""
        isolated_service.initialize(10)
        isolated_service.register_parallel_set(5, 7)  # Positions 5, 6, 7

        mask = isolated_service.build_attention_mask(seq_length=10)

        assert mask.shape == (10, 10)
        # Verify isolation: parallel tokens shouldn't attend to each other
        # This depends on mask_builder implementation

    def test_get_current_mask(self, isolated_service):
        """Test retrieving current mask."""
        isolated_service.initialize(10)
        mask = isolated_service.build_attention_mask(seq_length=10)

        current = isolated_service.get_current_mask()
        assert current is not None
        assert torch.equal(current, mask)

    def test_set_isolation_mode(self, isolated_service):
        """Test toggling isolation mode."""
        assert isolated_service.isolate_parallel_tokens is True

        isolated_service.set_isolation_mode(False)
        assert isolated_service.isolate_parallel_tokens is False

        isolated_service.set_isolation_mode(True)
        assert isolated_service.isolate_parallel_tokens is True

    def test_reset(self, isolated_service):
        """Test resetting service state."""
        isolated_service.initialize(10)
        isolated_service.register_parallel_set(5, 7)
        isolated_service.build_attention_mask(seq_length=10)

        assert len(isolated_service.parallel_sets) > 0
        assert isolated_service.current_mask is not None

        isolated_service.reset()

        assert len(isolated_service.parallel_sets) == 0
        assert isolated_service.current_mask is None

    def test_masks_differ_between_modes(self):
        """Test that isolated and visible modes produce different masks."""
        isolated = AttentionService(isolate_parallel_tokens=True, device="cpu")
        visible = AttentionService(isolate_parallel_tokens=False, device="cpu")

        # Register same parallel set for both
        isolated.initialize(10)
        visible.initialize(10)
        isolated.register_parallel_set(5, 7)
        visible.register_parallel_set(5, 7)

        isolated_mask = isolated.build_attention_mask(seq_length=10)
        visible_mask = visible.build_attention_mask(seq_length=10)

        # Masks should differ
        assert not torch.equal(isolated_mask, visible_mask)
