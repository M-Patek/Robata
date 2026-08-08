from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from robata.inference.mage_small_encoder import (
    MageCompatibleSmallEncoder,
    MageSmallEncoderError,
    MageSmallEncoderPolicy,
    select_mage_visual_token_runs,
    uniform_temporal_run_indices,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    image_id = 99
    input_ids = torch.tensor(
        [[10, 20, image_id, image_id, 30, 40, image_id, image_id, image_id, 50]]
    )
    attention_mask = torch.ones_like(input_ids)
    features = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    return input_ids, attention_mask, features, image_id


def test_uniform_temporal_indices_keep_boundaries_and_are_stable() -> None:
    assert uniform_temporal_run_indices(run_count=8, max_temporal_runs=4) == (0, 2, 4, 7)
    assert uniform_temporal_run_indices(run_count=3, max_temporal_runs=4) == (0, 1, 2)


def test_selection_keeps_complete_runs_without_mixing_features() -> None:
    input_ids, attention_mask, features, image_id = _inputs()
    output_ids, output_mask, output_features, kept, run_count = select_mage_visual_token_runs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_features=features,
        image_token_id=image_id,
        max_temporal_runs=2,
    )
    assert kept == (0, 1)
    assert run_count == 2
    assert output_ids.tolist() == [
        [10, 20, image_id, image_id, 30, 40, image_id, image_id, image_id, 50]
    ]
    assert output_mask.tolist() == [[1] * 10]
    torch.testing.assert_close(output_features, features)


def test_selection_drops_runs_and_preserves_nonvisual_tokens() -> None:
    image_id = 99
    vision_start_id = 101
    vision_end_id = 102
    input_ids = torch.tensor(
        [
            [
                10,
                vision_start_id,
                image_id,
                image_id,
                vision_end_id,
                20,
                vision_start_id,
                image_id,
                image_id,
                vision_end_id,
                30,
                vision_start_id,
                image_id,
                image_id,
                vision_end_id,
                40,
            ]
        ]
    )
    attention_mask = torch.ones_like(input_ids)
    features = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    output_ids, _, output_features, kept, run_count = select_mage_visual_token_runs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_features=features,
        image_token_id=image_id,
        max_temporal_runs=2,
        vision_start_token_id=vision_start_id,
        vision_end_token_id=vision_end_id,
    )
    assert kept == (0, 2)
    assert run_count == 3
    assert output_ids.tolist() == [
        [
            10,
            vision_start_id,
            image_id,
            image_id,
            vision_end_id,
            20,
            30,
            vision_start_id,
            image_id,
            image_id,
            vision_end_id,
            40,
        ]
    ]
    torch.testing.assert_close(output_features, torch.cat((features[:2], features[4:]), dim=0))


def test_selection_rejects_feature_placeholder_mismatch() -> None:
    input_ids, attention_mask, features, image_id = _inputs()
    with pytest.raises(MageSmallEncoderError, match="does not match"):
        select_mage_visual_token_runs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=features[:2],
            image_token_id=image_id,
            max_temporal_runs=2,
        )


def test_policy_is_explicit_and_shadow_only() -> None:
    policy = MageSmallEncoderPolicy(visual_layer_count=16, max_temporal_runs=4)
    assert policy.shadow_only is True
    assert policy.selection_mode == "UNIFORM_TEMPORAL_RUN_KEEP_NO_EMPTY_SPANS_V2"
    assert len(policy.semantic_sha256) == 64
    with pytest.raises(ValueError):
        MageSmallEncoderPolicy(visual_layer_count=0)


class _FakeVisual:
    def __init__(self, *, hidden_size: int = 4) -> None:
        self.encoder = SimpleNamespace(
            layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        )
        self.merger = object()
        self.embeddings = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32))
        )
        self.hidden_size = hidden_size
        self.seen_layer_counts: list[int] = []

    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        patch_positions: torch.Tensor,
    ) -> SimpleNamespace:
        del pixel_values, grid_thw, patch_positions
        self.seen_layer_counts.append(len(self.encoder.layers))
        values = torch.arange(4 * self.hidden_size, dtype=torch.float32).reshape(
            4, self.hidden_size
        )
        return SimpleNamespace(last_hidden_state=values)


class _FakeMageModel:
    def __init__(self, *, visual_hidden_size: int = 4, decoder_hidden_size: int = 4) -> None:
        self.visual = _FakeVisual(hidden_size=visual_hidden_size)
        self.model = SimpleNamespace(visual=self.visual)
        self.config = SimpleNamespace(image_token_id=99)
        self._embedding = torch.nn.Embedding(200, decoder_hidden_size)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self._embedding


def _prepared_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[10, 99, 99, 20, 99, 99, 30]]),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "pixel_values": torch.zeros((4, 3, 16, 16), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "patch_positions": torch.zeros((4, 3), dtype=torch.long),
    }


def test_prepare_uses_requested_layers_and_restores_resident_model() -> None:
    model = _FakeMageModel()
    original_layers = model.visual.encoder.layers
    encoder = MageCompatibleSmallEncoder(
        model=model,
        policy=MageSmallEncoderPolicy(visual_layer_count=2, max_temporal_runs=2),
    )
    result = encoder.prepare(_prepared_inputs())
    assert model.visual.seen_layer_counts == [2]
    assert model.visual.encoder.layers is original_layers
    assert result.shadow_only is True
    assert result.inputs_embeds.shape == (1, 7, 4)
    assert result.telemetry.visual_layer_count == 2
    assert result.telemetry.feature_content_sha256


def test_prepare_rejects_features_outside_decoder_embedding_space() -> None:
    model = _FakeMageModel(visual_hidden_size=3, decoder_hidden_size=4)
    encoder = MageCompatibleSmallEncoder(
        model=model,
        policy=MageSmallEncoderPolicy(visual_layer_count=4, max_temporal_runs=2),
    )
    with pytest.raises(MageSmallEncoderError, match="decoder hidden size"):
        encoder.prepare(_prepared_inputs())


def test_selection_rejects_dropped_run_without_vision_wrappers() -> None:
    input_ids = torch.tensor([[10, 99, 99, 20, 99, 99, 30, 99, 99, 40]])
    attention_mask = torch.ones_like(input_ids)
    features = torch.zeros((6, 2), dtype=torch.float32)
    with pytest.raises(MageSmallEncoderError, match="vision-start"):
        select_mage_visual_token_runs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=features,
            image_token_id=99,
            max_temporal_runs=2,
            vision_start_token_id=101,
            vision_end_token_id=102,
        )
