import json
from pathlib import Path

import pytest
import torch

from scripts.download_model import verify_model_directory
from src.smoke_test import load_config, load_dataset, mask_padding_labels, validate_forward
from src.model_io import apply_model_transform


def test_verify_model_directory(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    result = verify_model_directory(tmp_path)
    assert result["weights_found"] and result["total_bytes"] > 0


def test_verify_model_directory_rejects_missing_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="weight"):
        verify_model_directory(tmp_path)


def test_verify_model_directory_rejects_missing_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors"}})
    )
    with pytest.raises(ValueError, match="shards"):
        verify_model_directory(tmp_path)


def test_configuration_and_dataset():
    config = load_config(Path(__file__).parents[1] / "configs/smoke_test.yaml")
    rows = load_dataset(Path(__file__).parents[1] / "data/smoke_train.jsonl")
    assert config["max_steps"] == 1
    assert len(rows) == 4


def test_label_masking():
    ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[1, 1, 0]])
    assert mask_padding_labels(ids, mask).tolist() == [[1, 2, -100]]


def test_forward_validation_and_finite_loss():
    class Output:
        logits = torch.zeros(1, 3, 5)
        loss = torch.tensor(1.0)

    assert validate_forward(Output(), 1, 3, 5) == 1.0
    Output.loss = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        validate_forward(Output(), 1, 3, 5)


def test_identity_transform_and_reject_unknown():
    marker = object()
    assert apply_model_transform(marker, {"model_transform": "none"}) is marker
    with pytest.raises(ValueError, match="Unsupported"):
        apply_model_transform(marker, {"model_transform": "surgery"})


@pytest.mark.integration
def test_real_model_smoke():
    pytest.skip("Run explicitly with a downloaded MobileLLM-R1-140M model")
