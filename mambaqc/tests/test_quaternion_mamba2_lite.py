import torch
import pytest

from mambaqc.models import QuaternionMamba2Lite, quaternion_mamba2_lite_small


def test_lite_forward_shapes():
    model = QuaternionMamba2Lite(vocab_size=2048, d_model=32, n_layers=2, d_state=8)
    input_ids = torch.randint(0, 2048, (2, 5))
    labels = torch.randint(0, 2048, (2, 5))

    outputs = model(input_ids, labels=labels)

    assert outputs["logits"].shape == (2, 5, 2048)
    assert outputs["loss"] is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Preset configs are large; skip on CPU runtimes"
)
def test_lite_presets():
    model = quaternion_mamba2_lite_small(vocab_size=2048)
    input_ids = torch.randint(0, 2048, (1, 3))
    outputs = model(input_ids)

    assert outputs["logits"].shape[:2] == (1, 3)
