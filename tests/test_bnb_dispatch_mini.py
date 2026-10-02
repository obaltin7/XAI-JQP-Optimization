"""
GPU-free test that dispatch inside from_pretrained never calls model.to on a bnb (4/8-bit) model
(run_baselines.bnb_safe_dispatch).

transformers 4.44.2 + accelerate>=1.x on a single GPU: dispatch_model calls model.to(device) for a single-device
device_map, and 4.44.2 rejects that for bnb models. The fake dispatch below mimics this behaviour (no accelerate needed).

Run:
    pytest tests/test_bnb_dispatch_mini.py -q
"""
import enum
import os
import sys
import types

import pytest
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_baselines import _bnb_safe, bnb_safe_dispatch, is_bnb_model  # noqa: E402

BNB_TO_ERROR = "`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models."


class _QuantMethod(str, enum.Enum):  # mimics transformers.utils.quantization_config.QuantizationMethod
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"


class FakeModel(nn.Module):
    """Mimics PreTrainedModel.to: raises ValueError like 4.44.2 if bnb flags are set; otherwise counts the call."""

    def __init__(self, **flags):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.to_calls = 0
        for k, v in flags.items():
            setattr(self, k, v)

    def to(self, *args, **kwargs):
        self.to_calls += 1
        if is_bnb_model(self):
            raise ValueError(BNB_TO_ERROR)
        return self


def fake_dispatch(model, device_map, force_hooks=False, **kwargs):
    """Mimics accelerate>=1.x dispatch_model: single device without force_hooks -> model.to(device)."""
    if len(set(device_map.values())) > 1 or force_hooks:
        model.hooked = True
    else:
        model.to(list(device_map.values())[0])
    return model


DISPATCH_TIME_FLAGS = {"is_quantized": True, "quantization_method": _QuantMethod.BITS_AND_BYTES}


def test_is_bnb_model_flags():
    assert is_bnb_model(FakeModel(is_loaded_in_4bit=True))
    assert is_bnb_model(FakeModel(is_loaded_in_8bit=True))
    assert is_bnb_model(FakeModel(**DISPATCH_TIME_FLAGS))  # at dispatch time is_loaded_in_4bit is not set yet
    assert is_bnb_model(FakeModel(is_quantized=True, quantization_method="bitsandbytes"))
    assert not is_bnb_model(FakeModel())
    assert not is_bnb_model(FakeModel(is_quantized=True, quantization_method=_QuantMethod.GPTQ))


def test_unguarded_dispatch_reproduces_runpod_error():
    model = FakeModel(**DISPATCH_TIME_FLAGS)
    with pytest.raises(ValueError, match="is not supported for `4-bit`"):
        fake_dispatch(model, {"": 0})


@pytest.mark.parametrize("flags", [DISPATCH_TIME_FLAGS, {"is_loaded_in_4bit": True}])
def test_guarded_dispatch_never_calls_to_on_bnb_model(flags):
    model = FakeModel(**flags)
    out = _bnb_safe(fake_dispatch)(model, {"": 0})
    assert out is model
    assert model.to_calls == 0
    assert model.hooked is True


def test_guarded_dispatch_leaves_non_bnb_model_unchanged():
    model = FakeModel()
    _bnb_safe(fake_dispatch)(model, {"": 0})
    assert model.to_calls == 1  # fp16/GPTQ/AWQ path: behaviour unchanged
    assert not hasattr(model, "hooked")


def test_context_manager_patches_and_restores():
    mu = types.SimpleNamespace(dispatch_model=fake_dispatch)
    model = FakeModel(**DISPATCH_TIME_FLAGS)
    with bnb_safe_dispatch(mu):
        assert mu.dispatch_model is not fake_dispatch
        mu.dispatch_model(model, device_map={"": 0})
    assert mu.dispatch_model is fake_dispatch
    assert model.to_calls == 0


def test_context_manager_restores_on_error_and_tolerates_missing_dispatch():
    mu = types.SimpleNamespace(dispatch_model=fake_dispatch)
    with pytest.raises(RuntimeError):
        with bnb_safe_dispatch(mu):
            raise RuntimeError("load error")
    assert mu.dispatch_model is fake_dispatch
    with bnb_safe_dispatch(types.SimpleNamespace()):  # transformers version without dispatch_model
        pass
