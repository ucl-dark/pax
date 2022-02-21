import os
from pathlib import Path

from flax import linen as nn
from flax import serialization


def save_model(filename: str, model: nn.Module) -> None:
    path = Path(filename)
    with path.open("wb") as fp:
        fp.write(serialization.to_bytes(model))


def load_model(filename: str, model: nn.Module) -> nn.Module:
    with open(filename, "rb") as fp:
        return serialization.from_bytes(model, fp.read())
