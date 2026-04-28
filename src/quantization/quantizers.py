from typing import Any, Dict
import torch
from transformers import BitsAndBytesConfig, GPTQConfig


class QuantizationConfig:

    _DTYPE_ALIASES = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }

    @classmethod
    def _resolve_dtypes(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in config.items():
            if isinstance(value, str):
                normalized = value.lower().split(".")[-1]
                config[key] = cls._DTYPE_ALIASES.get(normalized, value)
            else:
                config[key] = value
        return config
    
    @classmethod
    def from_config(cls, **kwargs):
        if kwargs is None:
            raise ValueError("No configuration parameters provided for quantization.")
        if cls._TARGET_CLASS is None:
            raise NotImplementedError(
                f"QuantizationConfig subclass {cls.__name__} must define a _TARGET_CLASS."
            )
        resolved_kwargs = cls._resolve_dtypes(kwargs)

        # Return the official object (E.g., BitsAndBytesConfig) instanciated
        return cls._TARGET_CLASS(**resolved_kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

class BitsAndBytes(QuantizationConfig):
    _TARGET_CLASS = BitsAndBytesConfig


class GPTQ(QuantizationConfig):
    _TARGET_CLASS = GPTQConfig
