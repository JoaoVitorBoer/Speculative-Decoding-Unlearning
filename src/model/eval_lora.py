import logging
import tempfile
from typing import Optional

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class LoRAModelForEvaluation:
    """
    Load a base model together with saved LoRA adapters, merge them and return
    a standalone model for inference.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a model with saved LoRA adapters, merge the adapters into the base
        model and unload PEFT wrappers.

        Args:
            pretrained_model_name_or_path: Path to the saved LoRA adapters.
            base_model_name_or_path: Optional explicit base model path. When
                omitted, it is read from the adapter config.
            **kwargs: Additional arguments forwarded to the base model loader.
        """
        quantization_config = kwargs.pop("quantization_config", None)
        reload_quantized = quantization_config is not None
        adapter_path = pretrained_model_name_or_path
        # base_load_kwargs = dict(kwargs)

        logger.info(f"Loading LoRA adapter config from {adapter_path}")
        peft_config = PeftConfig.from_pretrained(adapter_path)

        base_model_path = base_model_name_or_path or peft_config.base_model_name_or_path
        if base_model_path is None:
            raise ValueError(
                "Base model path not provided and not found in adapter config. "
                "Please supply `base_model_name_or_path` or ensure the adapter "
                "config contains `base_model_name_or_path`."
            )

        logger.info(f"Loading base model from {base_model_path}")
        assert "quantization_config" not in kwargs, (
            "quantization_config should not be passed when loading the base model. "
            "It will be applied after merging LoRA adapters."
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            # merge requires the base weights, so load without quantization
            **kwargs,
        )

        logger.info(f"Loading LoRA adapters from {adapter_path}")
        lora_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )

        assert_lora_loaded(lora_model)

        logger.info("Merging LoRA adapters into the base model")
        merged_model = lora_model.merge_and_unload()

        assert_lora_merged(merged_model)

        if reload_quantized:
            logger.info("Re-loading merged model with quantization applied")
            with tempfile.TemporaryDirectory() as tmpdir:
                merged_model.save_pretrained(tmpdir)
                merged_model = AutoModelForCausalLM.from_pretrained(
                    tmpdir,
                    quantization_config=quantization_config,
                    **kwargs,
                )
                
                assert_quantized(merged_model, quantization_config)
                logger.info(
                    f"Model re-loaded with quantization {merged_model.config.quantization_config}"
                )

        return merged_model


def assert_lora_loaded(model: PeftModel) -> None:
    assert isinstance(model, PeftModel), "LoRA model is not a PeftModel"
    assert model.peft_config, "No PEFT config found on model (no adapters loaded?)"
    assert any(
        "lora_" in n for n, _ in model.named_parameters()
    ), "No LoRA parameters (lora_*) found in model"


def assert_lora_merged(model) -> None:
    assert not isinstance(model, PeftModel), "Merged model is still a PeftModel"
    assert not any(
        "lora_" in n for n, _ in model.named_parameters()
    ), "Found LoRA parameters after merge_and_unload"

def assert_quantized(model, quantization_config) -> None:
    # HF sets one of these for bitsandbytes quantized loads
    is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    is_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    assert is_4bit or is_8bit, (
        "Model does not appear to be loaded in 4-bit or 8-bit. "
        "Check that bitsandbytes is installed and your quantization_config is valid."
    )

    # Optional: verify the config object ended up on the model config
    qc = getattr(model.config, "quantization_config", None)
    assert qc is not None, "model.config.quantization_config is missing after reload"