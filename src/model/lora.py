import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from omegaconf import DictConfig, open_dict, ListConfig
from typing import Optional
import torch
import logging
import os
from peft import PeftConfig, PeftModel
from transformers import BitsAndBytesConfig
from huggingface_hub import repo_exists

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)


class LoRAModelForCausalLM:
    """
    Wrapper class for loading models with LoRA adapters.
    Supports the specified LoRA configuration parameters.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        lora_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        assert lora_config is not None, ValueError(
            "LoRA config must be provided for LoRAModelForCausalLM."
        )
        logger.info(f"\x1b[32mLoading Model: {pretrained_model_name_or_path}\x1b[0m")
        quantization_config = kwargs.pop("quantization_config", None)

        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            quantization_config=quantization_config,
            **kwargs,
        )

        if quantization_config is not None:
            base_model = prepare_model_for_kbit_training(base_model)

        peft_config = LoraConfig(
            target_modules=list(lora_config["target_modules"]),
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            r=lora_config["r"],
            bias=lora_config["bias"],
            task_type=lora_config["task_type"],
            use_rslora=lora_config.get("use_rslora", False),
        )
        logger.info(f"\x1b[32mApplying LoRA with config: {peft_config}\x1b[0m")
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()

        return model
