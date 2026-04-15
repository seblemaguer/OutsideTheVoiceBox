import json
import os
import re
from typing import Dict

import fsspec
import yaml
from coqpit import Coqpit

from TTS.config.shared_configs import *
from TTS.utils.generic_utils import find_module

from TTS.utils.manage import ModelManager
from TTS.config import register_config as coqui_register_config
from TTS.config import read_json_with_comments, _process_model_name
from TTS.tts.models import setup_model as coqui_setup_model


from .modified_xtts import ModifiedXtts


def download_model_by_name(model_name: str):
    manager = ModelManager(progress_bar=True)
    model_path, config_path, model_item = manager.download_model(model_name)
    if "fairseq" in model_name or (model_item is not None and isinstance(model_item["model_url"], list)):
        # return model directory if there are multiple files
        # we assume that the model knows how to load itself
        return None, None, None, None, model_path
    if model_item.get("default_vocoder") is None:
        return model_path, config_path, None, None, None
    vocoder_path, vocoder_config_path, _ = manager.download_model(model_item["default_vocoder"])
    return model_path, config_path, vocoder_path, vocoder_config_path, None


############################################################################################
## Hijack coqui config/model setup to include our modified version
############################################################################################

def setup_tts_model(config: Coqpit, samples: list[list]|list[dict]|None = None) -> "BaseTTS":
    if ("model" in config) and (config["model"] == "modified_xtts"):
        model = ModifiedXtts.init_from_config(config=config, samples=samples)
    else:
        model = coqui_setup_model(config, samples)
    return model


def register_config(model_name: str) -> Coqpit:
    """Find the right config for the given model name.

    Args:
        model_name (str): Model name.

    Raises:
        ModuleNotFoundError: No matching config for the model name.

    Returns:
        Coqpit: config class.
    """
    if model_name == "modified_xtts":
        model_name = "xtts"

    return coqui_register_config(model_name)

def load_config(config_path: str) -> Coqpit:
    """Import `json` or `yaml` files as TTS configs. First, load the input file as a `dict` and check the model name
    to find the corresponding Config class. Then initialize the Config.

    Args:
        config_path (str): path to the config file.

    Raises:
        TypeError: given config file has an unknown type.

    Returns:
        Coqpit: TTS config object.
    """
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            # backwards compat.
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config


__all__ = ["ModifiedXtts", "setup_tts_model", "download_model_by_name", "register_config"]
