from pathlib import Path
import warnings
from omegaconf import OmegaConf, DictConfig, ListConfig

warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated"
)

warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`"
)


def convert_paths(cfg: DictConfig) -> None:
    """
    Recursively convert all values of keys ending with '_file' or '_dir' into pathlib.Path
    This modifies the DictConfig in-place.
    """
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            convert_paths(value)  # Recurse
        elif isinstance(key, str) and key.endswith(("_file", "_dir", "_path")):
            cfg[key] = Path(value)
        elif isinstance(key, str) and key.endswith(("_files", "_dirs", "_pathes")):
            cfg[key] = [Path(elt) for elt in value]

def str_config(cfg: DictConfig):
    """
    Pretty print the full Hydra/OmegaConf config to logger.debug.
    Resolves all interpolations.
    """
    # Convert to a standard Python dict with all interpolations resolved
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Convert to YAML for pretty printing
    cfg_yaml = OmegaConf.to_yaml(cfg_dict)

    return f"Expanded configuration:\n{cfg_yaml}\n"

__all__ = ["convert_paths", "str_config"]
