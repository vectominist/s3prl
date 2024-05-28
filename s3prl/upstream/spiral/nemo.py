from abc import ABC
import os
from pathlib import Path
from time import sleep
import logging
import copy

from omegaconf import DictConfig, OmegaConf
import wget
import hydra

try:
    from omegaconf import DictConfig, OmegaConf
    from omegaconf import errors as omegaconf_errors
except ModuleNotFoundError:
    _HAS_HYDRA = False


def _convert_config(cfg: "OmegaConf"):
    """Recursive function convertint the configuration from old hydra format to the new one."""
    if not _HAS_HYDRA:
        logging.error(
            "This function requires Hydra/Omegaconf and it was not installed."
        )
        exit(1)

    # Get rid of cls -> _target_.
    if "cls" in cfg and "_target_" not in cfg:
        cfg._target_ = cfg.pop("cls")

    # Get rid of params.
    if "params" in cfg:
        params = cfg.pop("params")
        for param_key, param_val in params.items():
            cfg[param_key] = param_val

    # Recursion.
    try:
        for _, sub_cfg in cfg.items():
            if isinstance(sub_cfg, DictConfig):
                _convert_config(sub_cfg)
    except omegaconf_errors.OmegaConfBaseException as e:
        logging.warning(
            f"Skipped conversion for config/subconfig:\n{cfg}\n Reason: {e}."
        )


def maybe_download_from_cloud(
    url, filename, subfolder=None, cache_dir=None, refresh_cache=False
) -> str:
    """
    Helper function to download pre-trained weights from the cloud
    Args:
        url: (str) URL of storage
        filename: (str) what to download. The request will be issued to url/filename
        subfolder: (str) subfolder within cache_dir. The file will be stored in cache_dir/subfolder. Subfolder can
            be empty
        cache_dir: (str) a cache directory where to download. If not present, this function will attempt to create it.
            If None (default), then it will be $HOME/.cache/torch/NeMo
        refresh_cache: (bool) if True and cached file is present, it will delete it and re-fetch

    Returns:
        If successful - absolute local path to the downloaded file
        else - empty string
    """
    # try:
    if cache_dir is None:
        cache_location = Path.joinpath(Path.home(), ".cache/torch/NeMo")
    else:
        cache_location = cache_dir
    if subfolder is not None:
        destination = Path.joinpath(cache_location, subfolder)
    else:
        destination = cache_location

    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    destination_file = Path.joinpath(destination, filename)

    if os.path.exists(destination_file):
        logging.info(f"Found existing object {destination_file}.")
        if refresh_cache:
            logging.info("Asked to refresh the cache.")
            logging.info(f"Deleting file: {destination_file}")
            os.remove(destination_file)
        else:
            logging.info(f"Re-using file from: {destination_file}")
            return str(destination_file)
    # download file
    wget_uri = url + filename
    logging.info(f"Downloading from: {wget_uri} to {str(destination_file)}")
    # NGC links do not work everytime so we try and wait
    i = 0
    max_attempts = 3
    while i < max_attempts:
        i += 1
        try:
            wget.download(wget_uri, str(destination_file))
            if os.path.exists(destination_file):
                return destination_file
            else:
                return ""
        except:
            logging.info(f"Download from cloud failed. Attempt {i} of {max_attempts}")
            sleep(0.05)
            continue
    raise ValueError("Not able to download url right now, please try again.")


def maybe_update_config_version(cfg: "DictConfig"):
    """
    Recursively convert Hydra 0.x configs to Hydra 1.x configs.

    Changes include:
    -   `cls` -> `_target_`.
    -   `params` -> drop params and shift all arguments to parent.
    -   `target` -> `_target_` cannot be performed due to ModelPT injecting `target` inside class.

    Args:
        cfg: Any Hydra compatible DictConfig

    Returns:
        An updated DictConfig that conforms to Hydra 1.x format.
    """
    if not _HAS_HYDRA:
        logging.error(
            "This function requires Hydra/Omegaconf and it was not installed."
        )
        exit(1)
    if cfg is not None and not isinstance(cfg, DictConfig):
        try:
            temp_cfg = OmegaConf.create(cfg)
            cfg = temp_cfg
        except omegaconf_errors.OmegaConfBaseException:
            # Cannot be cast to DictConfig, skip updating.
            return cfg

    # Make a copy of model config.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)

    # Convert config.
    _convert_config(cfg)

    # Update model config.
    OmegaConf.set_struct(cfg, True)

    return cfg


class Serialization(ABC):
    @classmethod
    def from_config_dict(cls, config: DictConfig):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)

        config = maybe_update_config_version(config)

        if ("cls" in config or "target" in config) and "params" in config:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        elif "_target_" in config:
            # regular hydra-based instantiation
            instance = hydra.utils.instantiate(config=config)
        else:
            # models are handled differently for now
            instance = cls(cfg=config)

        if not hasattr(instance, "_cfg"):
            instance._cfg = config
        return instance

    def to_config_dict(self) -> DictConfig:
        """Returns object's configuration to config dictionary"""
        if (
            hasattr(self, "_cfg")
            and self._cfg is not None
            and isinstance(self._cfg, DictConfig)
        ):
            # Resolve the config dict
            config = OmegaConf.to_container(self._cfg, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)

            config = maybe_update_config_version(config)

            self._cfg = config

            return self._cfg
        else:
            raise NotImplementedError(
                "to_config_dict() can currently only return object._cfg but current object does not have it."
            )
