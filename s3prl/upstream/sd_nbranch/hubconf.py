import logging
import tarfile

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def sd_nbranch_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
        my_tar = tarfile.open(ckpt)
        target_dir = ckpt[:-4]
        my_tar.extractall(target_dir)
        my_tar.close()

    return _UpstreamExpert(ckpt, **kwargs)


def sd_nbranch_local(*args, **kwargs):
    return sd_nbranch_custom(*args, **kwargs)


def sd_nbranch_url(*args, **kwargs):
    return sd_nbranch_custom(*args, **kwargs)


# def vghubert(refresh=False, *args, **kwargs):
#     """
#     The default model - Base
#         refresh (bool): whether to download ckpt/config again if existed
#     """
#     return wav2vec2_base_960(refresh=refresh, *args, **kwargs)
