import logging
import tarfile

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def cotraining_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):

    return _UpstreamExpert(ckpt, **kwargs)


def cotraining_local(*args, **kwargs):
    return cotraining_custom(*args, **kwargs)


def cotraining_url(*args, **kwargs):
    return cotraining_custom(*args, **kwargs)


# def vghubert(refresh=False, *args, **kwargs):
#     """
#     The default model - Base
#         refresh (bool): whether to download ckpt/config again if existed
#     """
#     return wav2vec2_base_960(refresh=refresh, *args, **kwargs)
