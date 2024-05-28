import logging
import tarfile

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def spiral_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, **kwargs)


def spiral_local(*args, **kwargs):
    return spiral_custom(*args, **kwargs)


def spiral_url(*args, **kwargs):
    return spiral_custom(*args, **kwargs)


def spiral_base(refresh=False, **kwargs):
    kwargs[
        "ckpt"
    ] = "https://huggingface.co/huawei-noah/SPIRAL-base/resolve/main/st2vec-last.ckpt"
    return spiral_custom(refresh=refresh, **kwargs)
