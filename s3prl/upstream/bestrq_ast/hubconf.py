import logging
import tarfile

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)


def bestrq_ast_custom(
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


def bestrq_ast_local(*args, **kwargs):
    return bestrq_ast_custom(*args, **kwargs)


def bestrq_ast_url(*args, **kwargs):
    return bestrq_ast_custom(*args, **kwargs)
