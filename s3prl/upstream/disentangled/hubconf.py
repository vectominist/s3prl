import logging


from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)



def disentangled_custom(
    ckpt: str,
    refresh: bool = False,
    **kwargs,
):
    return _UpstreamExpert(ckpt, **kwargs)


def disentangled_local(*args, **kwargs):
    return disentangled_custom(*args, **kwargs)


def disentangled_url(*args, **kwargs):
    return disentangled_custom(*args, **kwargs)
