"""Factory for building decoders."""

from .base import Decoder, IterativeDecoder
from .bp import BPDecoder
from .dmembp import DMemBPDecoder
from .dmemoffsetbp import DMemOffsetBPDecoder
from .mwpm import MWPMDecoder
from .uf import UnionFindDecoder


DECODER_NAME_TO_CLASS: dict[str, type[Decoder]] = {
    "BP": BPDecoder,
    "DMemBP": DMemBPDecoder,
    "DMemOffsetBP": DMemOffsetBPDecoder,
    "MWPM": MWPMDecoder,
    "UnionFind": UnionFindDecoder,
}
ALL_DECODERS = list(DECODER_NAME_TO_CLASS.keys())
ITERATIVE_DECODERS = [
    name
    for name, cls in DECODER_NAME_TO_CLASS.items()
    if issubclass(cls, IterativeDecoder)
]


def create_decoder(name: str, **kwargs) -> Decoder:
    """Create a decoder by name, with kwargs passed to the constructor.

    Check out `qecdec.decoders.ALL_DECODERS` for the list of available decoder names.
    """
    if name not in ALL_DECODERS:
        raise ValueError(f"Invalid decoder name: {name!r}")
    cls = DECODER_NAME_TO_CLASS[name]
    return cls(**kwargs)
