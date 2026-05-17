"""Baseline decoder name lists shared by qecbench callers and plotting code."""

BASELINE_DECODERS_GRAPHLIKE = [
    "BP",
    "MemBP",
    "RelayBP",
    "EnsSerialBP",
    "MWPM",
    "UnionFind",
]
DEFAULT_BASELINE_DECODERS_GRAPHLIKE = [
    "BP",
    "MemBP",
    "RelayBP",
]
BASELINE_DECODERS_HYPERGRAPH = [
    "BP",
    "MemBP",
    "RelayBP",
    "EnsSerialBP",
    "BPOSD",
]
DEFAULT_BASELINE_DECODERS_HYPERGRAPH = [
    "BP",
    "MemBP",
    "RelayBP",
]
ALL_BASELINE_DECODERS = set(BASELINE_DECODERS_GRAPHLIKE + BASELINE_DECODERS_HYPERGRAPH)
