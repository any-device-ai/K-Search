"""FlashRT tunable-task support for k-search.

Keep this module lightweight: it must import on a host with no GPU, no CUDA
toolkit and no FlashRT install. Everything device-specific lives behind the
referee CLI, which is reached through `transport.py`.
"""

__all__ = ["prompts", "transport"]
