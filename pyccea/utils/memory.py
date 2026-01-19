import gc
import os
import ctypes


def force_memory_release() -> None:
    """Force memory release back to the OS when supported."""
    gc.collect()
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        return
