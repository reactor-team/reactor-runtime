"""Move objects and their arrays between processes through shared memory.

An object is pickled with its flat buffers (contiguous numpy arrays) written
into a shared block instead of into the pickle; the small header that names the
block travels on whatever carries messages, a queue or a socket::

    slot = SharedSlot(directory=shm_dir)           # the writer's growing block
    header = pack(result, slot)                    # bytes for the queue or socket

    reader = SlotReader(directory=shm_dir)         # the other process
    result = unpack(header, reader)                # its arrays are copies it owns

A block is a file in a memory-backed directory (``/dev/shm`` by default) that
both processes see. The runtime's distributed runner and the model server both
build on this package; it reads no manifest and no environment variable.
"""

from reactor_runtime.ipc.blocks import (
    SHM_DIR,
    Block,
    SharedSlotAllocationFailed,
    attach_block,
    create_block,
    default_directory,
    unlink_blocks,
)
from reactor_runtime.ipc.payload import pack, unpack
from reactor_runtime.ipc.slot import SharedSlot, SlotReader, new_prefix

__all__ = [
    "SHM_DIR",
    "Block",
    "SharedSlot",
    "SharedSlotAllocationFailed",
    "SlotReader",
    "attach_block",
    "create_block",
    "default_directory",
    "new_prefix",
    "pack",
    "unlink_blocks",
    "unpack",
]
