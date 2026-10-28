"""Derived key lookup for records the live writer has archived.

Not an authority: the archive/checkpoint remain the only evidence, and the
index is rebuilt from archived records on demand. It lives in unlinked
temporary files so lookups need neither a scan of the archive nor Python
memory that grows with run history. Open addressing over full-key-verified
slots keeps lookups exact.
"""
import hashlib
import os
import struct
import tempfile
import threading

SLOT = struct.Struct('<16sQq')  # key digest, key offset + 1 (0 = empty), value
LENGTH = struct.Struct('<H')


class ArchiveIndex:
    def __init__(self, directory=None, capacity=1024):
        self.directory = directory; self.lock = threading.Lock()
        self.keys = tempfile.TemporaryFile(dir=directory); self.key_end = 0
        self.capacity = capacity; self.count = 0
        self.slots = self._table(capacity)

    def _table(self, capacity):
        table = tempfile.TemporaryFile(dir=self.directory)
        os.ftruncate(table.fileno(), capacity*SLOT.size)
        return table

    def _key(self, reference):
        offset = reference-1
        (length,) = LENGTH.unpack(os.pread(self.keys.fileno(), LENGTH.size, offset))
        return os.pread(self.keys.fileno(), length, offset+LENGTH.size)

    def _find(self, table, capacity, digest, key):
        position = int.from_bytes(digest[:8], 'little') & (capacity-1)
        while True:
            stored, reference, value = SLOT.unpack(os.pread(table.fileno(), SLOT.size, position*SLOT.size))
            # key=None (rehash) only seeks an empty slot.
            if not reference or (key is not None and stored == digest and self._key(reference) == key):
                return position, reference, value
            position = (position+1) & (capacity-1)

    def get(self, key):
        with self.lock:
            _, reference, value = self._find(self.slots, self.capacity, hashlib.blake2b(key, digest_size=16).digest(), key)
            return value if reference else None

    def put(self, key, value, keep=None):
        """Upsert; `keep(old, new)` may retain an existing value."""
        if len(key) > 0xffff:raise ValueError('Index key is too long')
        digest = hashlib.blake2b(key, digest_size=16).digest()
        with self.lock:
            position, reference, old = self._find(self.slots, self.capacity, digest, key)
            if reference and keep is not None and keep(old, value):return
            if not reference:
                record = LENGTH.pack(len(key))+key
                if os.pwrite(self.keys.fileno(), record, self.key_end) != len(record):raise OSError('Short index write')
                reference = self.key_end+1; self.key_end += len(record); self.count += 1
            if os.pwrite(self.slots.fileno(), SLOT.pack(digest, reference, value), position*SLOT.size) != SLOT.size:
                raise OSError('Short index write')
            if self.count*2 > self.capacity:self._grow()

    def _grow(self):
        capacity = self.capacity*2; table = self._table(capacity); chunk = 4096
        try:
            for start in range(0, self.capacity, chunk):
                data = os.pread(self.slots.fileno(), chunk*SLOT.size, start*SLOT.size)
                for digest, reference, value in SLOT.iter_unpack(data):
                    if not reference:continue
                    position = self._find(table, capacity, digest, None)[0]
                    if os.pwrite(table.fileno(), SLOT.pack(digest, reference, value), position*SLOT.size) != SLOT.size:
                        raise OSError('Short index write')
        except BaseException:
            table.close()
            raise
        self.slots.close(); self.slots = table; self.capacity = capacity
