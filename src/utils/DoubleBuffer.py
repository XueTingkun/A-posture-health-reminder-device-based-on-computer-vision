import threading
import json
import time
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class BufferState(Enum):
    EMPTY = 0
    READY = 1
    READING = 2


@dataclass
class FrameBuffer:
    """Buffer slot storing a single frame of data"""

    json_meta: Optional[Dict[str, Any]] = None
    image_data: Optional[memoryview] = None
    timestamp: float = 0.0
    state: BufferState = BufferState.EMPTY

    def clear(self):
        """Release references to allow GC reclamation"""
        self.json_meta = None
        self.image_data = None
        self.state = BufferState.EMPTY


class DoubleBuffer:
    """
    Thread-safe double buffer class for high-speed exchange of JSON metadata
    and binary image data
    """

    def __init__(self, drop_frames: bool = False):
        self._drop_frames = drop_frames
        self._buffers = [FrameBuffer(), FrameBuffer()]
        self._write_idx = 0
        self._read_idx = 1
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame_count = 0

    def write(
        self, metadata: Dict[str, Any], image: Union[bytes, bytearray, memoryview]
    ) -> bool:
        # Convert to memoryview (zero-copy view)
        if isinstance(image, (bytes, bytearray)):
            image_view = memoryview(image)
        else:
            image_view = image

        with self._lock:
            write_buf = self._buffers[self._write_idx]

            # Block if not dropping frames and buffer is unread
            if not self._drop_frames and write_buf.state == BufferState.READY:
                return False

            # Write data
            write_buf.json_meta = metadata
            write_buf.image_data = image_view
            write_buf.timestamp = time.time()
            write_buf.state = BufferState.READY

            # Swap buffers (atomic operation)
            self._write_idx, self._read_idx = self._read_idx, self._write_idx
            self._frame_count += 1

            # Notify waiting reader threads
            self._cond.notify()

        return True

    def try_read(self) -> Optional[Tuple[Dict[str, Any], memoryview, float, int]]:
        """
        Non-blocking read

        Returns:
            (metadata, image_view, timestamp, buffer_id) or None
            buffer_id is used for subsequent release() call
        """
        with self._lock:
            read_buf = self._buffers[self._read_idx]

            if read_buf.state != BufferState.READY:
                return None

            read_buf.state = BufferState.READING
            buf_id = self._read_idx

            return (read_buf.json_meta, read_buf.image_data, read_buf.timestamp, buf_id)

    def read(
        self, timeout: Optional[float] = None
    ) -> Tuple[Dict[str, Any], memoryview, float, int]:
        """
        Blocking read until new data is available

        Returns:
            (metadata, image_view, timestamp, buffer_id)
            buffer_id must be passed to release() to free the buffer

        Raises:
            TimeoutError: Timeout expired without receiving new frame
        """
        with self._cond:
            deadline = time.time() + timeout if timeout else None

            # CRITICAL FIX: Use loop to check condition and re-fetch read_buf each iteration
            # Because _read_idx may have been swapped by producer after wait() returns
            while True:
                read_buf = self._buffers[self._read_idx]

                if read_buf.state == BufferState.READY:
                    break

                if timeout:
                    remaining = deadline - time.time()
                    if remaining <= 0 or not self._cond.wait(timeout=remaining):
                        raise TimeoutError("Read timeout, no new frame received")
                else:
                    self._cond.wait()

            # Mark as reading and record current buffer ID
            read_buf.state = BufferState.READING
            buffer_id = self._read_idx

            return (
                read_buf.json_meta,
                read_buf.image_data,
                read_buf.timestamp,
                buffer_id,
            )

    def release(self, buffer_id: int):
        """
        Release the buffer after reading is complete, allowing it to be written again

        Args:
            buffer_id: Buffer ID returned by read() or try_read() (0 or 1)
        """
        with self._lock:
            if 0 <= buffer_id < len(self._buffers):
                buf = self._buffers[buffer_id]
                if buf.state == BufferState.READING:
                    buf.clear()

    @property
    def frame_count(self) -> int:
        return self._frame_count
