const QUANTUM = 128;

interface FrameBuffer {
  // Returns at most one frame per push call.
  // Chunks are expected to be no larger than one AudioWorklet quantum.
  push(chunk: Float32Array): Float32Array | null;
  readonly pendingSamples: number;
}

export function createFrameBuffer(hopSize: number): FrameBuffer {
  const buffer = new Float32Array(hopSize + QUANTUM);
  let writePos = 0;

  return {
    get pendingSamples(): number {
      return writePos;
    },

    push(chunk: Float32Array): Float32Array | null {
      const copyLen = Math.min(chunk.length, buffer.length - writePos);
      buffer.set(chunk.subarray(0, copyLen), writePos);
      writePos += copyLen;

      if (writePos >= hopSize) {
        const frame = new Float32Array(hopSize);
        frame.set(buffer.subarray(0, hopSize));

        const remaining = writePos - hopSize;
        if (remaining > 0) {
          buffer.copyWithin(0, hopSize, writePos);
        }
        writePos = remaining;

        if (copyLen < chunk.length) {
          const unconsumed = chunk.subarray(copyLen);
          const reCopy = Math.min(unconsumed.length, buffer.length - writePos);
          buffer.set(unconsumed.subarray(0, reCopy), writePos);
          writePos += reCopy;
        }

        return frame;
      }

      return null;
    },
  };
}
