const QUANTUM = 128;

interface FrameBuffer {
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
      if (writePos + chunk.length > buffer.length) {
        writePos = 0;
      }
      buffer.set(chunk, writePos);
      writePos += chunk.length;

      if (writePos >= hopSize) {
        const frame = new Float32Array(hopSize);
        frame.set(buffer.subarray(0, hopSize));

        const remaining = writePos - hopSize;
        if (remaining > 0) {
          buffer.copyWithin(0, hopSize, writePos);
        }
        writePos = remaining;

        return frame;
      }

      return null;
    },
  };
}
