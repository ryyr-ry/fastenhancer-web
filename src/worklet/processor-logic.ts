/**
 * AudioWorklet処理スケジューラ
 * フレーム処理の予算管理、フレームドロップ判定、
 * 連続超過時のパススルー遷移を制御する。
 */

interface FrameInfo {
  dropped: boolean;
  processingTimeMs: number;
  budgetMs: number;
}

interface ProcessorScheduler {
  readonly budgetMs: number;
  readonly isPassthrough: boolean;
  recordProcessingTime(ms: number): FrameInfo;
}

const OVERRUN_THRESHOLD = 5;
const RECOVERY_THRESHOLD = 3;

export function createProcessorScheduler(hopSize: number, sampleRate: number): ProcessorScheduler {
  const budgetMs = (hopSize / sampleRate) * 1000;
  let consecutiveOverruns = 0;
  let consecutiveSuccesses = 0;
  let passthrough = false;

  return {
    get budgetMs() { return budgetMs; },
    get isPassthrough() { return passthrough; },

    recordProcessingTime(ms: number): FrameInfo {
      const dropped = ms > budgetMs;

      if (passthrough) {
        if (dropped) {
          consecutiveSuccesses = 0;
        } else {
          consecutiveSuccesses++;
          if (consecutiveSuccesses >= RECOVERY_THRESHOLD) {
            passthrough = false;
            consecutiveOverruns = 0;
            consecutiveSuccesses = 0;
          }
        }
      } else {
        if (dropped) {
          consecutiveOverruns++;
          consecutiveSuccesses = 0;
          if (consecutiveOverruns >= OVERRUN_THRESHOLD) {
            passthrough = true;
            consecutiveSuccesses = 0;
          }
        } else {
          consecutiveOverruns = 0;
        }
      }

      return { dropped, processingTimeMs: ms, budgetMs };
    },
  };
}
