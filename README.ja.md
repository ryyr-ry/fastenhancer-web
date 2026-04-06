# fastenhancer-web

ブラウザ上でリアルタイムに通話音声のノイズを除去するためのライブラリです。[FastEnhancer](https://github.com/aask1357/fastenhancer)（ICASSP 2026）を中核に据え、推論処理を C で実装し、WebAssembly SIMD として配布しています。利用側からは整理された TypeScript API として扱えます。

> **English documentation: [README.md](./README.md)**

---

## なぜ fastenhancer-web か

ブラウザ向けのノイズ除去は、汎用の推論ランタイムに依存する構成になりがちです。たとえば ONNX Runtime Web v1.24.3 の標準 WASM ファイル（`ort-wasm-simd-threaded.wasm`）は、それ単体で **11.79 MB** あります。これは jsdelivr CDN 上で確認した実測値です。ここに JavaScript のグルーコードやモデル重みまで加わると、初回ロードはすぐに数 MB 規模になります。**fastenhancer-web はニューラルネットワーク推論そのものを C で実装しているため、追加のランタイムを不要にし、配布サイズを最小限に抑えられます。**

| モデル | パラメータ数 | WASM + 重み (gzip) | 処理時間 (SIMD) | 予算使用率 |
|-------|-----------|----------------------|----------------------|-------------------|
| **Tiny** | 28K | **124 KB** | 0.45 ms | 4.2% |
| **Base** | 101K | **391 KB** | 1.63 ms | 15% |
| **Small** | 207K | **780 KB** | 3.88 ms | 36% |

- 48 kHz ネイティブ処理 — リサンプリング由来の劣化なし
- relaxed-simd FMA を使った WASM SIMD 高速化
- 実行時 `malloc` ゼロ — 必要メモリの初期化時一括確保
- SharedArrayBuffer / COOP / COEP ヘッダー不要
- CSP 対応（`unsafe-eval` 不要）

---

## インストール

```bash
npm install fastenhancer-web
bun add fastenhancer-web
pnpm add fastenhancer-web
yarn add fastenhancer-web
```

> **注意:** このパッケージは **ESM専用** です。バンドラー（Vite, esbuild, webpack 5+）またはESモジュール対応ランタイム（Node.js 18+ `"type": "module"`）が必要です。CommonJS `require()` はサポートされていません。
>
> **TypeScript利用者向け:** このパッケージは `"moduleResolution": "bundler"` を使用しています。プロジェクトで `"node16"` や `"nodenext"` を使用している場合、`tsconfig.json` で `"moduleResolution": "bundler"` を設定してください。

---

## クイックスタート

### Layer 3: React Hook

```tsx
import { useDenoiser } from 'fastenhancer-web/react';

function CallScreen() {
  const { outputStream, start, stop, state } = useDenoiser('small');

  return (
    <div>
      <button onClick={start} disabled={state === 'loading'}>
        {state === 'processing' ? 'ノイズ除去中...' : 'ノイズ除去開始'}
      </button>
      <button onClick={stop} disabled={state !== 'processing'}>
        停止
      </button>
      {outputStream && <audio autoPlay ref={el => {
        if (el) el.srcObject = outputStream;
      }} />}
    </div>
  );
}
```

### Layer 2: ストリーム API

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createStreamDenoiser(micStream);
const cleanStream = denoiser.outputStream;
// ...
denoiser.destroy();
```

### Layer 1: フレーム単位処理

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createDenoiser();
const output = denoiser.processFrame(inputFloat32Array); // 48 kHz で 512 サンプル
denoiser.destroy();
```

### ゼロコンフィグ vs セルフホスト

デフォルトでは WASM バイナリとモデル重みは **JavaScript モジュール内に base64 として埋め込まれています**。外部依存ゼロ — `.wasm` ファイルの配信不要、CDN 不要、CORS の心配なし：

```typescript
// ゼロコンフィグ: すべて埋め込み（デフォルト）
const model = await loadModel('small');

// セルフホスト: CDN からフェッチ
const model = await loadModel('small', { baseUrl: 'https://cdn.example.com/models/' });
```

ツリーシェイキング対応バンドラーは実際にインポートしたモデルのみを含みます。

---

## API リファレンス

### `useDenoiser(modelSize, options?)` — React Hook

```typescript
import { useDenoiser } from 'fastenhancer-web/react';

const {
  state,          // 'idle' | 'loading' | 'processing' | 'error' | 'destroyed'
  error,          // Error | null
  inputStream,    // MediaStream | null
  outputStream,   // MediaStream | null
  bypass,         // boolean
  start,          // (inputStream?: MediaStream) => Promise<void>
  stop,           // () => void
  setBypass,      // (enabled: boolean) => void
  destroy,        // () => void
} = useDenoiser('small');
```

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `modelSize` | `'tiny' \| 'base' \| 'small'` | 使用するモデルサイズ（必須） |
| `options.baseUrl` | `string` | WASM/重みファイルのベース URL |
| `options.simd` | `boolean` | SIMD の有効/無効を明示指定（デフォルトは自動検出） |
| `options.workletUrl` | `string` | カスタム AudioWorklet processor の URL |
| `options.audioConstraints` | `MediaTrackConstraints` | 自動 getUserMedia 時のオーディオ制約 |
| `options.onWarning` | `(msg: string) => void` | 警告コールバック |
| `options.onError` | `(err: Error) => void` | エラーコールバック |

**主な特性:**
- `start()` 引数なしでマイクを自動取得（getUserMedia を内部で呼び出し）
- `start(stream)` で既存の MediaStream を使用（WebRTC、カスタム制約など）
- Hook が取得したマイクは stop/destroy/アンマウント時に自動解放
- アンマウント時の自動クリーンアップ
- React 18+ の Strict Mode での安全な利用（二重 mount / unmount への耐性）
- race condition への強さと古い `start()` 結果の自動破棄

### `loadModel(modelSize, options?)` — モデル・リソースローダー

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
// model.createDenoiser()         — Layer 1: フレーム単位処理
// model.createStreamDenoiser()   — Layer 2: リアルタイム AudioWorklet ストリーム
// model.wasmBytes                — 生の WASM バイナリ（上級者向け）
// model.weightData               — 重みバイナリ（上級者向け）
// model.exportMap                — WASM エクスポート名マッピング（上級者向け）
```

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `modelSize` | `'tiny' \| 'base' \| 'small'` | 使用するモデルサイズ（必須） |
| `options.baseUrl` | `string` | リソースファイルのベース URL（省略時はゼロコンフィグの埋め込みロード） |
| `options.simd` | `boolean` | SIMD の有効/無効を明示指定（デフォルトは自動検出） |

結果はキャッシュされます — `loadModel('small')` を2回呼んでも同一の Promise が返ります。

### `createDenoiser(options)` — フレーム単位処理 API（低レベル）

```typescript
import { createDenoiser } from 'fastenhancer-web';
```

> **注意:** 通常は `loadModel('small').then(m => m.createDenoiser())` の利用を推奨します。リソースのロードを自動的に処理します。直接の `createDenoiser()` 呼び出しには WASM ファクトリと重みデータを手動で渡す必要があります。

### `createStreamDenoiser(options)` — AudioWorklet 統合（低レベル）

```typescript
import { createStreamDenoiser } from 'fastenhancer-web';
```

> **注意:** 通常は `loadModel()` 経由の `model.createStreamDenoiser(micStream)` の利用を推奨します。WASM、重み、エクスポートマップを自動処理します。直接の `createStreamDenoiser()` 呼び出しにはすべてのバイナリリソースを手動で渡す必要があります。

### `diagnose()` — ブラウザ互換性チェック

```typescript
import { diagnose } from 'fastenhancer-web';

const result = await diagnose();
// { wasm: true, simd: true, audioContext: true, audioWorklet: true, overall: true, recommended: true, issues: [] }
```

### `getModels()` / `recommendModel(options?)` — モデル選択

```typescript
import { getModels, recommendModel } from 'fastenhancer-web';

const models = getModels();
// [{ id: 'tiny', ... }, { id: 'base', ... }, { id: 'small', ... }]

const recommended = recommendModel({ priority: 'quality' });
// { id: 'small', reason: 'Highest-quality noise removal. Recommended for most environments.' }
```

### エラークラス

すべてのエラーは `FastEnhancerError` を継承しており、機械可読な `code` プロパティを持ちます。

| クラス | コード | 発生条件 |
|-------|------|------|
| `WasmLoadError` | `WASM_LOAD_FAILED` | WASM モジュールの読み込み失敗時 |
| `ModelInitError` | `MODEL_INIT_FAILED` | モデル初期化失敗時（重み破損、CRC 不一致など） |
| `AudioContextError` | `AUDIO_CONTEXT_ERROR` | AudioContext の生成や状態遷移の失敗時 |
| `WorkletError` | `WORKLET_ERROR` | AudioWorklet の登録や通信の失敗時 |
| `ValidationError` | `VALIDATION_ERROR` | 不正な引数 |
| `DestroyedError` | `DESTROYED_ERROR` | 破棄済みインスタンスへの操作 |

```typescript
import { WasmLoadError, DestroyedError } from 'fastenhancer-web/errors';
```

---

## モデル比較

すべてのモデルは 48 kHz 音声をネイティブ処理し、hop size は 512 サンプルです（1 フレームあたりの予算は 10.67 ms です）。

| | Tiny | Base | Small |
|---|------|------|-------|
| **パラメータ数** | 28K | 101K | 207K |
| **WASM SIMD サイズ** | 46 KB | 45 KB | 46 KB |
| **重みサイズ** | 111 KB | 397 KB | 814 KB |
| **合計 (gzip)** | **124 KB** | **391 KB** | **780 KB** |
| **処理時間 (SIMD median)** | 0.45 ms | 1.63 ms | 3.88 ms |
| **処理時間 (SIMD P99)** | 0.67 ms | 1.89 ms | 4.77 ms |
| **予算使用率** | 4.2% | 15% | 36% |
| **RNNFormer ブロック数** | 2 | 3 | 3 |
| **Encoder ブロック数** | 2 | 2 | 3 |
| **チャネル数** | 24 | 48 | 64 |

**推奨:**
- **Small** — 多くの用途で推奨。最高品質のノイズ除去を提供。
- **Base** — 品質と速度のバランスが良い。モバイル端末にも適する。
- **Tiny** — 最小リソース消費。低消費電力環境やレイテンシ最優先の用途向け。

---

## アーキテクチャ

```
マイク入力 (48 kHz)
    │
    ▼
┌─────────────────────────────┐
│  AudioWorkletProcessor      │  ← オーディオスレッド上で動作
│  ┌───────────────────────┐  │
│  │  フレームバッファ       │  │  4 × 128 quantum → 512 サンプル
│  │  WASM SIMD エンジン    │  │  FFT → Encoder → RNNFormer → Decoder → iFFT
│  │  フレームドロップ検知   │  │  5 連続ドロップ → 自動バイパス
│  └───────────────────────┘  │
└─────────────────────────────┘
    │
    ▼
ノイズ除去済み出力 (48 kHz)
```

**設計上の要点:**

| 設計判断 | 詳細 |
|----------|------|
| ロード戦略 | デフォルトは埋め込み JS モジュール、`baseUrl` 指定時は `fetch()` |
| Worklet 読み込み | インライン Blob URL（外部ファイル不要）、厳格 CSP には `workletUrl` |
| WASM インスタンス化 | 生の `WebAssembly.instantiate()` — worklet 内に Emscripten グルーなし |
| オーディオチャネル | モノラル専用 — ステレオは最初のチャネルにダウンミックス |
| メモリ | 全バッファ初期化時一括確保（実行時 malloc ゼロ） |
| パッケージサイズ | 全3モデル＋両バリアント埋め込み（tarball 約 1.85 MB）、ツリーシェイキング対応 |

---

## ブラウザ対応

| ブラウザ | 状態 | 備考 |
|---------|--------|-------|
| Chrome 91+ | ✅ 完全対応 | WASM SIMD + AudioWorklet 対応 |
| Edge 91+ | ✅ 完全対応 | Chromium ベース |
| Firefox 89+ | ✅ E2E 検証済み | WASM SIMD の 89 以降での利用可 |
| Safari 16.4+ | ⚠️ 未検証 | WASM SIMD の 16.4 以降での利用可 |

**要件:**
- WASM SIMD 対応（非対応時は scalar へフォールバック）
- AudioWorklet 対応
- `SharedArrayBuffer` / `Cross-Origin-Isolation` ヘッダー不要

<details>
<summary><b>CSP ディレクティブ</b></summary>

```
script-src 'self' 'wasm-unsafe-eval' blob:;
worker-src blob:;
```

- `wasm-unsafe-eval` — `WebAssembly.instantiate()` に必要
- `blob:` — デフォルトの worklet 読み込みに必要（`workletUrl` 使用時は不要）

</details>

---

## 比較

| | fastenhancer-web | rnnoise-wasm | ONNX Runtime Web |
|---|---|---|---|
| **最小バンドル** | **124 KB** (Tiny, gzip) | ~95 KB | 11.79 MB (WASM のみ) |
| **ゼロコンフィグ** | ✅ JS に埋め込み | ❌ `.wasm` 配信必要 | ❌ `.wasm` + `.onnx` 配信必要 |
| **特殊ヘッダー** | ✅ 不要 | ✅ 不要 | ❌ SharedArrayBuffer 必要（マルチスレッド時） |
| **サンプルレート** | 48 kHz ネイティブ | 48 kHz | モデル依存 |
| **モデル品質** | FastEnhancer (ICASSP 2026) | RNNoise (2018) | モデル依存 |
| **実行時 malloc** | ゼロ | ゼロ | あり |
| **ツリーシェイキング** | ✅ モデル単位 | N/A | N/A |
| **TypeScript 型** | ✅ 内蔵 | ❌ | ✅ 内蔵 |
| **React フック** | ✅ 内蔵 | ❌ | ❌ |

---

## 開発

```bash
# 依存関係をインストール
bun install

# vitest 全テストを実行（305 テスト、22 ファイル）
bun run test

# TypeScript をビルド
bun run build:ts

# すべての WASM バリアントをビルド（Emscripten SDK が必要）
bun run build:wasm:all

# 全体をビルド（WASM + TS）
bun run build:all

# E2E ブラウザテストを実行（Playwright ブラウザが必要）
bunx playwright test
```

### テスト構成

| スイート | 環境 | フレームワーク | テスト数 | 目的 |
|-------|-------------|-----------|-------|---------|
| C native | gcc (MinGW) | Unity | ~201 | C モジュールの正しさ |
| WASM | Emscripten (scalar + SIMD) | vitest | 30 | Emscripten 変換 + SIMD 正当性 |
| TypeScript unit | Node.js | vitest | 275 | API / worklet / ライフサイクルの正しさ |
| Browser E2E | Chrome + Firefox + WebKit | Playwright | 75 | AudioWorklet 統合 |
| **合計** | | | **~581** | |

---

## 数値精度

PyTorch の参照実装と WASM SIMD 出力を 40 フレームで比較した結果は次のとおりです。

| モデル | MSE | 最大絶対差 |
|-------|-----|------------------------|
| Tiny | 2.69 × 10⁻¹⁴ | 3.52 × 10⁻⁶ |
| Base | 1.37 × 10⁻¹⁴ | 4.23 × 10⁻⁷ |
| Small | 1.69 × 10⁻¹⁴ | 5.85 × 10⁻⁷ |

---

## ライセンス

MIT

## クレジット

- [FastEnhancer](https://github.com/aask1357/fastenhancer) — 元になっているニューラルネットワークアーキテクチャ（ICASSP 2026）
- [fastenhancer.c.wasm](https://github.com/kdrkdrkdr/fastenhancer.c.wasm) — 参照となる C/WASM 実装
