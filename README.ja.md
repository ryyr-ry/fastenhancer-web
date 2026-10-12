# fastenhancer-web

ブラウザ上のリアルタイム音声と録音済みファイルのノイズを除去するライブラリです。モノラル・ステレオに対応し、録音済み音声は1行で処理できます。[FastEnhancer](https://github.com/aask1357/fastenhancer)（ICASSP 2026）を中核に据え、推論処理を C で実装し、WebAssembly SIMD として配布しています。利用側からは TypeScript API として扱えます。

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

> ステレオ・録音済みファイルのAPIは `main` とデモサイトで利用できます。現在のnpm公開版（0.1.1）には、まだ含まれていません。

```bash
npm install fastenhancer-web
bun add fastenhancer-web
pnpm add fastenhancer-web
yarn add fastenhancer-web
```

[デモサイト](https://ryyr-ry.github.io/fastenhancer-web/) · [音声ファイルのデモ](https://ryyr-ry.github.io/fastenhancer-web/#/file)

> **注意:** このパッケージは **ESM専用** です。バンドラー（Vite, esbuild, webpack 5+）またはESモジュール対応ランタイム（Node.js 18+ `"type": "module"`）が必要です。CommonJS `require()` はサポートされていません。
>
> **TypeScript利用者向け:** このパッケージは `"moduleResolution": "bundler"` を使用しています。プロジェクトで `"node16"` や `"nodenext"` を使用している場合、`tsconfig.json` で `"moduleResolution": "bundler"` を設定してください。

---

## クイックスタート

### 録音済み音声を1行で処理

```typescript
import { enhanceAudio, enhanceBuffer, enhancePcm } from 'fastenhancer-web';

const cleanWav = await enhanceAudio(file); // File/Blob → 保存できるWAV Blob
const fastWav = await enhanceAudio(file, { parallel: { maxWorkers: 4 } });
const cleanBuffer = await enhanceBuffer(buffer); // AudioBuffer → AudioBuffer
const cleanPcm = await enhancePcm({ channels: [left, right], sampleRate: 44100 });
```

モノラル・ステレオのチャンネル数と長さを保持します。WAV・AudioBuffer・PCMは入力のサンプルレートを保持し、それ以外のブラウザ対応音声形式は48 kHzでデコードします。モデル読み込み、レート変換、フレーム分割、末尾処理、遅延補正、WAV出力を内部で行います。標準モデルはSmall、ブラウザの推論はmodule Workerで実行します。`{ model: 'tiny', signal, onProgress }` でモデル、キャンセル、進捗を指定できます。WAV出力は16ビットPCMです。

Node/BunでもWAVの読み込みから保存まで1行で実行できます。

```typescript
import { enhanceFile } from 'fastenhancer-web/node';
await enhanceFile('input.wav', 'output.wav');
```

### ステレオのフレーム処理

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const stereo = await model.createChannelDenoiser(2);
const [cleanLeft, cleanRight] = stereo.processFrame([leftFrame, rightFrame]);
stereo.destroy();
```

入力は48 kHz、各チャンネル512サンプルです。左右の状態とWASMメモリを独立させて処理します。音声向けのモノラルモデルを左右に適用する方式ですので、元の定位を完全に保持することまでは保証しません。ストリームAPIでは入力トラックからチャンネル数を取得し、必要に応じて `{ channelCount: 2 }` を指定できます。[設計・配布方法・制約](docs/stereo-offline.md)もご参照ください。

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

### 録音済み音声のAPI

| 関数 | 入力 → 出力 | 実行環境 |
|------|-------------|----------|
| `enhanceAudio(file, options?)` | `File`/`Blob` → 16ビットPCMのWAV `Blob` | ブラウザ（WAVはNode/Bunでも利用可） |
| `enhanceBuffer(buffer, options?)` | `AudioBuffer` → `AudioBuffer` | ブラウザ |
| `enhancePcm(audio, options?)` | `{ channels: Float32Array[], sampleRate }` → 同じ形式のPCM | ブラウザ / Node / Bun |
| `enhanceFile(inputPath, outputPath, options?)` | WAVのパス → WAV保存 | Node/Bun（`fastenhancer-web/node` からインポート） |
| `enhanceWavStream(file, options?)` | WAVの`Blob` → WAVバイト列の非同期イテレーター | ブラウザ（呼び出し元での実行にも対応） |

すべてモノラル・ステレオに対応しています。PCMとWAVのサンプルレートは8〜192 kHzの整数値に対応します。WAV以外のファイルはブラウザ対応のコーデックで読み込み、48 kHzで出力します。チャンネルの順序とデコード後のサンプル数を保持します。並列WAV処理は小さな範囲を読み込み・書き出します。PCM・AudioBuffer・圧縮音声のデコードは全体の配列が必要で、Blobを返す場合は最終出力もメモリに保持します。`enhanceWavStream` ではバイト列を順次受け取れます。

| オプション | 既定値 | 説明 |
|------------|--------|------|
| `model` | `'small'` | `'tiny'` / `'base'` / `'small'` |
| `simd` | 自動検出 | SIMDの有効・無効を指定 |
| `baseUrl` | 埋め込みアセット | セルフホストするモデルのURL |
| `signal` | — | キャンセル用の `AbortSignal` |
| `onProgress` | — | 0〜1の処理進捗を受け取るコールバック |
| `worker` | 自動 | ブラウザのmodule Workerを使用。`true` は必須、`false` は呼び出し元で定期的に処理を中断しながら実行 |
| `parallel` | `false` | `{ maxWorkers?: 1〜32, engine?: 'auto' \| 'channels' \| 'batch' }`：出力を維持した並列処理 |

`parallel` は明示的に指定し、`worker: false` との併用はエラーになります。Worker数はジョブ全体の上限で、CPUの余裕に合わせて制限し、同じ実行元の複数ジョブでも予算を共有します。`auto` は4秒以上の録音で効果が見込める場合に専用SIMDエンジンを使い、履歴を保持してフレーム・周波数位置を分担します。scalar・短い音声・自動選択で1 Workerとなる場合・従来のセルフホスト構成では、従来エンジンを使います。時間区間ごとに履歴をリセットする方式は使いません。

Node/Bunでは `enhanceFile` と並列PCMアダプターの `enhancePcm` を `fastenhancer-web/node` からインポートし、`worker` を省略してください。並列 `enhanceFile` はWAVを逐次読み書きし、成功後だけ出力を置き換えます。オプションを省略したファイルAPIは従来の動作を維持します。WAV出力は16ビットPCMで、元のコンテナメタデータは保持しません。キャンセルした結果は返却・保存されませんが、ブラウザのデコード呼び出し自体は途中で中断できません。バイト列を逐次消費する場合は、反復が失敗したら途中の出力を破棄してください。[設計・性能・配布方法](docs/offline-parallel-processing.md)もご参照ください。

### `model.createChannelDenoiser(channelCount?)` — モノラル・ステレオのフレーム処理

既定のチャンネル数は2で、1を指定するとモノラルになります。`processFrame([left, right])` は48 kHz、各チャンネル512個の有限値を受け取り、チャンネル別の配列を返します。`reset()` で時間方向の状態を初期化し、`destroy()` で両方のエンジンを解放します。`bypass`、`agcEnabled`、`hpfEnabled` は全チャンネルに適用します。ステレオの推論・メモリの負担はおおむねモノラルの2倍です。以下の処理時間の表はモノラルの計測値です。

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
| `options.channelCount` | `1 \| 2` | 出力チャンネル数（入力トラックから取得し、不明な場合はモノラル） |
| `options.onWarning` | `(msg: string) => void` | 警告コールバック |
| `options.onAutoBypass` | `(enabled: boolean) => void` | 自動保護・復帰の通知（手動バイパスとは独立） |
| `options.onError` | `(err: Error) => void` | エラーコールバック |
| `options.keepAliveInBackground` | `boolean` | バックグラウンド時も音声処理を維持（モバイル向け） |

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
// model.createChannelDenoiser(2) — 左右を独立したフレーム処理
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
│  │  負荷監視・復帰確認    │  │  持続する過負荷 → 未処理音 → 連続確認で復帰
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
| オーディオチャネル | モノラル・ステレオ — チャンネルごとに独立して処理 |
| メモリ | 全バッファ初期化時一括確保（実行時 malloc ゼロ） |
| パッケージサイズ | 全3モデル＋両バリアント埋め込み（tarball 約 1.85 MB）、ツリーシェイキング対応 |

リアルタイム処理の自動保護は、約200 msの平均負荷と超過時間の蓄積で判断します。わずかな超過が5回続いただけでバイパスする方式を改め、重大な遅延が続く場合は早く保護します。32 ms後から復帰を試し、失敗時は最大128 msまで間隔を広げます。4つの連続フレームを確認してからクロスフェードでノイズ除去へ戻します。手動バイパスも含め、未処理音と処理音のSTFT遅延をそろえます。React・ストリームのデモには自動保護中の表示を追加しています。[設計と検証の詳細](docs/performance-monitoring.md)をご参照ください。

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

AudioWorkletはストリーム処理に必要です。録音済み音声のAPIは、利用可能な場合にmodule Workerを使用し、WAV以外の形式はブラウザのコーデックで読み込みます。今回追加したステレオ・録音済み音声の処理はChromiumで検証済みです。Firefox・Safariでの追加機能の検証は未実施です。

<details>
<summary><b>CSP ディレクティブ</b></summary>

```
script-src 'self' 'wasm-unsafe-eval' blob:;
worker-src 'self' blob:;
```

- `wasm-unsafe-eval` — `WebAssembly.instantiate()` に必要
- `blob:` — デフォルトの worklet 読み込みに必要（`workletUrl` 使用時は不要）
- `worker-src` の `'self'` — 録音済み音声を処理するmodule Workerと、そのバンドル済みアセットに必要

</details>

---

## 📱 バックグラウンド音声（モバイル対応）

モバイルでアプリを切り替えると、ブラウザの音声処理が中断されることがあります。`keepAliveInBackground` を有効にすると音声の連続性を維持できます：

```typescript
// Layer 3 — React Hook
const denoiser = useDenoiser('small', { keepAliveInBackground: true });

// Layer 2 — Stream API
const sd = await model.createStreamDenoiser(mic, { keepAliveInBackground: true });
```

**動作内容：**
- サイレントオシレーターで AudioContext をアクティブに維持
- [Media Session API](https://developer.mozilla.org/docs/Web/API/Media_Session_API) にライブ音声として登録 — OS による音声パイプラインの強制停止を防止
- 一時停止/停止アクションをブロックし、誤操作による処理中断を防止
- バックグラウンド中は過負荷の確認期間と重大な遅延の連続回数を4倍に緩和
- フォアグラウンド復帰時に AudioContext を自動 resume

> **注意:** 効果はブラウザと OS の組み合わせに依存します。Chrome/Brave（Android）で最も効果的です。iOS Safari ではこれらのヒントにもかかわらず音声が中断される場合があります。

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

# vitest 全テストを実行
bun run test

# コミット済みの埋め込みアセットでステレオ・ファイル処理を検証（WASM再ビルド不要）
bunx vitest run tests/unit/api/channel-denoiser.test.ts tests/unit/api/offline.test.ts tests/unit/api/node-file.test.ts

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

| スイート | 環境 | フレームワーク | 目的 |
|----------|------|----------------|------|
| C native | gcc (MinGW) | Unity | C モジュールの正しさ |
| WASM | Emscripten (scalar + SIMD) | vitest | Emscripten 変換と SIMD の正しさ |
| TypeScript unit / adversarial | Node.js | vitest | API・worklet・ライフサイクル・左右の独立性・ファイル処理 |
| Browser E2E | Chromium / Firefox / WebKit | Playwright | AudioWorklet と録音済み音声用module Workerの統合 |

ステレオ・録音済み音声・性能監視・並列処理では、**単体・異常系420件**と**Chromiumの選択した41件**が通過しています。Base・Smallのscalar/SIMDの性能試験、ステレオの負荷変動・復帰、内部並列化の出力一致も含みます。BunでもPCM・WAVが従来処理と一致しました。本番用デモでは配信先と同じサブパスで、モノラル／ステレオ、標準／並列処理のWAV保存を確認しています。C・WASM・全ブラウザをすべて再実行した結果ではありません。[測定と検証範囲](docs/offline-parallel-processing.md)をご参照ください。

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
