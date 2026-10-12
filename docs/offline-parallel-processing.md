# 長時間音声の並列処理

`parallel` を指定すると、既存の出力を保ちながら複数のWorkerへ計算を分配します。オプションを省略したAPIは従来の経路を使います。時間区間を独立したモデルで処理する方法は使わず、各GRUの履歴とSTFTの合成順を維持します。

```ts
import { enhanceAudio, enhanceWavStream } from 'fastenhancer-web';

const cleanWav = await enhanceAudio(file, { parallel: { maxWorkers: 4 } });

// WAV入力だけを扱い、エンコード済みのバイト列を順次受け取ります。
// 消費側が次を要求するまで処理を進めず、キャンセルやbreakで資源を解放します。
for await (const bytes of enhanceWavStream(file, { parallel: { maxWorkers: 4 }, signal })) {
  await sink.write(bytes);
}
```

```ts
import { enhanceFile, enhancePcm } from 'fastenhancer-web/node';

await enhanceFile('input.wav', 'output.wav', { parallel: { maxWorkers: 4 } });
const pcm = await enhancePcm(audio, { parallel: { maxWorkers: 4 } });
```

Node/Bunで並列PCMを処理する場合は、Node用の入口を使います。ルートのPCM APIはNode Worker threadsをブラウザ用バンドルへ混入させません。`worker: false` と並列指定の併用はValidationErrorになります。

## 選択と上限

| 指定 | 動作 |
|---|---|
| `parallel` 省略／`false` | 従来の処理経路 |
| `{ maxWorkers: 4 }` | 上限4、エンジンは`auto` |
| `{ engine: 'channels' }` | 左右の履歴を持つ従来エンジンと、並列リサンプラー |
| `{ engine: 'batch' }` | 専用SIMDエンジン。scalarの場合は従来エンジンで数値結果を維持 |

`maxWorkers` は1〜32の整数です。指定値はチャンネルごとの数ではなく、ジョブ全体の上限です。実際にはCPUの余裕を残して制限します。既定の上限は4です。Node/LinuxはCPU affinityに加え、利用できる場合はcgroup v2のCPU割当も確認します。

同じ実行元のジョブはWorker予算を共有し、必要数をまとめて確保します。待機中の中止でキューを解除し、終了処理が完了してから予算を返します。タブ・別プロセス間のCPU予算を統合する仕組みではありません。

`auto` は、4秒以上・SIMD・埋め込みモデル・複数Workerという条件で専用エンジンを使います。ステレオでは3つ以上のWorkerが使える場合に選びます。短いファイル、1 Worker、自前配信、SIMDを使えない環境では従来エンジンで処理します。モデルの自動変更や未処理音へのバイパスは行いません。

## 状態を維持する内部並列化

専用エンジンは最大64フレームのバッチを処理します。

1. STFT・Encoder・PreNetはフレーム範囲を分配します。各範囲に直前の実入力512サンプルを与えます。
2. 各RNNFormerブロックのGRUは周波数位置を分配します。同じ位置の状態は同じWorkerが所有し、全バッチで時間順を維持します。
3. GRUの全位置が完成してから、Attentionをフレーム範囲で計算します。次のGRUブロックもこの順序を守ります。
4. PostNet・Decoder・iFFTをフレーム範囲で計算し、呼び出し元が元の順序でoverlap-addします。

周波数位置やフレームの分配では、行列の加算を別順序へ変更しません。共有メモリやpthreadを使わず、通常のmodule Worker間で内部所有の配列を転送します。COOP/COEPやSharedArrayBufferは不要です。

既存のリアルタイムWASMとWorkletは変更していません。追加のWASMはオフライン専用です。再ビルドしたscalarの通常処理には最大約3e-7の丸め差が出たため、scalarの処理には従来バイナリを維持し、追加scalarバイナリは同梱しません。

## レート変換とメモリ

リサンプラーは既存の窓付きsincを使い、左右の係数計算を共有します。出力位置はファイル全体の位置から計算し、必要な入力の前後だけを渡します。境界で位相をリセットせず、ファイル端の正規化・各サンプルの加算順・元のサンプル数を維持します。

並列 `enhanceAudio` のWAV経路と並列 `enhanceFile` は、必要な範囲を順次読み込み、モデル出力の次の変換に必要な範囲だけを保持してPCM16へ書き出します。全タイルを先にコピーしてキューへ積みません。内部の作業メモリは録音時間に比例して増える構成ではありません。

`enhanceFile` は一時ファイルへ書き、読み書きのハンドルを閉じ、成功後だけ出力をrenameで置き換えます。同じ入出力パスも扱います。キャンセル、途中の異常、進捗コールバックの例外では、一時出力を削除して既存の出力を保持します。

PCM/AudioBufferのAPIは元の全配列と完成した全配列が必要です。Blobを返すAPIには最終WAVのサイズ分のメモリが必要です。圧縮形式の `decodeAudioData` は全体をデコードし、途中では中断できません。これらの制約はWorkerを増やしても解消しません。

`enhanceWavStream` の出力は通常のRIFF PCM16 WAVです。既存の約4 GiBのRIFFサイズ制限を維持します。消費側は途中のバイト列を既に受け取るため、反復が失敗した場合は完成品として扱わず、出力を破棄してください。進捗1は計算・バイト列の生成の完了を示し、Nodeの保存完了は `enhanceFile` のPromiseの解決で確認します。

## 実測

Node 24.19.0、AMD EPYC 9V45、CPU割当4コアで、UIなどの余裕を残す3 Workerを使いました。16秒の実音声、SIMD、起動・転送を含む比較です。並列値は2回の平均です。すべての出力サンプルが従来処理と一致しています。

| モデル | チャンネル数 | 従来 | 内部並列化 | 速度比 |
|---|---:|---:|---:|---:|
| Base | 1 | 2.015秒 | 0.928秒 | 2.17倍 |
| Base | 2 | 3.891秒 | 1.885秒 | 2.06倍 |
| Small | 1 | 4.565秒 | 1.799秒 | 2.54倍 |
| Small | 2 | 9.146秒 | 3.588秒 | 2.55倍 |

端末、CPU割当、同時に動くアプリ、発熱によって速度は変わります。1時間のファイルや熱による性能低下を測った数値ではありません。詳細な測定記録は開発環境の `/workspace/onboarding/offline-parallel-study/implemented-results.json` に保存しています。

## ビルドと配布

通常のTypeScript・デモ・CIビルドはGitに保存した埋め込みアセットを使い、Emscriptenは不要です。Cのオフライン実装を変更する場合は、Emscripten 3.1.69を使って次を実行します。

テスト用のWASM・重みがないクリーンなチェックアウトでは `bun run restore:assets` で復元します。保存済みアセットから不足分だけを生成し、手元の再ビルド結果を上書きしません。CIでもこの手順の後に全単体・異常系テストを実行します。

```sh
EMCC=/path/to/emcc bun run build:wasm:offline
bunx tsc
```

ビルドスクリプトはコンパイラーの版を確認し、追加のSIMDアセットだけを生成します。既存のリアルタイムバイナリを再生成しません。コンパイラーを変更する場合は数値互換を再確認する必要があります。

バンドラーではES形式のmodule Workerとモデルの動的importが必要です。Viteでは `worker.format: 'es'` を使います。フレームエンジンはWorkerの入口から独立したモジュールにし、Worker同士の循環importを避けています。

自前配信では従来のWASM・重み・export mapに加え、`engine: 'batch'` を明示する場合は `fastenhancer-{model}-simd-offline.wasm` を同じモデルのbaseUrlへ配置してください。自前配信の`auto`は従来エンジンを使うため、追加ファイルは必須ではありません。CSPはmodule WorkerとWASMの実行を許可する必要があります。

## 検証

全3モデルのSIMD、scalarへのフォールバック、単一・複数Worker、モノラル・ステレオ、複数バッチにまたがる履歴、短い入力、末尾、片側無音、左右同一、極端な有限値を従来処理と比較します。既存の未使用Nyquist値がオーバーフローする入力も、従来と同じ有限な出力を維持します。

リサンプラーは84入力条件・252分割ケースで既存の値と比較します。WAVは各PCMビット深度、float32、拡張形式、範囲読み出し、遅延の排出、途中の不正サンプルを確認します。Workerの予算、起動失敗、異常な応答、兄弟Workerの停止、キャンセル、遅い通知、コールバック例外、早期の反復終了も検証します。

調査時の時間分割・履歴の差については、[実装前の調査記録](offline-parallelism-plan.md)をご参照ください。

単体・異常系420件、Chromiumの選択した41件が通過しています。Bun 1.3.11でもSmall・モノラルの自動内部並列化とWAV保存を比較しました。本番用デモは実際のバンドルとWorkerを使い、5秒の44.1 kHz音声、モノラル／ステレオ、標準／高速処理の4ケースで全WAVバイト列の一致を確認しました。Firefox・Safari、全native/WASMスイート、1時間の実ファイルのピークメモリは未検証です。
