# fastenhancer-web デモ

リアルタイム音声と録音済みファイルのノイズ除去を体験できるデモアプリケーション。

## 起動方法

```bash
cd demo
bun install
bun run dev
```

ブラウザで `http://localhost:5173` を開く。

## 機能

- マイク入力のリアルタイムノイズ除去
- モデルサイズ切り替え（Tiny / Base / Small）
- バイパス ON/OFF
- 処理パフォーマンス表示
- モノラル・ステレオ音声ファイルの処理、キャンセル、WAV保存
- ファイル処理の「高速処理」切り替え（並列化・WAVの逐次読み込み）

## 技術スタック

- React + TypeScript + Vite
- fastenhancer-web（親パッケージをソース参照）

