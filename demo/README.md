# fastenhancer-web デモ

ブラウザ上でリアルタイム通話ノイズ除去を体験できるデモアプリケーション。

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

## 技術スタック

- React + TypeScript + Vite
- fastenhancer-web（親パッケージをソース参照）

