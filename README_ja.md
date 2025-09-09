# Poly Mask Annotator for Jupyter

> **目的**  
> **Jupyter 上だけで完結**する、コピペで動かせる **ポリゴン型セグメンテーション用アノテーター**。  
> 研究室クラスター等の **外部にデータを出せない環境**でも利用できる **オフライン前提**のツールです。

![Sample_GIF](Animation.gif)

**現行バージョン:** v1.8.2

- 🖱️ ポリゴン描画・編集（追加／移動／削除／辺への挿入）
- ✨ Chaikin 平滑化 + 頂点ごとの **Smooth/Sharp** 指定（角の保持）
- 🔍 ズーム（スライダ / **Fit** / **100%** / **ホイール**、範囲を指定可能）
- 🧭 パン：従来パン **または** **ビューポートのドラッグスクロール**（再描画なし）
- 💾 画像ごとに **JSON サイドカー**（自動保存／読込）
- 🖼️ **uint8 インデックスマスク** + **カラーのプレビューマスク**を書き出し
- 🧰 親レイアウトによる縦横比の歪みを防ぐスクロール可能ビューポート

## v1.8.2 の新機能

- **Polyline（開いた線）のサポート**  
  - 描画前に **Type: Area / Polyline** を切り替え可能。  
  - **Area**（閉領域）：従来通り塗りつぶし＋境界線、エクスポートも塗りつぶし。  
  - **Polyline**（開いた線）：Finish しても閉じず、表示は線のみ、エクスポートは **1px の線マスク**。  
  - JSON サイドカーに `open_path` を保存・復元。  
  - 編集モード：開いた線は線分への近接クリック（約6px以内）で選択可能。  
  - 点の挿入／削除も open/closed を区別して処理。  
- **ツールバーのレイアウト改善** 

---

## このツールの狙い

- **オフライン／閉域環境対応**：Web 通信なし。パッケージを事前配置すればクラスター内で完結。
- **サーバ構築不要**：ノートブック内で動作。Qt 等のデスクトップGUI不要、ポート開放不要。
- **コピペで起動**：最小限のセルで起動。任意で「再描画なしパン」も追加可能。

---

## 必要環境

- Python **3.9〜3.11**（推奨）
- JupyterLab **4** または Notebook **7**（ipywidgets v8 対応）
- パッケージ：
  - `ipywidgets >= 8.0.0`
  - `ipycanvas >= 0.13.0`
  - `Pillow >= 10.0.0`
  - `scikit-image >= 0.21.0`
  - `numpy >= 1.24.0`

> **旧Jupyter（例: JupyterLab 3）** でも動く場合はありますが、`jupyterlab-widgets` 等の拡張が必要になることがあります。  
> 可能なら **JupyterLab 4 / Notebook 7** を推奨します。

### インストール（オンライン）

```bash
pip install -U ipywidgets ipycanvas pillow scikit-image numpy
```

### インストール（オフライン / クラスター）

1. ネット接続のある環境で wheelhouse を作成：
   ```bash
   mkdir wheelhouse
   pip download -d wheelhouse ipywidgets ipycanvas pillow scikit-image numpy
   ```
2. `wheelhouse/` をクラスターへ転送。
3. クラスター側で：
   ```bash
   pip install --no-index --find-links=wheelhouse ipywidgets ipycanvas pillow scikit-image numpy
   ```

> **補足（Linux x86_64、CPU最適化）**
> ```bash
> pip uninstall -y pillow
> pip install --upgrade --force-reinstall pillow-simd
> ```

---

## クイックスタート（コピペ）

> リポジトリの `poly_mask_annotator.py` をノートブックと同じフォルダに置き、以下を貼り付け：

```python
from IPython.display import HTML
HTML(\"\"\"
<style>
.pm-wheel-passive { touch-action: auto; overscroll-behavior: auto; }
/* 任意：ドラッグスクロール用カーソル表示 */
.pm-drag-scroll.pm-drag-scroll-active { cursor: grab; }
.pm-drag-scroll.pm-drag-scroll-active:active { cursor: grabbing; }
</style>
<script>
/* 任意：Pan ON 中はビューポートの左ドラッグでスクロール（再描画なし） */
(function(){
  function enableDragScroll(root){
    if (!root || root.__pmDragScrollInstalled) return;
    root.__pmDragScrollInstalled = true;
    root.addEventListener('pointerdown', function(e){
      if (!root.classList.contains('pm-drag-scroll-active')) return;
      if (e.button !== 0) return; // 左ボタンのみ
      const startX = e.clientX + root.scrollLeft;
      const startY = e.clientY + root.scrollTop;
      root.setPointerCapture && root.setPointerCapture(e.pointerId);
      const move = ev => { root.scrollLeft = startX - ev.clientX; root.scrollTop = startY - ev.clientY; };
      const up   = ev => { root.removeEventListener('pointermove', move); root.removeEventListener('pointerup', up); };
      root.addEventListener('pointermove', move); root.addEventListener('pointerup', up);
      e.preventDefault();
    }, true);
  }
  const applyAll = () => document.querySelectorAll('.pm-drag-scroll').forEach(el => enableDragScroll(el));
  const obs = new MutationObserver(applyAll);
  obs.observe(document.body, {subtree: true, childList: true}); applyAll();
})();
</script>
\"\"\")

from poly_mask_annotator import MaskAnnotator

ui = MaskAnnotator(
    folder=\"/path/to/images\",   # ← ここを変更
    enable_wheel_zoom=True,
    viewport_height=\"600px\",     # 未指定なら viewport_max_height=\"80vh\"（上限）
    min_zoom_pct=10, max_zoom_pct=400,
    pan_via_scroll=True,          # Pan ON 中はビューポート左ドラッグでスクロール（再描画なし）
)
ui
```

> 環境によってはノートブックで **インラインJSが禁止** されている場合があります。その場合 `pan_via_scroll` は無効化されます（通常のパンは利用可）。

---

## データと出力

- **対応拡張子**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- **サイドカー JSON**: 画像と同じ場所に `<画像名>.json` を自動保存／読込  
  - 保存内容：ラベル名、ID、色、スムージング、頂点座標、Sharp 頂点
  - JSON 内の画像サイズは実画像と一致している必要あり
- **エクスポート**:
  - `<stem>_mask.png` …… **uint8** インデックス画像（背景=0、各領域は `label_id`）
  - `<stem>_mask_color.png` …… RGB プレビュー（白／黒背景の切替）
- **重なり**: 後に作成した領域が前の領域を **上書き** します

---

## UI / 操作

### 新規ポリゴンの作成
- **＋ New region** → **左クリック**で頂点追加
- **右クリック** または **⌫ Last point** で最後の頂点を削除
- **最初の頂点付近をクリック** するか **✓ Finish** で閉じる（3点以上必要）
- **Cancel** は作図を破棄し、**作図前の状態を完全復元**（スナップショット）

### 既存ポリゴンの編集
- **Edit mode** を有効化 → **Edit** ドロップダウンから領域を選択
- **ドラッグ**で頂点移動、**Delete point** で選択頂点の削除
- **Insert point** ON → 辺上をクリックすると最近傍位置に頂点を挿入
- **Vertex: Smooth/Sharp** で頂点ごとの角保持を切替（スムージング有効時）
- 色／ラベルの変更（**Edit color** / **Edit label → Apply**）
- **Delete region** で領域全削除

### ズーム / パン
- ズーム範囲は `min_zoom_pct..max_zoom_pct` で指定
- スライダ / **Fit** / **100%** / **ホイール**（`enable_wheel_zoom=True`）
- パン：**Pan** を ON → キャンバス上をドラッグ、または  
  `pan_via_scroll=True` + 上記JS → **Pan ON 中はビューポート左ドラッグ**でスクロール（再描画なし）
- **Center** でパン座標リセット

### ファイルフィルター & ナビゲーション
- グロブ（例：`*.png` / `*_mask.jpg` / `A*.*`）、**Recursive** で下層も検索
- サイドカー JSON が見つかるとリストに **[JSON]** 表示

### Undo/Redo
- 主要な編集操作をサポート（メモリ上 ~200 ステップ）

---

## コンストラクタ（API）

```python
MaskAnnotator(
    folder: str,
    max_display: int = 1000,
    allow_color_preview_export: bool = True,
    enable_wheel_zoom: bool = False,
    edge_margin_px: int = 48,
    file_filter: str = "",
    recursive: bool = False,
    show_scrollbars: bool = True,
    viewport_max_height: str = "600px",
    viewport_height: str | None = None,  # 例: "70vh" or "600px"
    min_zoom_pct: int = 10,
    max_zoom_pct: int = 400,
    pan_via_scroll: bool = False,
)
```

- `viewport_height` を指定すると **高さ固定**（スクロール可）。未指定なら `viewport_max_height`（上限）で管理します。
- `enable_wheel_zoom=False` にすると、ノートのページスクロールを優先できます。
- `edge_margin_px` で画像周囲に余白を取り、画像外にも頂点を打てます。

---

## パフォーマンスのコツ

- **ブラウザの GPU アクセラレーション**を有効化（例：Chrome の `chrome://gpu`）
- **ipycanvas / ipywidgets を新しめ**に
- リモート実行でも **手元のブラウザ**でノートを開くと快適
- 超高解像度画像は、作業用に **ロスレスで適度に縮小** → エクスポート結果は頂点形状に依存

---

## セキュリティ / プライバシー

- 通信なし。データはローカル／クラスター内に留まります。
- JSON と PNG は **画像と同じ場所**に保存（書込権限が必要）。
- インラインJSがブロックされる環境では `pan_via_scroll` は無効化されますが、通常のパンは利用可。

---

## トラブルシューティング

- **何も表示されない** → ipywidgets / ipycanvas のバージョン、JupyterLab 4 / Notebook 7 を確認。カーネル再起動も有効。
- **右クリックが効かない**（トラックパッド設定等） → **⌫ Last point** ボタンで代替。
- **ズーム／パンが重い** → GPU 加速の確認、ipycanvas更新、Linux/x86_64なら `pillow-simd` も検討。

---

## ライセンス

MIT（`LICENSE` を参照）。

---

## 謝辞

- **ipywidgets**, **ipycanvas**
- Okabe–Ito カラーパレット
- `skimage.draw.polygon` によるポリゴン塗りつぶし
