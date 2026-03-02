# セットアップ・学習手順

## 1. 必要なライブラリ

### uv のインストール

[uv](https://docs.astral.sh/uv/) が未導入の場合は先にインストールしてください。

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 仮想環境の作成とライブラリのインストール

```bash
# プロジェクトディレクトリで仮想環境を作成
uv venv

# ライブラリをインストール
uv pip install torch argparse python-dotenv transformers numpy pandas torchmetrics peft bitsandbytes
```

### ライブラリ一覧

| ライブラリ | 用途 |
|---|---|
| `torch` | PyTorch：モデル定義・学習・推論の基盤 |
| `transformers` | HuggingFace Transformers：SciBERT / LLM2Vec 等の事前学習済みモデルのロード |
| `peft` | Parameter-Efficient Fine-Tuning（LoRA等）：LLM2Vec系モデルの効率的なファインチューニング |
| `bitsandbytes` | LLM2Vec系モデルの量子化（4bit / 8bit）対応 |
| `numpy` | 数値計算・配列操作 |
| `pandas` | データ読み込み・整形 |
| `torchmetrics` | F1スコア等の評価指標の計算 |
| `python-dotenv` | `.env` ファイルから環境変数を読み込む |
| `argparse` | コマンドライン引数のパース（Python標準ライブラリ） |

---

## 2. 環境変数の設定

`FineCite/` プロジェクトディレクトリ直下に `.env` ファイルを作成し、以下を記述してください（パスは環境に合わせて変更）。

```
# FineCite/.env
FINECITE_PATH=/path/to/FineCite
DATA_DIR=/path/to/FineCite/data
CACHE_DIR=/path/to/FineCite/.cache
OUT_DIR=/path/to/FineCite/output
```

| 変数名 | 説明 |
|---|---|
| `FINECITE_PATH` | リポジトリのルートパス（Pythonパスに追加される） |
| `DATA_DIR` | データセットが格納されているディレクトリ |
| `CACHE_DIR` | モデルキャッシュ・データキャッシュの保存先 |
| `OUT_DIR` | 学習済みモデル・ログの出力先 |

---

## 3. 抽出モデルの学習

引用が含まれる論文テキストから、**どの部分が引用文脈（citation context）に該当するか**をトークンレベルで識別するモデルを学習します。Named Entity Recognition に近い**系列ラベリング**タスクです。学習データには `data/finecite/train.jsonl`（FineCite 独自データセット）が使用されます。

**Step 4 で必要となるため、`--save_model` を必ず指定してください。**

```bash
uv run python finecite/train_extraction.py --model_name scibert --ext_type bilstm_crf --save_model
```

### 主なオプション

| オプション | デフォルト値 | 説明 |
|---|---|---|
| `--model_name` | `scibert` | 使用する埋め込みモデル。`scibert` / `llm2vec_mistral` / `llm2vec_llama3` |
| `--ext_type` | `bilstm_crf` | デコーダの種別。`linear` / `bilstm` / `crf` / `bilstm_crf` |
| `--iob_labels` | `False` | IOBラベル形式を使用する場合に指定 |
| `--batch_size` | `4` | バッチサイズ |
| `--learning_rate` | `3e-5` | ベースモデルの学習率 |
| `--crf_learning_rate` | `0.005` | CRF層の学習率（`crf` / `bilstm_crf` 使用時） |
| `--dropout` | `0.1` | ドロップアウト率 |
| `--save_model` | `False` | 学習後にモデルを保存する場合に指定 |
| `--debug` | `False` | デバッグモード（少量データで動作確認） |
| `--debug_size` | `100` | デバッグ時に使用するサンプル数 |
| `--seed` | `4455` | 乱数シード |

### 実行例

```bash
# SciBERT + BiLSTM-CRF（デフォルト設定）
uv run python finecite/train_extraction.py --model_name scibert --ext_type bilstm_crf --save_model

# LLM2Vec-Mistral + CRF
uv run python finecite/train_extraction.py --model_name llm2vec_mistral --ext_type crf --save_model

# デバッグモードで動作確認
uv run python finecite/train_extraction.py --model_name scibert --ext_type linear --debug --debug_size 50
```

---

## 4. 分類モデルの学習

Step 3 で学習した抽出モデルを読み込み、その抽出結果をもとに**引用の意図（citation intent）を分類**するモデルを学習します。「どこを引用しているか（抽出）」に続く「**なぜ引用しているか（分類）**」を答える2段階パイプラインの第2ステップです。

**Step 3 の抽出モデルの学習・保存（`--save_model`）が先に完了している必要があります。**

```bash
uv run python finecite/train_classification.py --model_name scibert --dataset acl-arc --cls_type weighted --save_model
```

### 主なオプション

| オプション | デフォルト値 | 説明 |
|---|---|---|
| `--model_name` | `scibert` | 使用する埋め込みモデル。`scibert` / `llm2vec_mistral` / `llm2vec_llama3` |
| `--ext_model` | `scibert` | 読み込む抽出モデルの種別 |
| `--dataset` | `acl-arc` | 使用するデータセット。`acl-arc` / `act2` / `scicite` / `multicite` |
| `--ext_type` | `bilstm_crf` | 読み込む抽出モデルのデコーダ種別 |
| `--cls_type` | `linear` | 分類ヘッドの種別。`weighted` / `balanced` / `linear` / `inf` / `perc` / `back` |
| `--batch_size` | `4` | バッチサイズ |
| `--learning_rate` | `2e-5` | ベースモデルの学習率 |
| `--crf_learning_rate` | `0.005` | CRF層の学習率 |
| `--dropout` | `0.1` | ドロップアウト率 |
| `--save_model` | `False` | 学習後にモデルを保存する場合に指定 |
| `--cached_data` | `False` | キャッシュ済み前処理データを使用する場合に指定 |
| `--debug` | `False` | デバッグモード |
| `--debug_size` | `100` | デバッグ時に使用するサンプル数 |
| `--seed` | `4455` | 乱数シード |

### 実行例

```bash
# SciBERT + acl-arc データセット（重み付き分類）
uv run python finecite/train_classification.py --model_name scibert --dataset acl-arc --cls_type weighted --save_model

# SciBERT + scicite データセット
uv run python finecite/train_classification.py --model_name scibert --dataset scicite --cls_type linear --save_model

# LLM2Vec-Mistral + multicite データセット
uv run python finecite/train_classification.py --model_name llm2vec_mistral --ext_model llm2vec_mistral --dataset multicite --cls_type weighted --save_model

# デバッグモードで動作確認
uv run python finecite/train_classification.py --model_name scibert --dataset acl-arc --debug --debug_size 50
```

---

## 5. 学習の実行順序まとめ

```
[Step 1] uv venv && uv pip install ...                    # 仮想環境の作成とライブラリのインストール
    ↓
[Step 2] .env ファイルを作成                              # 環境変数の設定
    ↓
[Step 3] uv run python finecite/train_extraction.py       # 「どこを引用しているか」の抽出モデルを学習（--save_model 必須）
    ↓
[Step 4] uv run python finecite/train_classification.py   # 「なぜ引用しているか」の分類モデルを学習（Step 3 の出力モデルを使用）
```

> **注意：** GPU（CUDA）が利用可能な環境での実行を推奨します。  
> LLM2Vec系モデル（`llm2vec_mistral` / `llm2vec_llama3`）は大幅に多くのメモリを必要とします。
