# FineCite フォルダ・ファイル構成

## ルートディレクトリ

```
FineCite/
├── README.md                        # プロジェクト概要・セットアップ手順・引用情報
├── doc/                             # ドキュメントフォルダ
│   └── structure.md                 # 本ファイル：フォルダ・ファイル構成の説明
├── data/                            # データセットフォルダ
│   └── finecite/
│       ├── train.jsonl              # 学習用データセット（JSONL形式）
│       └── test.jsonl               # テスト用データセット（JSONL形式）
└── finecite/                        # メインパッケージ
    ├── __init__.py
    ├── utils.py                     # 共通ユーティリティ（シード固定・クラス重み計算など）
    ├── train_extraction.py          # 抽出モデル学習スクリプト（コマンドライン実行用）
    ├── train_classification.py      # 分類モデル学習スクリプト（コマンドライン実行用）
    ├── train_extraction_model.ipynb # 抽出モデル学習用 Jupyter Notebook
    ├── train_classification.ipynb   # 分類モデル学習用 Jupyter Notebook
    ├── data_processing/             # データ前処理サブパッケージ
    │   ├── __init__.py
    │   ├── data_processor.py        # データ読み込み・トークナイズ・特徴量生成（CLSProcessor等）
    │   └── prompts.py               # LLM2Vec等向けプロンプト定義
    └── model/                       # モデル定義サブパッケージ
        ├── __init__.py
        ├── model.py                 # ExtractionModel（系列ラベリング）・ClassificationModel（意図分類）の定義
        ├── trainer.py               # CustomTrainer：学習・評価・ログ出力を管理するトレーナークラス
        ├── classifier.py            # 分類ヘッドの実装
        └── utils.py                 # モデルロード関数・MODEL_DESCRIPTION（対応モデル設定）の定義
```

---

## 各ファイル・フォルダの詳細説明

### ルートディレクトリ

| ファイル / フォルダ | 説明 |
|---|---|
| `README.md` | プロジェクトの概要、環境セットアップ手順、モデル学習コマンド例、ライセンス、論文引用情報を記載 |
| `doc/` | プロジェクトに関するドキュメントを格納するフォルダ |
| `data/` | データセットを格納するフォルダ |
| `finecite/` | モデル・学習・前処理コードをまとめたメインパッケージ |

---

### `data/finecite/`

| ファイル | 説明 |
|---|---|
| `train.jsonl` | 学習用データセット。引用文脈と細粒度アノテーションラベルを含む（JSONL形式） |
| `test.jsonl` | テスト・評価用データセット（JSONL形式） |

---

### `finecite/`（メインパッケージ）

| ファイル | 説明 |
|---|---|
| `__init__.py` | パッケージ初期化ファイル |
| `utils.py` | 乱数シード固定 (`set_seed`)・クラス重み計算 (`get_class_weights`) などの共通ユーティリティ関数 |
| `train_extraction.py` | 引用文脈の抽出モデルをコマンドラインから学習するスクリプト。引数でモデル種別・デコーダ種別などを指定可能 |
| `train_classification.py` | 引用意図の分類モデルをコマンドラインから学習するスクリプト。事前学習済み抽出モデルを読み込んで使用 |
| `train_extraction_model.ipynb` | 抽出モデル学習の手順をインタラクティブに実行できる Jupyter Notebook |
| `train_classification.ipynb` | 分類モデル学習の手順をインタラクティブに実行できる Jupyter Notebook |

---

### `finecite/data_processing/`

データの読み込み・前処理・特徴量化を担うサブパッケージ。

| ファイル | 説明 |
|---|---|
| `__init__.py` | `load_processor` 関数をエクスポート |
| `data_processor.py` | データ読み込み・トークナイズ・パディング・ラベル変換などの前処理クラス群。`CLSProcessor`（分類用）等を含む |
| `prompts.py` | LLM2Vec系モデルに渡すプロンプトテキストの定義 |

---

### `finecite/model/`

モデル・学習器・ユーティリティを定義するサブパッケージ。

| ファイル | 説明 |
|---|---|
| `__init__.py` | `ExtractionModel`・`ClassificationModel`・`CustomTrainer`・`load_classifier`・`load_tokenizer_embedding_model`・`MODEL_DESCRIPTION` をエクスポート |
| `model.py` | `ExtractionModel`：IOBラベルを予測する系列ラベリングモデル。`ClassificationModel`：引用意図を予測する多ラベル分類モデル |
| `trainer.py` | `CustomTrainer`：学習ループ・バリデーション・スコア計算・出力サンプルのログ保存を管理するトレーナークラス |
| `classifier.py` | 分類ヘッド（線形・重み付きなど）の実装 |
| `utils.py` | トークナイザ・埋め込みモデルのロード関数、`MODEL_DESCRIPTION`（SciBERT / LLM2Vec-Mistral / LLM2Vec-LLaMA3 の設定）の定義 |

---

## システム全体の処理フロー

```
データ (data/finecite/train.jsonl, test.jsonl)
    │
    ▼
data_processing（前処理・トークナイズ・特徴量化）
    │
    ▼
model/model.py（ExtractionModel / ClassificationModel）
    │
    ▼
model/trainer.py（CustomTrainer：学習・評価・ログ出力）
```

### 対応モデル

| モデル名 | 種別 |
|---|---|
| `scibert` | SciBERT（allenai/scibert_scivocab_uncased） |
| `llm2vec_mistral` | LLM2Vec + Mistral-7B-Instruct-v0.2 |
| `llm2vec_llama3` | LLM2Vec + LLaMA3 |

### 抽出タスクのデコーダ種別

| デコーダ | 説明 |
|---|---|
| `linear` | 線形層によるシンプルなデコーダ |
| `bilstm` | 双方向LSTMデコーダ |
| `crf` | 条件付き確率場（CRF）デコーダ |
| `bilstm_crf` | 双方向LSTM + CRF（デフォルト） |
