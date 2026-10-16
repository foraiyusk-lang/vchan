# HANDOFF — reimu_ai_vc(RVC による「霊夢」風声質変換)

元セッション: ローカルの「japanese clarification request」(PID 26580)。クラウドセッションで続けるためのメモ。

## 目的
RVC(torch 2.7.1+cu118)で、AquesTalk f1 等で生成した音声から「霊夢」の声質変換モデルを学習し、
リアルタイム変換アプリ(ピッチ統計補正つき)と評価(CER)まで作る。

## 構成(このリポジトリに入っているもの)
- `scripts/` — 環境構築・音声生成・学習制御・推論・評価のスクリプト
  - `install_env.sh`, `download_assets.sh` : 環境と事前学習モデルの導入
  - `make_koe_corpus.py`, `select_dataset.py` : コーパス作成・学習データ選定
  - `run_training.py`, `train_supervisor.sh`, `start_training_detached.ps1`, `kill_train.ps1` : 学習の実行・監視
  - `convert_offline.py`, `reimu_rtrvc.py`, `reimu_realtime.py` : 推論(オフライン/リアルタイム)
  - `calibrate_pitch.py`, `pitch_detect.py`, `eval_cer.py` : ピッチ補正・評価
- `tools/AquesBatch.cs` — AquesTalk バッチ生成器のソース(DLL/exe/ライセンスは含めない)
- `corpus/` — テキストコーパス
- `requirements_cu118_official.txt`, `README.md`

## 入っていないもの(ローカルPCにだけある)
`.venv/`(約9GB), `rvc/`(RVC 本体・学習ログ・モデル, 約8GB), `data/`(生成音声, 約700MB), `AquesTalk.dll` 等。
再構築は `scripts/install_env.sh` → `scripts/download_assets.sh`。学習済みモデル(G/D .pth, .index)はローカル `rvc/logs/reimu_v1/`。

## クラウド環境でできること / できないこと
- できる: スクリプトの読解・修正・レビュー、評価コードの改善。
- できない: GPU 学習・音声の実行確認(GPU/モデル/音声ファイルが無い)。実行確認はローカルPCで。

## 進捗(最後に確認できた時点)
- 完了: 環境構築、AquesTalk バッチ生成基盤、学習データの大量生成。
- 進行中: RVC 学習(前処理・F0/特徴抽出・学習・index 作成)。
- 未着手: 推論/リアルタイム変換アプリの仕上げと評価。
