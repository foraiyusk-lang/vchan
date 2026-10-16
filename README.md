# reimu_ai_vc — 霊夢(bouyomichan 女性1)AIボイスチェンジャー

自分の声 → 霊夢の声 に変換する **学習型(RVC)ボイスチェンジャー**。
ターゲット声は bouyomichan 同梱の AquesTalk「女性1(f1)」で大量生成した音声。
(旧DSP版 `../reimu_voice_changer` とは別プロジェクト)

## 構成
```
tools/AquesBatch.cs|exe   AquesTalk f1 を1プロセスで数千文まとめて合成(x86 DLL を P/Invoke)
corpus/                   Tatoeba日本語文 + ITAコーパス → koe_items.tsv(AquesTalk記法)
scripts/make_koe_corpus.py  fugashi+UniDic の発音形で読み化(は→ワ, 長音→ー など)
scripts/select_dataset.py   生成済み11h から学習用サブセットを選ぶ
scripts/run_training.py     前処理→F0(rmvpe)→HuBERT特徴→学習→index を一括実行
scripts/convert_offline.py  録音ファイルをオフライン変換(霊夢ピッチ補正つき)
scripts/reimu_realtime.py   リアルタイム変換GUI(霊夢ピッチ補正つき)
scripts/calibrate_pitch.py  自分の声の高さを測定(reimu_user_pitch.json)
rvc/                        RVC-Project/Retrieval-based-Voice-Conversion-WebUI (2026-08版, 局所改変あり)
.venv/                      Python 3.12 + torch 2.7.1+cu118
```

## 手順
```powershell
# 1. 霊夢音声の大量生成 (約11時間分が15秒)
.venv\Scripts\python.exe scripts\make_koe_corpus.py
tools\AquesBatch.exe corpus\koe_items.tsv data\wav_8k logs\gen_errors.tsv
# 2. 学習サブセット選択 → 学習 (GTX1650 4GB: 約0.6秒/ステップ)
.venv\Scripts\python.exe scripts\select_dataset.py v1 --short 1100 --para 100
.venv\Scripts\python.exe scripts\run_training.py reimu_v1 data\train_v1 --epochs 40 --save-every 2
# 3. 変換
.venv\Scripts\python.exe scripts\convert_offline.py rvc\assets\weights\<model>.pth in.wav out.wav
.venv\Scripts\python.exe scripts\calibrate_pitch.py
.venv\Scripts\python.exe scripts\reimu_realtime.py
```

## GTX 1650 (4GB) で分かったこと
- RVCは16系GPUで自動的にfp32になり、既定のままだと **2.1秒/ステップ**。
  `cudnn.benchmark=True` にするだけで **0.55〜0.7秒/ステップ** (3〜4倍)。
  → `rvc/train/train.py` に環境変数 `REIMU_CUDNN_BENCH` / `REIMU_FORCE_HALF` を追加(局所改変)。
  fp16 強制は 1.0〜1.3秒/ステップで fp32+benchmark に劣る。
- バッチサイズは **1が最速**(bs=2〜4はVRAM逼迫で1サンプル約2秒)。
- `python train/xxx.py` は `-P` 付きで起動し `PYTHONPATH=rvc` を渡す
  (スクリプト置き場 `rvc/train/` が `train` パッケージを隠す循環importの回避)。

## ピッチ補正(霊夢らしさ)
本物の霊夢(f1)は中央値約276Hz・抑揚幅約1.3半音とほぼ単調。RVC標準はキー移動だけなので、
`reimu_rtrvc.py` で入力F0を「自分の中央値からの偏差×flatten」に変換して276Hz中心へ写像する
(`REIMU_FLATTEN`, 既定0.35 / 0=完全に単調 / 1=自分の抑揚を維持)。
