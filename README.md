# DEEP LEARNING

# 転移学習

transfer_learning が該当ディレクトリである

## データセットの作成

基本的に preprocess.py でデータを作成される。ここではデータのダウンロードおよびデータの加工などの前処理を行う

## データセット

各該当する学習方法ディレクトリ下にデータセットディレクトリを配置している。
※transfer_learning/dataset/
モデルを作成する際はそのディレクトリから作成される。クレンジングや、教師データの整理はここで行う

## 学習モデルの作成

create_model.py が学習モデルを作成するメインプログラムである。
これを実行することで、transfer_learning/テーマ名/result に h5 ファイルが作成される。

## 学習モデルの読み込み

output_evaluation.py で学習したモデルを実際に読み込み、画像を評価し出力を print する。

## スクリプト

### release/create_model/テーマ.sh

データセットの作成およびデータ学習モデルの作成は release/create_model/テーマ.sh を実行することで
パッケージの import および、プログラムの読み込みを全て実施する。

### release/テーマ.sh

release/create_model/テーマ.sh より作成された、もしくはすでに作成されているモデルを読み込み評価するスクリプト
