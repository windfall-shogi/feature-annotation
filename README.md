# feature-annotation
入力局面での合法手や利きの数などの特徴量を計算

## 依存するライブラリ
- tensorflow
- dm-sonnet
- python-shogi (動作確認用)

## 特徴・仕様
- 入力する局面は常に手番側から見た局面(手番側をblack, 非手番側をwhiteと呼称している)
- 入力の配列は縦型(手番が先手の時は1筋から順番, 手番が後手の時は9筋から逆順に数える)
- 合法手を計算(王手千日手、打ち歩詰めを除く)
  - 出力形式は[バッチサイズ, 139, 9, 9]のブール型の配列(データフォーマットがNCHW形式の場合)
  - 駒の移動の行動は移動元ではなく移動先で表現
  - 駒の移動のインデックスの順序は(distance - 1) * 16 + promotion * 8 + direction (0--127)
  - 桂馬の移動の順序は右に跳ねる,左に跳ねる,右に跳ねる(成),左に跳ねる(成)　(128--131)
  - 駒を打つ (132--138)
  - 駒の番号、方向の順序はソースコードを参照してください
- 駒の利きを計算
  - 各マスに利きが幾つあるかの個数を手番側,非手番側のそれぞれで計算
  - 駒の種類ごと,方向ごとはソースコードを改造してください
  - ピンや王の自殺などを考慮しない利きをnaiveと呼称　(出力される利きの個数はnaiveではない利きの個数)
  - 相手のnaiveな利きがある場合,自身の王はそのマスには利きがないとする　(駒を互いに取り合って、最後に王が残る状況でも王の利きがないとなります)
- 王手の有無を計算
  - ブール型

## テスト
- random_action.py
  - 出力の内容がpython-shogiで計算したものと一致するかを比較
  - 初期局面からランダムな差し手で局面を生成する
  - 使い方のサンプルを兼ねる
- annotation/*/test/*.py
  - 各部分のユニットテスト
  - 実行する際は、このreadme.mdがあるディレクトリをカレントディレクトリにして実行する

---
作成：井本 康宏  
リポジトリ：https://github.com/windfall-shogi/feature-annotation.git  
twitter:@Windfall_shogi 