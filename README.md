# content-model
##概要
コンテンツモデル(content model)をPythonで実装  
無限コンテンツモデル(infinite content model)をPythonで実装
##content_model.pyの使い方(隠れマルコフモデル)
```python
# Sample code.
from content_model import CM

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
K = 10          # 隠れ変数の数
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

cm = CM("data.txt")
cm.set_param(alpha, beta, K, N, converge)
cm.learn()
cm.output_model()
```
##infinite_content_model.pyの使い方(無限隠れコンテンツモデル)
```python
# Sample code.
from infinite_content_model import ICM

alpha = 0.01    # 初期ハイパーパラメータalpha   ハイパーパラメータの値によって隠れ変数の数が変動する
beta = 0.01     # 初期ハイパーパラメータbeta    ハイパーパラメータの値によって隠れ変数の数が変動する
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

icm = ICM("data.txt")
icm.set_param(alpha, beta, N, converge)
icm.learn()
icm.output_model()
```
##入力フォーマット
1単語をスペースで分割した1行1文形式  
文書の先頭に#(シャープ)記号を入れなければならない
```
# 文書1
単語1 単語2 単語3 ...
単語1 単語1 単語2 ...
単語1 単語1 単語2 ...
# 文書2
単語1 単語2 単語3 ...
単語1 単語1 単語2 ...
単語1 単語1 単語2 ...
...
```
例として[Wiki.py](https://github.com/KentoW/wiki)を使用して収集した アニメのあらすじ文書をdata.txtに保存
##出力フォーマット
必要な情報は各自で抜き取って使用してください．
```
model	content_model             # モデルの種類
@parameter
corpus_file	data.txt                    # トレーニングデータのPATH
hyper_parameter_alpha	0.295146        # ハイパーパラメータalpha
hyper_parameter_beta	0.088102        # ハイパーパラメータbeta
number_of_hidden_variable	5           # 隠れ変数の数
number_of_iteration	92                  # 収束した時のイテレーション回数
@likelihood                             # 対数尤度
initial likelihood	-546.599624598
last likelihood	-511.957618247
@vocaburary                             # 学習で使用した単語v
target_word	出産
target_word	拓き
target_word	土
target_word	吉日
...
@count
trans_sum	0	2134            # 遷移分布に必要な情報(分母の数)    左の数字から順に 隠れ変数のID，その隠れ変数から遷移する単語の数     (なおID=0は初期状態を意味する)
trans_freq	0	1	488         # 遷移分布に必要な情報(分子の数)    左の数字から順に 遷移元の隠れ変数のID，遷移先の隠れ変数のID，遷移した数
trans_freq	0	2	862
trans_freq	0	3	597
trans_freq	0	4	187
trans_freq	0	5	0
trans_sum	1	19685
trans_freq	1	1	366
trans_freq	1	2	5554
trans_freq	1	3	4173
trans_freq	1	4	105
trans_freq	1	5	9487
...
word_sum	1	20123           # 単語分布に必要な情報(分母の数)    左の数字から順に 隠れ変数のID，その隠れ変数から生成される単語の数
word_freq	1	い	1051        # 単語分布に必要な情報(分子の数)    左から順に 隠れ変数のID，その隠れ変数から生成される単語，と生成された頻度
word_freq	1	人	785
word_freq	1	いる	684
word_freq	1	年	349
word_freq	1	しまう	348
word_freq	1	いく	290
...
@data
# 戦勇。
topic	5	平和 だっ た 世界 に 突如 魔物 が...
topic	5	ルキメデス 復活 の 危機 が 去っ て...
topic	5	魔王 が 倒さ れ た 後 も 多く の...
topic	5	魔王 退治 の ため に アルバ と ロス と ルキ...
# 屍鬼
topic	1	人口 1 3 0 0 人 の 小さな 村 外 場 村 外部...
# BPS バトルプログラマーシラセ
topic	2	天才 的 頭脳 で 超 人的 な 演算 処理 能力 を...
topic	2	東京 物理 大学 戦略 ネットワーク 研究 室 に...
# あらいぐまラスカル
topic	4	動物 の 大好き な 1 0 歳 の スターリング は...
topic	4	やがて 無事 育っ た ラスカル は...
```
