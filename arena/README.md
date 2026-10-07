# arena：推定プログラムの対戦場

egfg が作る推定プログラムと、LLM が問題ごとに書いた推定プログラムを、同じ条件で比べる。仕様：`docs/superpowers/specs/2026-10-06-arena-design.md`。

## 構成

| 場所 | 中身 |
|---|---|
| `arena/problems/*.json` | 公開の問題（構造と表の作り方だけ。表の値はない） |
| `arena/solutions/egfg_c/` | egfg が生成した C のプログラム（`arena/solve_egfg_c.py`） |
| `arena/solutions/egfg/` | egfg が生成した Python のプログラム（以前の約束、`arena/solve_egfg.py`） |
| `arena/solutions/llm_c/` | 書く側の LLM が書いた C のプログラム |
| `arena/solutions/llm/` | 書く側の LLM が書いた Python のプログラム（以前の約束） |
| `arena/results/` | 採点結果 |
| `arena/rounds/` | ラウンドごとの記録 |

## プログラムの約束（C）

対戦は C で行う（numpy の呼び出しの手間に時間が左右されないようにするため）。各問題に `<名前>.c` を用意する。

```c
// tables[i]: 因子 i の表（倍精度、因子の scope の順の行優先）
// out: 周辺分布を、変数名の順（文字列としての並び）に連結して書き込む（それぞれ和が 1）
void infer(const double *const *tables, double *out);
```

- 採点では、両側とも `gcc -O3 -march=native -std=c11 -ffp-contract=fast -shared -fPIC` でコンパイルする（積和の融合は両側で有効）。GCC のベクトル拡張は使ってよい。
- 使えるヘッダは math.h、string.h、stddef.h、stdint.h、stdlib.h、float.h だけ。スレッド（OpenMP、pthread）とインラインアセンブリは禁止。
- 静的な配列（作業領域）は使ってよい。表の値に依存する計算はすべて `infer` の中で行う。
- 時間は、C の中のループで何千回も呼んだ平均を 7 組とった中央値（1 回あたり）。Python から C への呼び出しの手間は、多数回の呼び出しで薄める。

## 以前の約束（Python、参考）

```python
def infer(tables: dict[int, numpy.ndarray]) -> dict[str, numpy.ndarray]:
    # tables: 因子の番号 -> 表（軸は因子の scope の順）
    # 戻り値: 変数名 -> 正規化した周辺分布（和が 1）
```

- import できるのは numpy、math、itertools、functools、collections、operator だけ（採点の前に検査する）。
- 問題の JSON（構造と表の性質。例：`lowrank` なら rank）を見て書いてよい。表の値は実行時にしか渡されない。
- 前処理（分解など）も `infer` の中で行う（時間に含まれる）。

## 計算の結果の保存（任意）

環境変数 `EGFG_CACHE` にディレクトリを指定すると、生成した C の共有ライブラリ（ソースのハッシュごと）と e-graph の飽和の結果（構造・設定・egfg のコードの版ごと）を保存して使い回す（`src/egfg/cache.py`）。種類ごとに最近の 200 件（`EGFG_CACHE_MAX`）だけ残す。既定では無効（テストは常に計算する）。

## 採点

`uv run python arena/score.py --solutions arena/solutions/egfg_c arena/solutions/llm_c --out arena/results/<名前>.json`

- 正しさ：採点のたびに選ぶ乱数の種で表を 3 組作り、参照の周辺分布との最大誤差が 1e-6 以下。
- 速さ：温めの 1 回のあと 7 回測った中央値。
- 誤りのある側は、その問題で負け。

## egfg の改善の判定（取り置き）

`uv run python arena/holdout_gate.py --seed <種> --out arena/results/<名前>.json`

- 公開されていない問題（同じ種類、より広い大きさの範囲、種から生成）で、egfg の演算回数を best-JT+ と比べた比の幾何平均と、生成したプログラムの正しさ。
- egfg の改善は、既存のテストがすべて通り、この値が悪くならない（かつ全問正しい）ときだけ採用する。

## 役割と決まり

- **書く側（LLM）**：公開の問題ごとに、できるだけ速く正しいプログラムを書く。egfg のソース（`src/egfg/`）と egfg の解答（`arena/solutions/egfg/`）は読まない。
- **改善する側（LLM）**：採点結果と書く側のプログラムを読み、egfg（規則、表現、探索、コード生成）を改善する。取り置きの問題の生成の中身や乱数の種に合わせ込まない。足した規則は、乱数での評価と総当たりとの一致で確かめる。

## 既存のライブラリとの比較

- Python から呼ぶ（正しさの確認と目安）：`uv run --group libs python arena/score_libs.py --seed <種> --against arena/results/<score.py の結果>.json --out arena/results/libs_seed<種>.json`。GTSAM（`DiscreteFactorGraph` と `DiscreteMarginals`）と pyAgrum（Shafer-Shenoy）。グラフは一度だけ作り、全変数の周辺分布を求める時間を測る。値にはバインディングの呼び出しの手間が含まれる。
- C++ の GTSAM（公平な時間）：`uv run python arena/gtsam_bench/run.py --gtsam <GTSAM のインストール先> --seed <種> --against ... --out arena/results/gtsam_cpp_seed<種>.json`。GTSAM は自分でビルドする（pip の wheel にはヘッダがない）。公開の問題に加えて、大きめの問題（鎖 32 変数 K=32 など）でも、egfg の C と比べる。
- 結果のまとめ：`arena/rounds/libs.md`。
- 線形ガウスのフィルタ（C）：`uv run python arena/gaussian_bench/run.py --gtsam <GTSAM のインストール先> --out arena/results/gaussian_c.json`。egfg の計算を C に書き出したもの（`src/egfg/gccodegen.py`）を、同じコード生成のカルマンフィルタ・情報フィルタと、GTSAM（`KalmanFilter`、因子グラフの消去）と比べる。まとめ：`arena/rounds/gaussian_c.md`。dynamax（JAX）も比べる（`--group libs` を付ける）。
- 縮約順序の既存ツールと Funsor：`uv run --group libs python arena/compare_existing.py --seed <種> --out arena/results/existing_seed<種>.json`。opt_einsum と cotengra の計画を egfg の項に直し、同じコストモデルと C のコード生成で比べる。Python から使ったときの時間（egfg の numpy のプログラム、opt_einsum、Funsor）も測る。まとめ：`arena/rounds/existing.md`。
