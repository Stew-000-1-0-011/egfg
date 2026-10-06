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

- 採点では、両側とも `gcc -O3 -march=native -std=c11 -shared -fPIC` でコンパイルする。
- 使えるヘッダは math.h、string.h、stddef.h、stdint.h、stdlib.h、float.h だけ。スレッド（OpenMP、pthread）とインラインアセンブリは禁止。
- 静的な配列（作業領域）は使ってよい。表の値に依存する計算はすべて `infer` の中で行う。
- 時間は、何千回も呼んだ平均を 7 組とった中央値（1 回あたり）。

## 以前の約束（Python、参考）

```python
def infer(tables: dict[int, numpy.ndarray]) -> dict[str, numpy.ndarray]:
    # tables: 因子の番号 -> 表（軸は因子の scope の順）
    # 戻り値: 変数名 -> 正規化した周辺分布（和が 1）
```

- import できるのは numpy、math、itertools、functools、collections、operator だけ（採点の前に検査する）。
- 問題の JSON（構造と表の性質。例：`lowrank` なら rank）を見て書いてよい。表の値は実行時にしか渡されない。
- 前処理（分解など）も `infer` の中で行う（時間に含まれる）。

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
