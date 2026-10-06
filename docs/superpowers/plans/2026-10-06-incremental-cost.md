# egfg フェーズ C 実装計画：インクリメンタルなコスト

**Goal:** 1 ステップの計算を「準備（観測の前）」と「待ち時間（観測の後）」に分け、待ち時間を重く見る目的関数で抽出する。オンラインの評価を `prepare()` と `update()` に分ける。

**Spec:** `docs/superpowers/specs/2026-10-06-incremental-cost-design.md`

## Global Constraints

- 重みを渡さなければ、フェーズ 1・A・B の結果は変わらない。既存のテストは変更しない。
- `Extraction.cost` はこれまでどおり重みなしの DAG コスト。重みつきのコストは別の値として持つ。

## 仕様で決めきれていなかった点の扱い

1. **重みは e-class ごと**：e-class に含まれる因子の集合（どのノードから数えても同じ）が、そのステップの観測の因子を含むなら `w_after`、含まなければ `w_before`。
2. **M**：先頭・途中それぞれで、前向きアルゴリズムのそのステップのコスト ＋ 1。
3. **重みつきのコストを使う場所**：木抽出の選択、出発点（木抽出と種）の比較、貪欲 DAG 抽出の改善の判定、ILP の目的関数と代替の比較。
4. **`prepare()`**：観測の表の代わりに仮の表を持つ局所の因子グラフで、観測に依存しないノードをすべて評価して覚えておく。`update()` はそれを既知の値として評価器に渡す（`evaluate` に `known` 引数を加える）。

## File Structure

- `src/egfg/extract.py`：`class_leaves`、各抽出法の `weights` 引数、`Extraction.weighted_cost`。
- `src/egfg/cost.py`：`split_cost(dag, fg, after_ids)`。
- `src/egfg/evaluate.py`：`evaluate(..., known=None)`。
- `src/egfg/dynamic.py`：テンプレートの準備・待ち時間のコスト、`compile_filter(..., objective=...)`、`Filter.prepare()`／`update()`。
- `experiments/incremental.py`、`experiments/INCREMENTAL_REPORT.md`、`results/incremental.csv`。
- `tests/test_incremental.py`。

## Tasks

1. 抽出の重みと依存の判定（テスト：同じ e-class の因子の集合が等しい、重みなしで結果が変わらない）。
2. コストの分割（テスト：HMM の手計算、準備 ＋ 待ち時間 ＝ 合計）。
3. 目的関数つきの `compile_filter`（テスト：`total` は B と同じ、`latency` の待ち時間 ≤ `total` と前向きアルゴリズム）。
4. `prepare()`／`update()`（テスト：観測なしで `prepare()` を呼べる、`step()` と一致、全目的で総当たりと一致）。
5. 実験とレポート。
