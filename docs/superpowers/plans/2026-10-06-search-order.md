# egfg フェーズ E 実装計画：探索の順番

**Spec:** `docs/superpowers/specs/2026-10-06-search-order-design.md`

## Global Constraints

- 既定の `saturate`（幅優先）の動作と結果は変えない。戦略は新しいモジュール `search.py` に置く。

## 仕様で決めきれていなかった点の扱い

1. **規則の組分け**：`rule_groups(rules)` で、`_rules` と同じ規則を scope・push・reorder・pull に分ける（規則そのものは変えない）。
2. **staged の押し込み**：仕様では押し込み（push）を飽和させるとしていたが、すでに並べ替えた形が多いと押し込みだけで e-graph が爆発的に大きくなった（ノード上限 2 万で 45 万ノード）。そこで 1 ラウンドあたり push を数回（既定 3 回）だけ適用し、その都度スコープの解析を飽和させる。ノード上限は、ラウンドの終わりだけでなく各段階のあとで確かめる。
3. **backoff の実行の仕方**：egglog のバックオフの状態は 1 回のスケジュールの中でしか続かない（`eg.run` を 1 ステップずつ呼ぶと毎回リセットされ、当てはまりの多い規則がずっと止まったままになる）。そこで数ステップ（既定 8）を 1 つのスケジュールとして実行し、その合間にノード上限を確かめる。規則がすべて休んでいて変化がない区切りもあるので、変化のない区切りが 2 回続いたら止める。
4. **複数の種**：`EGraphData.seed_sets` に種ごとの選択を持たせ、抽出の出発点は「木抽出」「全部の種を合わせたもの」「各種」のうち最も安いものにする。これで「seeds の結果 ≤ best-JT」が成り立つ。
5. **restart の結果**：ラウンドの中で見つけた最安の計算を覚えておき、最後の抽出がそれより悪ければそれを返す。各ラウンドは前の最安の計算を種にするので、コストは増えない。
6. **best-JT+**：各クリークで、隣から届くメッセージの積を前と後ろからの部分積で作る版と、普通の版の両方を、すべての候補の順序で試し、最も安いものを選ぶ（best-JT 以下になる）。

## File Structure

- `src/egfg/egraph.py`：`build_egraph`、`rule_groups`、複数の種（`seed_sets`）。
- `src/egfg/extract.py`：出発点の候補に各種を加える。
- `src/egfg/jtree.py`：`elimination_order`（min-fill・min-degree・min-weight、同点のランダムな破り方）、`TreeTerms(share_products=True)`。
- `src/egfg/baselines.py`：`candidate_orders`、`best_junction_tree`。
- `src/egfg/search.py`：戦略、途中の記録、`extract_result`。
- `experiments/search_order.py`、`experiments/SEARCH_ORDER_REPORT.md`、`results/search_order.csv`、`results/search_order_curve.csv`。
- `tests/test_search.py`、`tests/test_search_experiment.py`。
