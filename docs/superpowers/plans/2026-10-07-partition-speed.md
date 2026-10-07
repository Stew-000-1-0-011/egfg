# egfg フェーズ L 実装計画：分割の探索を速くする

**Spec:** `docs/superpowers/specs/2026-10-07-partition-speed-design.md`

## 仕様で決めきれていなかった点の扱い

1. **解き方を 1 つの関数に**：局所問題を解く処理（飽和、抽出、種との比較）を、モジュールの最上位の関数にする。並列なしでも、ワーカーでも、同じ関数を使う。抽出の設定（抽出法、時間、call_overhead、shape_penalty）は、関数ではなく値で渡す（プロセスに送れるように）。
2. **並列の窓**：候補を W 個ずつ見る。窓の中で、まだ解いていない局所問題を重複なしに集めて同時に解き、その後で窓の候補を順に評価する。採用があれば、その時点で窓を捨てて、新しい状態の候補の先頭から始める。
3. **木の絞り込み**：合併の段階の時間（全体の半分）を、ラウンドごとに生き残った木で分け合う。各ラウンドでは、木ごとに採用を最大 r = 2 手に制限する。
4. **手の名前**：境目の因子に限った付け替えを `factor` とし、フェーズ K の付け替え（スコープを含む任意のクラスタへ）を `factor_any` として残す。既定の交換の手は (`merge`, `move`, `factor`)。フェーズ K の再現は (`merge`, `move`, `factor_any`, `split`)。

## File Structure

- `src/egfg/partition.py`：`solve_local`（最上位）、窓での並列評価、`trees` / `jobs` / `moves`。
- `src/egfg/pipeline.py`：`optimize(..., partition_trees="halving", partition_jobs=1, partition_moves=None)`。
- `tests/test_partition.py`：並列と並列なしの一致（1 件）。
- `experiments/partition_speed.py`、`experiments/PARTITION_SPEED_REPORT.md`、`results/partition_speed.csv`。
