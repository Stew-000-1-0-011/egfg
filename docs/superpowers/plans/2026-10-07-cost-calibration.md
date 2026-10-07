# egfg フェーズ M 実装計画：生成した C に合わせたコストのモデル

**Spec:** `docs/superpowers/specs/2026-10-07-cost-calibration-design.md`

## 仕様で決めきれていなかった点の扱い

1. **入れ子への分解**：`ccodegen` の `_topo`、`_refs`、`_fusion` をそのまま使う。DAG（`Dag`）でも、抽出の途中の選択（e-class → e-node）でも同じ関数で数えられるように、「ノード、根、スコープ」を受け取る形にする。
2. **歩幅**：最内のループの変数について、被演算子の歩幅を数える。葉は因子のスコープの順、中間の表と入力は変数名の順。その変数を持たない被演算子は歩幅 0（同じ要素を読み続ける）で、飛び飛びには数えない。
3. **ループの手間**：各ループに入る回数（外側のループの回数の積）の和。
4. **キャッシュ**：中間の表の合計のバイト数が 256 KiB を超えた分（を 8 で割った要素数）。
5. **定数項**：呼び出し 1 回の固定の手間として、切片を 1 つ持つ。
6. **単位**：予測は ns の実数。抽出の内部では、整数が必要なところ（ILP）で 1000 倍して丸める。
7. **Extraction.cost**：演算回数のまま残す（他のモジュールが演算回数として使っているため）。予測時間は、抽出と分割の探索の中の比較にだけ使い、`OptimizeResult` に `model_ns` として記録する。
8. **低ランク分解**：分解の葉は普通の葉として読む（分解の計算そのものは数えない）。較正のデータには低ランクの問題を入れない。
9. **ランダムな計算**：貪欲な抽出の結果から始め、ランダムに選んだ e-class の e-node を、循環しない限りランダムに取り替える（取り替えの回数を変えて、いくつか作る）。

## File Structure

- `src/egfg/cmodel.py`（新規）：入れ子への分解、特徴、較正の読み込み、予測、抽出用の DAG 全体のコスト。
- `src/egfg/extract.py`：貪欲・木・ILP の抽出に、コストのモデルを渡せるようにする。
- `src/egfg/pipeline.py`、`src/egfg/partition.py`：`cost="flops"|"c"`。
- `experiments/cost_calibration.py`：計算を集めて測り、非負の最小二乗法で合わせ、交差検証する。`results/cost_calibration.json`、`results/cost_model.csv`。
- `experiments/cost_model_use.py`：`cost="flops"` と `"c"` で選んだ C の実行時間を比べる。`results/cost_model_use.csv`。
- `experiments/COST_MODEL_REPORT.md`。
- `tests/test_cmodel.py`：3 件。
