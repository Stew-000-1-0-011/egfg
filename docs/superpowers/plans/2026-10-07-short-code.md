# egfg フェーズ J 実装計画：出力コードを短くする

**Spec:** `docs/superpowers/specs/2026-10-07-short-code-design.md`

## 仕様で決めきれていなかった点の扱い

1. **呼び出しの並べ方**：トポロジカル順のままだと、同じカーネルの呼び出しが別のカーネルに挟まれて連続しない（鎖の前向きのメッセージと周辺分布が交互になる）。依存関係を守ったまま、直前と同じカーネルの呼び出しを優先して並べ直す（リスト・スケジューリング）。中間の配列の番号はこの順に振るので、連続する呼び出しの出力は連続する番号になる。
2. **プール**：中間の配列は、大きさごとに 1 つの 2 次元配列にまとめる。
3. **ループにする条件**：同じカーネルが 3 回以上続き、出力と各被演算子の番号が呼び出しの位置について 1 次式になるとき。
4. **カーネルの引数**：出力は `double *restrict`、入力は `const double *restrict`（出力と入力が同じ配列になることはない）。
5. **共通部分**：和の連続の融合と、葉（表、低ランク分解の U・V）の読み方は、今のコード生成と同じ処理を使う（関数に切り出して共有する）。

## File Structure

- `src/egfg/ccodegen.py`：`generate_c(..., compact=False)`、カーネルの形の正規化、並べ直し、ループ化。
- `src/egfg/extract.py`（または `pipeline.py`）：貪欲な抽出に「形の種類 × α」を足すオプション。
- `experiments/short_code.py`、`experiments/SHORT_CODE_REPORT.md`、`results/short_code.csv`。
- `tests/test_codegen.py` に 2 件。
