# egfg フェーズ B 実装計画：時刻の添字を持つテンプレート（フィルタリング）

**Goal:** 同じ形が時刻ごとに繰り返すモデルで、1 時刻分の計算だけを e-graph で探索し、それを全時刻に当てはめてフィルタリングする。

**Spec:** `docs/superpowers/specs/2026-10-06-time-template-design.md`

**Architecture:** 動的モデル（`dynamic.py`）から、1 時刻分の局所問題（先頭・途中の 2 種類）を普通の `FactorGraph` として作る。変数の名前は時刻の差（`a@-1`、`a@0`）で書く。前の時刻からのメッセージは入力の葉 `fwd`。局所問題をフェーズ A の `saturate`／抽出で一度だけ解き、テンプレート（入力の葉を含む `Dag`）にする。評価は、入力の葉に前の時刻のメッセージの表を渡して時刻順に行う。

## Global Constraints

- フェーズ 1・A の公開 API と結果を変えない。既存のテストは変更しない。
- 実行・依存追加は uv。テストは `uv run pytest`。

## 仕様で決めきれていなかった点の扱い

1. **遷移の表は全時刻で共通**、観測の表だけが時刻ごとに変わる（`DynamicModel` が初期・遷移の表を持ち、観測の表は時刻ごとに渡す）。
2. **前向きアルゴリズムの形**：教科書どおり「予測してから更新」にする。`fwd` と遷移の因子を掛けて時刻 t−1 の変数で和をとり（予測）、その結果に観測の因子を掛ける（更新）。メッセージと各周辺分布は、更新後の表から和をとって作る（同じ表を共有）。仕様の「すべて掛けてから和をとる」より安く、比較の相手として公平。
3. **種（ジャンクションツリー）の作り方**：`fwd` を仮の因子とみなした局所問題のジャンクションツリーを作り、項の中の仮の因子を入力の葉に置き換える。前向きのメッセージは境界の変数すべての同時分布なので、境界の変数をまとめた仮の因子（表は使わない、どのクリークにも割り当てない）をクリーク作りのときだけ加え、それを含むクリークでメッセージを作る（`TreeTerms.joint` を追加）。
4. **入力の葉の評価**：`evaluate` に、入力の葉の値を渡す引数 `inputs`（名前 → 表）を加える。渡されない入力の葉は、これまでどおりエラーにする。
5. **割り算**：前向きのメッセージを、成分 0（期待値半環では p）の和で割る。
6. **大きな T の正しさの確認**：展開した因子グラフの変数が 12 個以下なら総当たり、64 個以下なら時刻順の消去順序を与えたジャンクションツリー。それより大きいと、今のジャンクションツリーの実装（min-fill とクリーク木の作成が変数の数の 2 乗以上）では遅すぎるので、結合した状態の遷移行列を numpy で作って直接計算する前向き計算（egfg の評価器を使わない、独立した参照）と比べる。
7. **最頻値**：その時刻の各状態変数の max 周辺（その時刻までの観測のもとで、ほかの変数をすべて最大化した値）を求め、総当たりと定数倍まで一致することを確かめる。
8. **分割（D）**：ステップの中の分割は、今回は実装しない（実験の設定でも使わない）。R、S、X だけをテンプレートの探索に渡す。

---

## File Structure

- `src/egfg/dynamic.py`（新規）：`DynamicModel`、`unroll`、局所問題の構築、テンプレートの探索（`compile_filter`）、前向きアルゴリズム（`forward_program`）、時刻順の評価（`Filter`、`run_filter`）、全時刻分の展開（`instantiate`）。
- `src/egfg/jtree.py`：`TreeTerms.joint`。
- `src/egfg/evaluate.py`：`evaluate(..., inputs=None)`。
- `src/egfg/generators.py`：`hmm`、`factorial_hmm`、`coupled_hmm`。
- `experiments/template.py`、`experiments/TEMPLATE_REPORT.md`、`results/template.csv`。
- `tests/test_dynamic.py`、`tests/test_template_experiment.py`。

---

### Task 1: 評価器と JT の小さな拡張

- `evaluate(dag, fg, semiring, inputs=None)`：入力の葉は `inputs[名前]` の表を使う。
- `TreeTerms.joint(keep)`：`keep` をすべて含む最小番号のクリークで、積を `keep` 以外で和をとる。
- テスト：入力の葉に表を渡すと評価でき、渡さないとエラー。

### Task 2: 動的モデルと展開

- `DynamicModel(states, initial, transition, observation)`。初期・遷移は（スコープ、表）、観測はスコープだけ。
- `unroll(model, obs)`：`obs[t][k]` は時刻 t の k 番目の観測の表。因子の番号は時刻順に、先頭は初期→観測、途中は遷移→観測。
- 生成器：HMM、因子型 HMM（m 本）、結合 HMM（m 本）と、観測の表の乱数生成。
- テスト：変数・因子の数とスコープ。T = 1。

### Task 3: 局所問題とテンプレート

- `local_problem(model, kind)`：仮の表を持つ局所 `FactorGraph`、入力の葉のスコープ、クエリ（メッセージ `msg` と各状態変数の周辺分布）、種。
- `compile_filter(model, rules, seed, extractor, ...) -> FilterProgram`：先頭と途中を飽和・抽出し、テンプレート（`Dag`、コスト、探索の記録）を持つ。
- `forward_program(model) -> FilterProgram`：前向きアルゴリズムを同じ形で作る。
- テスト：途中の局所問題はどの時刻でも同じ項（時刻の差で書くので、作り方が t によらないことを確かめる）。

### Task 4: 評価と展開

- `Filter(program, model, semiring_for)`、`step(obs_tables)`：1 時刻進めて、その時刻の各状態変数の表を返す。
- `run_filter(...)`：全時刻を順に評価。`filter_marginals`、`filter_max_marginals`、`filter_moments`。
- `instantiate(program, model, T) -> Dag`：展開した因子グラフの上の 1 つの `Dag`（割り算なし）。
- テスト：仕様 6 章のフィルタリング・逐次評価・割り算・コスト・半環。

### Task 5: 実験とレポート

- `experiments/template.py`：モデル × 設定ごとに子プロセスで一度だけテンプレートを探索し（120 秒で打ち切り）、各 T で評価・コスト・正しさを記録する。`--quick` は数十秒。
- `results/template.csv` と `experiments/TEMPLATE_REPORT.md`（表のみ、図なし）。
