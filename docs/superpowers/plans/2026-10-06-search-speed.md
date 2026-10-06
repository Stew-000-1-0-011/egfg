# egfg フェーズ A 実装計画：探索の高速化と規模の曲線

**Goal:** 探索を軽くする 4 つのつまみ（規則セット R、分割 D、種まき S、抽出法 X）を実装し、規模を変えたときの時間・コストの曲線を測る。

**Spec:** `docs/superpowers/specs/2026-10-06-search-speed-design.md`

**Architecture:** 分割の有無にかかわらず、「クリーク木の部分木（クラスタ）ごとに局所問題を作り、飽和・抽出し、入力の葉を差し替えてつなぐ」という 1 本の経路で処理する。分割なしは「全クリークで 1 つのクラスタ」として扱い、このとき局所クエリはフェーズ 1 の `marginal_query` と同じ項になる。

## Global Constraints

- 既定値（R=`full`、D=`None`、S=偽、X=`ilp`）でフェーズ 1 の動作と結果を変えない。フェーズ 1 のテストは変更しない。
- 実行・依存追加は uv。テストは `uv run pytest`。
- 新しい公開 API はすべてキーワード引数で、既定値はフェーズ 1 と同じ。

## 仕様で決めきれていなかった点の扱い

1. **入力の葉のスコープ**：セパレータのうち、メッセージが実際に依存する変数（メッセージを作る積のスコープとセパレータの共通部分）にする。差し替え後のスコープとコストが一致するため。
2. **和をとる変数**：「積のスコープに現れる変数」のうち残すもの以外だけ和をとる。スコープに現れない変数で和をとると評価器がエラーになり、値も状態数倍にずれるため。
3. **空のメッセージ**：因子も入力もない側から来るメッセージは値 1 なので作らない（入力の葉も作らない）。
4. **不要なメッセージ**：送り先の側に周辺分布を受け持つ変数がないメッセージは作らない（コストに数えない）。
5. **クラスタの番号**：含むクリークの最小番号の順。変数の受け持ちは「番号が最小のクラスタ」なので、分割なしの JT と同じクリークで周辺分布を作る。
6. **種の項**：クリーク木上の Shafer–Shenoy の項。積の順序は「割り当てられた因子（番号順）→ 隣のクリークからのメッセージ（クリーク番号順）」で、クラスタの外の隣は入力の葉が代わりを務める。これはフェーズ 1 の `junction_tree_dag` と同じ形なので、`junction_tree_dag` もこの部品で作り直す（コストが変わらないことをテストで確認）。
7. **抽出の出発点**：種まきのとき「結果のコスト ≤ 種のコスト」を保証するため、e-graph は種の項のノードも書き出す（各 e-class で高さが最小の種のノード。子の高さは必ず小さいので循環しない）。木抽出の選択を種のノードで上書きしたものと木抽出のうち、DAG コストの安いほうを出発点にする。ILP は出発点を初期解と打ち切り時の代替に使い、貪欲 DAG 抽出は出発点から改善を始める。種がなければ出発点は木抽出そのもの（フェーズ 1 と同じ）。
8. **壁の判定**：時間上限（120 秒）の打ち切り、またはどれかのクラスタで e-graph がノード上限に達したこと。
9. **実験の並列実行**：全体を 1〜2 時間に収めるため、「設定 × 種類」の系列を 3 プロセスで並列に回す（各実行は別プロセスなので、時間の測定はほぼ独立）。

---

## File Structure

- `src/egfg/ir.py`：`Input` 項、`ENode(op="input")`、`Dag.inputs`（入力の葉の名前 → スコープ）。
- `src/egfg/cost.py`：入力の葉のコストは 0。
- `src/egfg/evaluate.py`：入力の葉が残る `Dag` はエラー。
- `src/egfg/egraph.py`：`T.input`、規則セット、入力の葉のスコープ、種の union、種のノードの書き出し。
- `src/egfg/jtree.py`（新規）：min-fill 順序、クリーク、クリーク木、因子の割り当て、クリーク木上のメッセージの項。
- `src/egfg/baselines.py`：`junction_tree_dag` を `jtree` で作り直す。`min_fill_order` は `jtree` から再公開。
- `src/egfg/extract.py`：入力の葉のスコープ、出発点の選択、貪欲 DAG 抽出 `extract_dag_greedy`。
- `src/egfg/decompose.py`（新規）：クラスタの併合、局所問題の構築、つなぎ合わせ。
- `src/egfg/pipeline.py`：`optimize` につまみを追加。
- `src/egfg/generators.py`：`random_sparse`。
- `experiments/scaling.py`、`experiments/SCALING_REPORT.md`、`results/scaling.csv`。
- `tests/test_phase_a.py`：フェーズ A のテスト。

---

### Task 1: 入力の葉（ir / cost / evaluate）

- `@dataclass(frozen=True) class Input: name: str`。`Term = Leaf | Mul | Sum | Input`。
- `term_scope(t, fg, inputs=None)`、`to_dag(terms, inputs=None)`（`Dag.inputs` に入れる）。
- `Dag` に `inputs: dict[str, frozenset[str]] = {}`。`dag_scopes` は入力の葉のスコープをそこから取る。
- `node_cost`：`input` は 0。
- `evaluate`：到達可能な入力の葉があれば `ValueError`。
- テスト：入力の葉を含む `Dag` のスコープとコスト、評価がエラーになること。

### Task 2: 規則セットと種（egraph）

- `saturate(fg, queries, max_iters=30, node_limit=50_000, rules="full", inputs=None, seeds=None)`。
- `rules` は `full` / `no_reverse` / `minimal`（未知の名前は `ValueError`）。
- `inputs` の各名前について `T.input(name)` のスコープを設定する。
- `seeds`（クエリ名 → 項）があれば `union(クエリ).with_(種)` を登録する。
- `EGraphData` に `inputs` と `seed`（e-class → 種のノード）を加える。
- テスト：どの規則セットでも抽出結果が総当たりと一致（小さなランダムグラフ）。`minimal` のノード数 ≤ `full`。

### Task 3: ジャンクションツリーの部品（jtree）

- `junction_tree(fg) -> JunctionTree(cliques, nbrs, assigned)`。
- `TreeTerms(jt, members, external, scope_of)`：`message(i, j)`（i→j。外の隣は `external[(k, i)]` の入力の葉）と `marginal(v)`。結果は `(項, スコープ, 因子の集合)`、空なら `None`。
- `baselines.junction_tree_dag` をこれで作り直す。フェーズ 1 のテスト（コスト 60 など）がそのまま通ること。

### Task 4: 抽出（extract）

- `class_scopes` が入力の葉を扱う。
- `_start(g, fg)`：木抽出と、種で上書きした選択の安いほう（注意点 7）。
- `extract_dag_ilp` の初期解・代替を `_start` にする。
- `extract_dag_greedy(g, fg, time_limit_s=60)`：出発点から、根から到達できる各 e-class のノードを他の候補に替えたときの DAG コストを計算し、循環がなく下がるなら替える。改善がなくなるか時間切れまで繰り返す。
- テスト：循環がない、コスト ≤ 木抽出、値が総当たりと一致。

### Task 5: 分割とつなぎ合わせ（decompose）

- `clusters(jt, budget) -> list[frozenset[int]]`：隣接するクラスタの組を、併合後の変数の数が小さい順に（同点はクリーク番号順）、予算以下なら併合。`None` は全クリークで 1 つ。
- `local_problems(fg, jt, clusters, seed) -> list[LocalProblem]`：各クラスタの入力の葉のスコープ、局所クエリ（素朴な形：因子を番号順、入力を名前順に左から掛け、和は名前順で最初の変数が最も外側）、種の項。
- `stitch(problems, extractions) -> Dag`：ノード名にクラスタの番号を付け、入力の葉を送り元の根に差し替える。入力の葉が残れば `ValueError`。
- テスト：予算 3・5・8 で、小さな木・輪・格子の周辺分布が総当たりと一致。入力の葉が残らない。予算を十分大きくすると、局所クエリがフェーズ 1 のクエリと同じになる。

### Task 6: 公開 API（pipeline）

- `optimize(fg, extractor="ilp", max_iters=30, node_limit=50_000, rules="full", cluster_budget=None, seed=False, time_limit_s=60)`。
- `OptimizeResult(saturations, extraction, clusters)`。`saturation` プロパティ（クラスタが 1 つのときだけ）、`hit_limit`、`num_nodes`、`max_cluster_nodes`、`saturate_s`、`extract_s`。
- テスト：種まき＋極端に小さいノード上限で、コスト ≤ JT（分割なし、分割あり）。組み合わせ設定で値が総当たりと一致。

### Task 7: 実験（experiments/scaling.py）

- `random_sparse(n, K, seed)`：ランダムな木に n/4 本の辺（重複・自己ループなし）を足す。
- 問題・設定・壁の扱い・記録する値は仕様 4 章のとおり。`--quick` は数十秒で終わる小さな版。
- 1 回の実行は子プロセス（`--worker`）で行い、120 秒で打ち切る。
- 正しさ：n ≤ 12 は総当たり、それ以上は JT の周辺分布と比較。
- テスト：`--quick` が完走し、CSV の列がそろう。

### Task 8: 本実験とレポート

- `uv run python experiments/scaling.py --out results/scaling.csv` を実行し、`experiments/SCALING_REPORT.md` を書く（表のみ、図なし）。
