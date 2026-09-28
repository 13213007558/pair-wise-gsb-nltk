"""
Tests for required_arcs / forbidden_arcs support in
ProbabilisticNonprojectiveParser (constrained Chu-Liu-Edmonds).

The parser is cross-checked against exhaustive enumeration of every
legal tree (single root, acyclic, constraints respected) for 4- and
5-token inputs, including negative scores and ties.  Crafted cases
cover infeasibility reporting and required arcs that must survive
cycle contraction.
"""

import itertools
import random

import pytest

from nltk.parse.nonprojectivedependencyparser import (
    DependencyScorerI,
    InfeasibleConstraintsError,
    ProbabilisticNonprojectiveParser,
)


class MatrixScorer(DependencyScorerI):
    """A scorer returning a fixed (n+1) x (n+1) matrix of arc scores."""

    def __init__(self, scores):
        self._scores = scores

    def train(self, graphs):
        pass

    def score(self, graph):
        return self._scores


def full_scores(n, values):
    """Build an (n+1)x(n+1) score matrix from a {(head, dep): score} dict."""
    return [
        [
            [values[(head, dep)]] if head != dep and dep != 0 else []
            for dep in range(n + 1)
        ]
        for head in range(n + 1)
    ]


def run_parse(n, scores, required=(), forbidden=()):
    parser = ProbabilisticNonprojectiveParser()
    parser.train([], MatrixScorer(scores))
    tokens = [f"w{i}" for i in range(1, n + 1)]
    return parser.parse_constrained(
        tokens, [None] * n, required_arcs=required, forbidden_arcs=forbidden
    )


def arcs_of(result):
    """The set of (head, dependent) arcs in a parse result's graph."""
    arcs = set()
    for head, node in result.graph.nodes.items():
        for deps in node["deps"].values():
            for dep in deps:
                arcs.add((head, dep))
    return arcs


def reaches_root(parent, start):
    node, seen = start, set()
    while node != 0:
        if node in seen:
            return False
        seen.add(node)
        node = parent[node]
    return True


def brute_force_best(n, weights, required, forbidden):
    """
    Exhaustively enumerate every legal tree (each token takes exactly one
    head, exactly one token attaches to the virtual root, acyclic, all
    required arcs present, no forbidden arc used) and return
    (score, arcs) of the best one, or None if none exists.
    """
    best = None
    for heads in itertools.product(range(n + 1), repeat=n):
        arcs = {(head, dep) for dep, head in enumerate(heads, start=1)}
        if any(head == dep for head, dep in arcs):
            continue
        if any(arc not in weights for arc in arcs):
            continue
        if any(arc in forbidden for arc in arcs):
            continue
        if not set(required) <= arcs:
            continue
        if sum(1 for head in heads if head == 0) != 1:
            continue
        parent = {dep: head for dep, head in enumerate(heads, start=1)}
        if not all(reaches_root(parent, start) for start in range(1, n + 1)):
            continue
        score = sum(weights[arc] for arc in arcs)
        if best is None or score > best[0]:
            best = (score, arcs)
    return best


def assert_legal_tree(result, n, values, required, forbidden):
    """The returned parse must itself be a legal, single-rooted tree."""
    arcs = arcs_of(result)
    assert len(arcs) == n
    assert set(required) <= arcs
    assert not (arcs & set(forbidden))
    assert sum(1 for (head, dep) in arcs if head == 0) == 1
    parent = {dep: head for head, dep in arcs}
    assert all(reaches_root(parent, start) for start in range(1, n + 1))
    assert sum(values[arc] for arc in arcs) == result.score


class TestAgainstExhaustiveEnumeration:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_scores_and_constraints(self, seed):
        rng = random.Random(seed)
        n = rng.choice([4, 5])
        # Small integer range forces negative scores and frequent ties.
        values = {
            (head, dep): rng.randint(-3, 3)
            for head in range(n + 1)
            for dep in range(1, n + 1)
            if head != dep
        }
        all_arcs = sorted(values)
        rng.shuffle(all_arcs)
        required = set(all_arcs[: rng.randint(0, 3)])
        forbidden = set(all_arcs[3 : 3 + rng.randint(0, 3)]) - required

        expected = brute_force_best(n, values, required, forbidden)
        result = run_parse(n, full_scores(n, values), required, forbidden)
        if expected is None:
            assert not result.feasible
            assert result.problems
        else:
            assert result.feasible, result.problems
            assert result.score == expected[0]
            assert_legal_tree(result, n, values, required, forbidden)
            # Deterministic: a second run yields the identical tree.
            again = run_parse(n, full_scores(n, values), required, forbidden)
            assert arcs_of(again) == arcs_of(result)

    def test_all_negative_scores(self):
        n = 4
        values = {
            (head, dep): -(head + 2 * dep) - 1
            for head in range(n + 1)
            for dep in range(1, n + 1)
            if head != dep
        }
        expected = brute_force_best(n, values, set(), set())
        result = run_parse(n, full_scores(n, values))
        assert result.feasible
        assert result.score == expected[0]
        assert result.score < 0
        assert_legal_tree(result, n, values, set(), set())


class TestTieBreaking:
    def test_ties_broken_in_stable_order(self):
        # Every arc scores 1: the smallest root child wins, and equal
        # in-arcs resolve towards the smallest head.
        n = 3
        values = {
            (head, dep): 1
            for head in range(n + 1)
            for dep in range(1, n + 1)
            if head != dep
        }
        result = run_parse(n, full_scores(n, values))
        assert result.feasible
        assert arcs_of(result) == {(0, 1), (1, 2), (1, 3)}

    def test_tie_between_root_children_prefers_smallest(self):
        # Root arcs dominate; 0->1 and 0->3 tie, so token 1 becomes the
        # root child.
        n = 3
        values = {
            (0, 1): 5, (0, 2): 1, (0, 3): 5,
            (1, 2): 0, (1, 3): 0,
            (2, 1): 0, (2, 3): 0,
            (3, 1): 0, (3, 2): 0,
        }
        result = run_parse(n, full_scores(n, values))
        assert result.feasible
        assert (0, 1) in arcs_of(result)
        assert result.score == 5


class TestCycleContraction:
    def test_required_arc_survives_contraction(self):
        # The greedy 1<->2 cycle is contracted; the required arc (1, 2)
        # inside the cycle must survive, and the best tree enters the
        # contracted component through token 3.
        n = 3
        values = {
            (0, 1): 4, (0, 2): 3, (0, 3): 10,
            (1, 2): 5, (1, 3): 2,
            (2, 1): 5, (2, 3): 2,
            (3, 1): 2, (3, 2): 2,
        }
        result = run_parse(3, full_scores(3, values), required={(1, 2)})
        assert result.feasible
        assert (1, 2) in arcs_of(result)
        expected = brute_force_best(3, values, {(1, 2)}, set())
        assert result.score == expected[0] == 17
        assert_legal_tree(result, 3, values, {(1, 2)}, set())

    def test_required_arc_survives_nested_contraction(self):
        # A 1<->2 cycle is contracted, then the supernode is absorbed
        # into a second contraction; the required arc (4, 3) must be
        # preserved through both rounds.
        n = 4
        values = {
            (head, dep): 0
            for head in range(n + 1)
            for dep in range(1, n + 1)
            if head != dep
        }
        values.update(
            {
                (0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 10,
                (1, 2): 5, (2, 1): 5,
                (3, 4): 1, (4, 3): 3,
            }
        )
        result = run_parse(4, full_scores(4, values), required={(4, 3)})
        assert result.feasible
        assert (4, 3) in arcs_of(result)
        expected = brute_force_best(4, values, {(4, 3)}, set())
        assert result.score == expected[0] == 18
        assert_legal_tree(result, 4, values, {(4, 3)}, set())

    def test_forbidden_arc_reroutes_through_contraction(self):
        # Forbidding the greedy arc into the 1<->2 cycle forces the
        # search to enter the contracted component via a weaker arc.
        n = 3
        values = {
            (0, 1): 1, (0, 2): 1, (0, 3): 6,
            (1, 2): 5, (1, 3): 1,
            (2, 1): 5, (2, 3): 1,
            (3, 1): 4, (3, 2): 3,
        }
        result = run_parse(3, full_scores(3, values), forbidden={(3, 1)})
        assert result.feasible
        assert (3, 1) not in arcs_of(result)
        expected = brute_force_best(3, values, set(), {(3, 1)})
        assert result.score == expected[0]
        assert_legal_tree(result, 3, values, set(), {(3, 1)})


class TestInfeasible:
    def setup_method(self):
        self.n = 3
        self.values = {
            (head, dep): 1
            for head in range(self.n + 1)
            for dep in range(1, self.n + 1)
            if head != dep
        }
        self.scores = full_scores(self.n, self.values)

    def test_arc_both_required_and_forbidden(self):
        result = run_parse(3, self.scores, required={(1, 2)}, forbidden={(1, 2)})
        assert not result.feasible
        assert any("both required and forbidden" in p for p in result.problems)

    def test_conflicting_required_heads(self):
        result = run_parse(3, self.scores, required={(1, 3), (2, 3)})
        assert not result.feasible
        assert any("conflicting required heads" in p for p in result.problems)

    def test_required_arcs_form_a_cycle(self):
        result = run_parse(3, self.scores, required={(1, 2), (2, 1)})
        assert not result.feasible
        assert any("cycle" in p for p in result.problems)

    def test_token_with_no_available_incoming_arc(self):
        forbidden = {(0, 2), (1, 2), (3, 2)}
        result = run_parse(3, self.scores, forbidden=forbidden)
        assert not result.feasible
        assert any("no available incoming arc" in p for p in result.problems)

    def test_single_root_unsatisfiable_when_root_arcs_forbidden(self):
        forbidden = {(0, 1), (0, 2), (0, 3)}
        result = run_parse(3, self.scores, forbidden=forbidden)
        assert not result.feasible
        assert any("single-root" in p for p in result.problems)

    def test_single_root_unsatisfiable_with_two_required_root_arcs(self):
        result = run_parse(3, self.scores, required={(0, 1), (0, 2)})
        assert not result.feasible
        assert any("single-root" in p for p in result.problems)

    def test_required_arc_without_a_score(self):
        scores = full_scores(3, self.values)
        scores[2][1] = []  # arc 2 -> 1 is not a candidate
        result = run_parse(3, scores, required={(2, 1)})
        assert not result.feasible
        assert any("not a candidate" in p for p in result.problems)

    def test_required_arc_out_of_range(self):
        result = run_parse(3, self.scores, required={(1, 9)})
        assert not result.feasible
        assert any("out of range" in p for p in result.problems)

    def test_required_arc_extension_impossible(self):
        # Token 2 must take 3 as its head, but token 3 can only be
        # headed by 2: the required arc cannot be extended to a tree.
        scores = full_scores(3, self.values)
        for head in range(4):
            if head != 2:
                scores[head][3] = []
        for head in range(4):
            if head != 3:
                scores[head][2] = []
        result = run_parse(3, scores, required={(3, 2)})
        assert not result.feasible
        assert result.problems


class TestParseGeneratorAPI:
    def test_parse_yields_single_graph(self):
        n = 3
        values = {
            (head, dep): 1
            for head in range(n + 1)
            for dep in range(1, n + 1)
            if head != dep
        }
        parser = ProbabilisticNonprojectiveParser()
        parser.train([], MatrixScorer(full_scores(n, values)))
        tokens = ["w1", "w2", "w3"]
        parses = list(parser.parse(tokens, [None] * n, required_arcs={(1, 2)}))
        assert len(parses) == 1
        graph = parses[0]
        assert 2 in graph.nodes[1]["deps"]["NTOP"]
        # Exactly one token attaches to the virtual root.
        root_deps = graph.nodes[0]["deps"]
        assert sum(len(root_deps[rel]) for rel in root_deps) == 1

    def test_parse_raises_with_explanation_on_infeasible(self):
        n = 3
        values = {
            (head, dep): 1
            for head in range(n + 1)
            for dep in range(1, n + 1)
            if head != dep
        }
        parser = ProbabilisticNonprojectiveParser()
        parser.train([], MatrixScorer(full_scores(n, values)))
        with pytest.raises(InfeasibleConstraintsError) as excinfo:
            list(
                parser.parse(
                    ["w1", "w2", "w3"], [None] * n, required_arcs={(1, 2), (2, 1)}
                )
            )
        assert any("cycle" in problem for problem in excinfo.value.problems)
