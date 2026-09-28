# Natural Language Toolkit: Dependency Grammars
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Jason Narad <jason.narad@gmail.com>
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

import logging
import math
import time

from nltk.parse.dependencygraph import DependencyGraph

#: Default wall-clock limit, in seconds, for a single
#: :meth:`NonprojectiveDependencyParser.parse` call. The parser enumerates every
#: grammatical head-assignment through the lattice; with a dense grammar (many
#: words able to head many others) that set is exponential in the number of
#: tokens, so a short input can exhaust CPU and memory (CWE-407). When the limit
#: is exceeded, ``parse`` raises ``TimeoutError``. A wall-clock bound (rather than
#: a token cap) is used because the blow-up depends on grammar *density*, not just
#: length; ``max_time=None`` disables it.
DEFAULT_MAX_TIME = 5.0

logger = logging.getLogger(__name__)

#################################################################
# Constrained Maximum Spanning Arborescence (Chu-Liu-Edmonds)
#################################################################


class InfeasibleConstraintsError(ValueError):
    """
    Raised by :meth:`ProbabilisticNonprojectiveParser.parse` when the
    ``required_arcs``/``forbidden_arcs`` constraints, or the
    single-root condition, cannot be satisfied.  The ``problems``
    attribute lists human-readable explanations.
    """

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("infeasible arc constraints: " + "; ".join(self.problems))


class ConstrainedParseResult:
    """
    The outcome of :meth:`ProbabilisticNonprojectiveParser.parse_constrained`.

    ``feasible`` reports whether a legal analysis exists.  When it does,
    ``graph`` holds the highest-scoring legal ``DependencyGraph`` and
    ``score`` its total arc score.  When it does not, ``problems``
    lists human-readable explanations of why no legal analysis exists
    (conflicting required arcs, required arcs forming a cycle, a token
    with no available incoming arc, or an unsatisfiable single-root
    condition).
    """

    def __init__(self, feasible, graph=None, score=None, problems=()):
        self.feasible = feasible
        self.graph = graph
        self.score = score
        self.problems = list(problems)

    def __bool__(self):
        return self.feasible

    def __repr__(self):
        if self.feasible:
            return f"<ConstrainedParseResult feasible score={self.score}>"
        return f"<ConstrainedParseResult infeasible: {'; '.join(self.problems)}>"


def _node_sort_key(node):
    """
    Deterministic ordering key for node identifiers.  Cycle contraction
    represents supernodes as frozensets, so a type-aware key is needed to
    keep tie-breaking stable.
    """
    if isinstance(node, frozenset):
        return (1, tuple(sorted(_node_sort_key(member) for member in node)))
    return (0, node)


def _find_cycle(best, root):
    """
    Return a cycle in the parent map ``best`` as a list of node
    identifiers, or ``None`` if the map is acyclic.  Nodes without an
    entry in ``best``, and the ``root``, terminate the search.
    """
    color = {}
    for start in best:
        if color.get(start, 0) != 0:
            continue
        path = []
        node = start
        while node is not None and node != root and color.get(node, 0) == 0:
            color[node] = 1
            path.append(node)
            node = best.get(node)
        if node is not None and node != root and color.get(node) == 1:
            return path[path.index(node) :]
        for member in path:
            color[member] = 2
    return None


def _canonical_cycle(cycle):
    """Rotate a cycle so it starts at its smallest member (stable printing)."""
    start = cycle.index(min(cycle))
    return cycle[start:] + cycle[:start]


def _max_arborescence(node_ids, root, weights):
    """
    Chu-Liu-Edmonds maximum spanning arborescence.

    :param node_ids: node identifiers; cycle contraction represents
        supernodes as frozensets of their members.
    :param root: the designated root; it receives no incoming arc.
    :param weights: dict mapping ``(head, dependent)`` to a numeric score.
        Only arcs present in the dict are allowed, so forbidden arcs are
        simply omitted.  Scores may be negative.
    :return: a dict mapping every non-root node to its head, or ``None``
        when no spanning arborescence exists.  Ties are broken
        deterministically towards the smallest head, then the smallest
        dependent, so equal-scoring inputs always yield the same tree.
    """
    incoming = {}
    for (head, dep), weight in weights.items():
        incoming.setdefault(dep, []).append((head, weight))

    best = {}
    for dep in node_ids:
        if dep == root:
            continue
        candidates = incoming.get(dep)
        if not candidates:
            return None
        best[dep] = min(
            candidates,
            key=lambda candidate: (-candidate[1], _node_sort_key(candidate[0])),
        )[0]

    cycle = _find_cycle(best, root)
    if cycle is None:
        return best

    # Contract the cycle into a supernode and recurse.  Arcs entering the
    # cycle are discounted by the greedy in-arc they would replace; arcs
    # leaving the cycle keep their weight.  The required-arc bonus (if
    # any) travels with each arc, so fixed arcs stay fixed through
    # contraction and expansion.
    cycle_members = frozenset(cycle)
    supernode = cycle_members
    contracted_nodes = [node for node in node_ids if node not in cycle_members]
    contracted_nodes.append(supernode)

    contracted_weights = {}
    origin = {}
    for (head, dep), weight in weights.items():
        head_in = head in cycle_members
        dep_in = dep in cycle_members
        if head_in and dep_in:
            continue
        if not head_in and not dep_in:
            new_edge, new_weight, new_origin = (head, dep), weight, (head, dep)
        elif dep_in:
            new_edge = (head, supernode)
            new_weight = weight - weights[(best[dep], dep)]
            new_origin = (head, dep)
        else:
            new_edge = (supernode, dep)
            new_weight = weight
            new_origin = (head, dep)
        key = (
            -new_weight,
            _node_sort_key(new_origin[0]),
            _node_sort_key(new_origin[1]),
        )
        if new_edge not in contracted_weights or key < contracted_weights[new_edge][0]:
            contracted_weights[new_edge] = (key, new_weight)
            origin[new_edge] = new_origin

    sub = _max_arborescence(
        contracted_nodes,
        root,
        {edge: weight for edge, (key, weight) in contracted_weights.items()},
    )
    if sub is None:
        return None

    # Expand the supernode: the arc entering the cycle replaces the greedy
    # in-arc of the member it lands on; every other member keeps its
    # greedy head.
    parents = {}
    entered = set()
    for dep, head in sub.items():
        origin_head, origin_dep = origin[(head, dep)]
        parents[origin_dep] = origin_head
        if origin_dep in cycle_members and origin_head not in cycle_members:
            entered.add(origin_dep)
    for member in cycle_members:
        if member not in entered:
            parents[member] = best[member]
    return parents


def _arc_weights(scores):
    """
    Flatten a 3-dimensional ``DependencyScorerI`` score matrix into a
    dict mapping ``(head, dependent)`` to a single weight.  Cells may
    hold several alternative scores (as in k-best parsing); the best one
    is used.  Empty cells mean the arc is not a candidate.  Self-loops
    and arcs into the virtual root (column 0) are never candidates.
    """
    weights = {}
    for head, row in enumerate(scores):
        for dep, cell in enumerate(row):
            if dep == 0 or head == dep or not cell:
                continue
            weights[(head, dep)] = max(cell)
    return weights


#################################################################
# DependencyScorerI - Interface for Graph-Edge Weight Calculation
#################################################################


class DependencyScorerI:
    """
    A scorer for calculated the weights on the edges of a weighted
    dependency graph.  This is used by a
    ``ProbabilisticNonprojectiveParser`` to initialize the edge
    weights of a ``DependencyGraph``.  While typically this would be done
    by training a binary classifier, any class that can return a
    multidimensional list representation of the edge weights can
    implement this interface.  As such, it has no necessary
    fields.
    """

    def __init__(self):
        if self.__class__ == DependencyScorerI:
            raise TypeError("DependencyScorerI is an abstract interface")

    def train(self, graphs):
        """
        :type graphs: list(DependencyGraph)
        :param graphs: A list of dependency graphs to train the scorer.
            Typically the edges present in the graphs can be used as
            positive training examples, and the edges not present as negative
            examples.
        """
        raise NotImplementedError()

    def score(self, graph):
        """
        :type graph: DependencyGraph
        :param graph: A dependency graph whose set of edges need to be
            scored.
        :rtype: A three-dimensional list of numbers.
        :return: The score is returned in a multidimensional(3) list, such
            that the outer-dimension refers to the head, and the
            inner-dimension refers to the dependencies.  For instance,
            scores[0][1] would reference the list of scores corresponding to
            arcs from node 0 to node 1.  The node's 'address' field can be used
            to determine its number identification.

        For further illustration, a score list corresponding to Fig.2 of
        Keith Hall's 'K-best Spanning Tree Parsing' paper::

              scores = [[[], [5],  [1],  [1]],
                       [[], [],   [11], [4]],
                       [[], [10], [],   [5]],
                       [[], [8],  [8],  []]]

        When used in conjunction with a MaxEntClassifier, each score would
        correspond to the confidence of a particular edge being classified
        with the positive training examples.
        """
        raise NotImplementedError()


#################################################################
# NaiveBayesDependencyScorer
#################################################################


class NaiveBayesDependencyScorer(DependencyScorerI):
    """
    A dependency scorer built around a MaxEnt classifier.  In this
    particular class that classifier is a ``NaiveBayesClassifier``.
    It uses head-word, head-tag, child-word, and child-tag features
    for classification.

    >>> from nltk.parse.dependencygraph import DependencyGraph, conll_data2

    >>> graphs = [DependencyGraph(entry) for entry in conll_data2.split('\\n\\n') if entry]
    >>> npp = ProbabilisticNonprojectiveParser()
    >>> npp.train(graphs, NaiveBayesDependencyScorer())
    >>> parses = npp.parse(['Cathy', 'zag', 'hen', 'zwaaien', '.'], ['N', 'V', 'Pron', 'Adj', 'N', 'Punc'])
    >>> len(list(parses))
    1

    """

    def __init__(self):
        pass  # Do nothing without throwing error

    def train(self, graphs):
        """
        Trains a ``NaiveBayesClassifier`` using the edges present in
        graphs list as positive examples, the edges not present as
        negative examples.  Uses a feature vector of head-word,
        head-tag, child-word, and child-tag.

        :type graphs: list(DependencyGraph)
        :param graphs: A list of dependency graphs to train the scorer.
        """

        from nltk.classify import NaiveBayesClassifier

        # Create training labeled training examples
        labeled_examples = []
        for graph in graphs:
            for head_node in graph.nodes.values():
                for child_index, child_node in graph.nodes.items():
                    if child_index in head_node["deps"]:
                        label = "T"
                    else:
                        label = "F"
                    labeled_examples.append(
                        (
                            dict(
                                a=head_node["word"],
                                b=head_node["tag"],
                                c=child_node["word"],
                                d=child_node["tag"],
                            ),
                            label,
                        )
                    )

        self.classifier = NaiveBayesClassifier.train(labeled_examples)

    def score(self, graph):
        """
        Converts the graph into a feature-based representation of
        each edge, and then assigns a score to each based on the
        confidence of the classifier in assigning it to the
        positive label.  Scores are returned in a multidimensional list.

        :type graph: DependencyGraph
        :param graph: A dependency graph to score.
        :rtype: 3 dimensional list
        :return: Edge scores for the graph parameter.
        """
        # Convert graph to feature representation
        edges = []
        for head_node in graph.nodes.values():
            for child_node in graph.nodes.values():
                edges.append(
                    dict(
                        a=head_node["word"],
                        b=head_node["tag"],
                        c=child_node["word"],
                        d=child_node["tag"],
                    )
                )

        # Score edges
        edge_scores = []
        row = []
        count = 0
        for pdist in self.classifier.prob_classify_many(edges):
            logger.debug("%.4f %.4f", pdist.prob("T"), pdist.prob("F"))
            # smoothing in case the probability = 0
            row.append([math.log(pdist.prob("T") + 0.00000000001)])
            count += 1
            if count == len(graph.nodes):
                edge_scores.append(row)
                row = []
                count = 0
        return edge_scores


#################################################################
# A Scorer for Demo Purposes
#################################################################
# A short class necessary to show parsing example from paper
class DemoScorer(DependencyScorerI):
    def train(self, graphs):
        print("Training...")

    def score(self, graph):
        # scores for Keith Hall 'K-best Spanning Tree Parsing' paper
        return [
            [[], [5], [1], [1]],
            [[], [], [11], [4]],
            [[], [10], [], [5]],
            [[], [8], [8], []],
        ]


#################################################################
# Non-Projective Probabilistic Parsing
#################################################################


class ProbabilisticNonprojectiveParser:
    """A probabilistic non-projective dependency parser.

    Nonprojective dependencies allows for "crossing branches" in the parse tree
    which is necessary for representing particular linguistic phenomena, or even
    typical parses in some languages.  This parser follows the MST parsing
    algorithm, outlined in McDonald(2005), which likens the search for the best
    non-projective parse to finding the maximum spanning tree in a weighted
    directed graph.

    >>> class Scorer(DependencyScorerI):
    ...     def train(self, graphs):
    ...         pass
    ...
    ...     def score(self, graph):
    ...         return [
    ...             [[], [5],  [1],  [1]],
    ...             [[], [],   [11], [4]],
    ...             [[], [10], [],   [5]],
    ...             [[], [8],  [8],  []],
    ...         ]


    >>> npp = ProbabilisticNonprojectiveParser()
    >>> npp.train([], Scorer())

    >>> parses = npp.parse(['v1', 'v2', 'v3'], [None, None, None])
    >>> len(list(parses))
    1

    Rule based example

    >>> from nltk.grammar import DependencyGrammar

    >>> grammar = DependencyGrammar.fromstring('''
    ... 'taught' -> 'play' | 'man'
    ... 'man' -> 'the' | 'in'
    ... 'in' -> 'corner'
    ... 'corner' -> 'the'
    ... 'play' -> 'golf' | 'dachshund' | 'to'
    ... 'dachshund' -> 'his'
    ... ''')

    >>> ndp = NonprojectiveDependencyParser(grammar)
    >>> parses = ndp.parse(['the', 'man', 'in', 'the', 'corner', 'taught', 'his', 'dachshund', 'to', 'play', 'golf'])
    >>> len(list(parses))
    4

    """

    def __init__(self):
        """
        Creates a new non-projective parser.
        """
        logging.debug("initializing prob. nonprojective...")

    def train(self, graphs, dependency_scorer):
        """
        Trains a ``DependencyScorerI`` from a set of ``DependencyGraph`` objects,
        and establishes this as the parser's scorer.  This is used to
        initialize the scores on a ``DependencyGraph`` during the parsing
        procedure.

        :type graphs: list(DependencyGraph)
        :param graphs: A list of dependency graphs to train the scorer.
        :type dependency_scorer: DependencyScorerI
        :param dependency_scorer: A scorer which implements the
            ``DependencyScorerI`` interface.
        """
        self._scorer = dependency_scorer
        self._scorer.train(graphs)

    def initialize_edge_scores(self, graph):
        """
        Assigns a score to every edge in the ``DependencyGraph`` graph.
        These scores are generated via the parser's scorer which
        was assigned during the training process.

        :type graph: DependencyGraph
        :param graph: A dependency graph to assign scores to.
        """
        self.scores = self._scorer.score(graph)

    def collapse_nodes(self, new_node, cycle_path, g_graph, b_graph, c_graph):
        """
        Takes a list of nodes that have been identified to belong to a cycle,
        and collapses them into on larger node.  The arcs of all nodes in
        the graph must be updated to account for this.

        :type new_node: Node.
        :param new_node: A Node (Dictionary) to collapse the cycle nodes into.
        :type cycle_path: A list of integers.
        :param cycle_path: A list of node addresses, each of which is in the cycle.
        :type g_graph, b_graph, c_graph: DependencyGraph
        :param g_graph, b_graph, c_graph: Graphs which need to be updated.
        """
        logger.debug("Collapsing nodes...")
        # Collapse all cycle nodes into v_n+1 in G_Graph
        for cycle_node_index in cycle_path:
            g_graph.remove_by_address(cycle_node_index)
        g_graph.add_node(new_node)
        g_graph.redirect_arcs(cycle_path, new_node["address"])

    def update_edge_scores(self, new_node, cycle_path):
        """
        Updates the edge scores to reflect a collapse operation into
        new_node.

        :type new_node: A Node.
        :param new_node: The node which cycle nodes are collapsed into.
        :type cycle_path: A list of integers.
        :param cycle_path: A list of node addresses that belong to the cycle.
        """
        logger.debug("cycle %s", cycle_path)

        cycle_path = self.compute_original_indexes(cycle_path)

        logger.debug("old cycle %s", cycle_path)
        logger.debug("Prior to update: %s", self.scores)

        for i, row in enumerate(self.scores):
            for j, column in enumerate(self.scores[i]):
                logger.debug(self.scores[i][j])
                if j in cycle_path and i not in cycle_path and self.scores[i][j]:
                    subtract_val = self.compute_max_subtract_score(j, cycle_path)

                    logger.debug("%s - %s", self.scores[i][j], subtract_val)

                    new_vals = []
                    for cur_val in self.scores[i][j]:
                        new_vals.append(cur_val - subtract_val)

                    self.scores[i][j] = new_vals

        for i, row in enumerate(self.scores):
            for j, cell in enumerate(self.scores[i]):
                if i in cycle_path and j in cycle_path:
                    self.scores[i][j] = []

        logger.debug("After update: %s", self.scores)

    def compute_original_indexes(self, new_indexes):
        """
        As nodes are collapsed into others, they are replaced
        by the new node in the graph, but it's still necessary
        to keep track of what these original nodes were.  This
        takes a list of node addresses and replaces any collapsed
        node addresses with their original addresses.

        :type new_indexes: A list of integers.
        :param new_indexes: A list of node addresses to check for
            subsumed nodes.
        """
        swapped = True
        while swapped:
            originals = []
            swapped = False
            for new_index in new_indexes:
                if new_index in self.inner_nodes:
                    for old_val in self.inner_nodes[new_index]:
                        if old_val not in originals:
                            originals.append(old_val)
                            swapped = True
                else:
                    originals.append(new_index)
            new_indexes = originals
        return new_indexes

    def compute_max_subtract_score(self, column_index, cycle_indexes):
        """
        When updating scores the score of the highest-weighted incoming
        arc is subtracted upon collapse.  This returns the correct
        amount to subtract from that edge.

        :type column_index: integer.
        :param column_index: A index representing the column of incoming arcs
            to a particular node being updated
        :type cycle_indexes: A list of integers.
        :param cycle_indexes: Only arcs from cycle nodes are considered.  This
            is a list of such nodes addresses.
        """
        max_score = -100000
        for row_index in cycle_indexes:
            for subtract_val in self.scores[row_index][column_index]:
                if subtract_val > max_score:
                    max_score = subtract_val
        return max_score

    def best_incoming_arc(self, node_index):
        """
        Returns the source of the best incoming arc to the
        node with address: node_index

        :type node_index: integer.
        :param node_index: The address of the 'destination' node,
            the node that is arced to.
        """
        originals = self.compute_original_indexes([node_index])
        logger.debug("originals: %s", originals)

        max_arc = None
        max_score = None
        for row_index in range(len(self.scores)):
            for col_index in range(len(self.scores[row_index])):
                if col_index in originals and (
                    max_score is None or self.scores[row_index][col_index] > max_score
                ):
                    max_score = self.scores[row_index][col_index]
                    max_arc = row_index
                    logger.debug("%s, %s", row_index, col_index)

        logger.debug(max_score)

        for key in self.inner_nodes:
            replaced_nodes = self.inner_nodes[key]
            if max_arc in replaced_nodes:
                return key

        return max_arc

    def original_best_arc(self, node_index):
        originals = self.compute_original_indexes([node_index])
        max_arc = None
        max_score = None
        max_orig = None
        for row_index in range(len(self.scores)):
            for col_index in range(len(self.scores[row_index])):
                if col_index in originals and (
                    max_score is None or self.scores[row_index][col_index] > max_score
                ):
                    max_score = self.scores[row_index][col_index]
                    max_arc = row_index
                    max_orig = col_index
        return [max_arc, max_orig]

    def parse_constrained(self, tokens, tags, required_arcs=None, forbidden_arcs=None):
        """
        Find the highest-scoring non-projective parse that respects the
        given hard arc constraints.  The constraints are folded into the
        Chu-Liu-Edmonds maximum spanning arborescence search itself --
        replacing arcs in an unconstrained MST afterwards would break
        both acyclicity and optimality.

        The virtual root has address 0 and ``tokens[i]`` has address
        ``i + 1``; arcs are ``(head, dependent)`` address pairs.
        Exactly one token is attached to the virtual root in every
        returned parse.  Negative scores are supported.  Ties are broken
        deterministically: the smallest-address root child is preferred,
        and equal-scoring arcs are resolved towards the smallest head,
        then the smallest dependent.

        :type tokens: list(str)
        :param tokens: A list of words or punctuation to be parsed.
        :type tags: list(str)
        :param tags: A list of tags corresponding by index to the words
            in the tokens list.
        :param required_arcs: arcs that must appear in the parse.
        :type required_arcs: iter(tuple(int, int)) or None
        :param forbidden_arcs: arcs that must not appear in the parse.
        :type forbidden_arcs: iter(tuple(int, int)) or None
        :return: The parse outcome.  When no legal analysis exists, its
            ``problems`` explain why (conflicting required arcs,
            required arcs forming a cycle, a token with no available
            incoming arc, or an unsatisfiable single-root condition).
        :rtype: ConstrainedParseResult

        >>> class Scorer(DependencyScorerI):
        ...     def train(self, graphs):
        ...         pass
        ...     def score(self, graph):
        ...         return [
        ...             [[], [5],  [1],  [1]],
        ...             [[], [],   [11], [4]],
        ...             [[], [10], [],   [5]],
        ...             [[], [8],  [8],  []],
        ...         ]
        >>> npp = ProbabilisticNonprojectiveParser()
        >>> npp.train([], Scorer())
        >>> result = npp.parse_constrained(['v1', 'v2', 'v3'], [None, None, None])
        >>> result.feasible
        True
        >>> result.score
        21
        >>> result.graph.root["word"]
        'v1'

        >>> result = npp.parse_constrained(['v1', 'v2', 'v3'], [None, None, None],
        ...                                required_arcs=[(3, 2)],
        ...                                forbidden_arcs=[(1, 2)])
        >>> result.feasible, result.score
        (True, 19)

        >>> result = npp.parse_constrained(['v1', 'v2', 'v3'], [None, None, None],
        ...                                required_arcs=[(1, 2), (2, 1)])
        >>> result.feasible
        False
        >>> result.problems
        ['the required arcs form a cycle: 1 -> 2 -> 1']
        """
        tokens = list(tokens)
        n = len(tokens)
        required = {tuple(arc) for arc in (required_arcs or ())}
        forbidden = {tuple(arc) for arc in (forbidden_arcs or ())}

        graph = DependencyGraph()
        for index, token in enumerate(tokens):
            graph.nodes[index + 1].update(
                {"word": token, "tag": tags[index], "rel": "NTOP", "address": index + 1}
            )
        self.initialize_edge_scores(graph)
        weights = _arc_weights(self.scores)

        problems = []
        for head, dep in sorted(required):
            if not (0 <= head <= n and 1 <= dep <= n):
                problems.append(
                    f"required arc {(head, dep)} is out of range "
                    f"for a {n}-token input"
                )
            elif head == dep:
                problems.append(f"required arc {(head, dep)} is a self-loop")
        # Forbidding an arc that could never be chosen is harmless.
        forbidden = {
            (head, dep)
            for (head, dep) in forbidden
            if 0 <= head <= n and 1 <= dep <= n and head != dep
        }
        for arc in sorted(required & forbidden):
            problems.append(f"arc {arc} is both required and forbidden")
        required_head = {}
        for head, dep in sorted(required):
            if dep in required_head:
                problems.append(
                    f"token {dep} has conflicting required heads: "
                    f"{required_head[dep]} and {head}"
                )
            else:
                required_head[dep] = head
        cycle = _find_cycle(required_head, root=0)
        if cycle is not None:
            cycle = _canonical_cycle(cycle)
            problems.append(
                "the required arcs form a cycle: "
                + " -> ".join(str(node) for node in cycle + [cycle[0]])
            )
        for head, dep in sorted(required):
            if 0 <= head <= n and 1 <= dep <= n and head != dep:
                if (head, dep) not in weights:
                    problems.append(
                        f"required arc {(head, dep)} is not a candidate arc "
                        "(the scorer gave it no score)"
                    )
        if problems:
            return ConstrainedParseResult(False, problems=problems)

        allowed = {
            arc: weight for arc, weight in weights.items() if arc not in forbidden
        }
        for dep in range(1, n + 1):
            if dep in required_head:
                continue
            if not any(arc_dep == dep for (arc_head, arc_dep) in allowed):
                problems.append(f"token {dep} has no available incoming arc")
        root_required = [dep for (head, dep) in sorted(required) if head == 0]
        if len(root_required) > 1:
            problems.append(
                "the single-root condition is unsatisfiable: tokens "
                f"{root_required} are all required to attach to the virtual root"
            )
        if not problems and n > 0:
            if root_required:
                root_candidates = root_required
            else:
                root_candidates = [
                    dep
                    for (head, dep) in sorted(allowed)
                    if head == 0 and dep not in required_head
                ]
            if not root_candidates:
                problems.append(
                    "the single-root condition is unsatisfiable: "
                    "no token can be attached to the virtual root"
                )
        if problems:
            return ConstrainedParseResult(False, problems=problems)
        if n == 0:
            return ConstrainedParseResult(True, graph=graph, score=0)

        # Fold the required arcs into the search with a bonus that
        # dominates every possible total score; the maximum arborescence
        # then contains all required arcs whenever some legal tree does.
        bonus = sum(abs(weight) for weight in allowed.values()) + 1
        best = None
        for root_child in root_candidates:
            # Exactly one root arc: only the candidate may attach to the
            # virtual root in this run.
            run_weights = {
                arc: weight
                for arc, weight in allowed.items()
                if arc[0] != 0 or arc[1] == root_child
            }
            for arc in required:
                run_weights[arc] += bonus
            parents = _max_arborescence(list(range(n + 1)), 0, run_weights)
            if parents is None:
                continue
            arcs = {(head, dep) for dep, head in parents.items()}
            if not required <= arcs:
                continue
            score = sum(allowed[arc] for arc in arcs)
            if best is None or score > best[0]:
                best = (score, root_child, parents)
        if best is None:
            return ConstrainedParseResult(
                False,
                problems=[
                    "no legal analysis satisfies the constraints: the required "
                    "arcs cannot be extended to a single-rooted spanning tree"
                ],
            )

        score, root_child, parents = best
        for dep in range(1, n + 1):
            graph.add_arc(parents[dep], dep)
        graph.root = graph.nodes[root_child]
        return ConstrainedParseResult(True, graph=graph, score=score)

    def parse(self, tokens, tags, required_arcs=None, forbidden_arcs=None):
        """
        Parses a list of tokens in accordance to the MST parsing algorithm
        for non-projective dependency parses.  Assumes that the tokens to
        be parsed have already been tagged and those tags are provided.  Various
        scoring methods can be used by implementing the ``DependencyScorerI``
        interface and passing it to the training algorithm.

        :type tokens: list(str)
        :param tokens: A list of words or punctuation to be parsed.
        :type tags: list(str)
        :param tags: A list of tags corresponding by index to the words in the tokens list.
        :param required_arcs: optional arcs that must appear in the parse;
            see :meth:`parse_constrained`.
        :param forbidden_arcs: optional arcs that must not appear in the
            parse; see :meth:`parse_constrained`.
        :return: An iterator of non-projective parses.
        :rtype: iter(DependencyGraph)
        :raises InfeasibleConstraintsError: if the arc constraints or the
            single-root condition cannot be satisfied; the exception's
            ``problems`` attribute explains why.
        """
        result = self.parse_constrained(tokens, tags, required_arcs, forbidden_arcs)
        if not result.feasible:
            raise InfeasibleConstraintsError(result.problems)
        yield result.graph


#################################################################
# Rule-based Non-Projective Parser
#################################################################


class NonprojectiveDependencyParser:
    """
    A non-projective, rule-based, dependency parser.  This parser
    will return the set of all possible non-projective parses based on
    the word-to-word relations defined in the parser's dependency
    grammar, and will allow the branches of the parse tree to cross
    in order to capture a variety of linguistic phenomena that a
    projective parser will not.
    """

    def __init__(self, dependency_grammar, max_time=DEFAULT_MAX_TIME):
        """
        Creates a new ``NonprojectiveDependencyParser``.

        :param dependency_grammar: a grammar of word-to-word relations.
        :type dependency_grammar: DependencyGrammar
        :param max_time: Wall-clock limit, in seconds, for a single ``parse()``
            call.  A dense grammar makes the head-assignment enumeration
            exponential (CWE-407); when the limit is exceeded, ``parse()``
            raises ``TimeoutError``.  Defaults to ``DEFAULT_MAX_TIME``; ``None``
            disables the bound.
        :type max_time: float or None
        """
        self._grammar = dependency_grammar
        self._max_time = max_time

    def parse(self, tokens):
        """
        Parses the input tokens with respect to the parser's grammar.  Parsing
        is accomplished by representing the search-space of possible parses as
        a fully-connected directed graph.  Arcs that would lead to ungrammatical
        parses are removed and a lattice is constructed of length n, where n is
        the number of input tokens, to represent all possible grammatical
        traversals.  All possible paths through the lattice are then enumerated
        to produce the set of non-projective parses.

        param tokens: A list of tokens to parse.
        type tokens: list(str)
        return: An iterator of non-projective parses.
        rtype: iter(DependencyGraph)
        """
        # Create graph representation of tokens
        self._graph = DependencyGraph()

        for index, token in enumerate(tokens):
            self._graph.nodes[index] = {
                "word": token,
                "deps": [],
                "rel": "NTOP",
                "address": index,
            }

        for head_node in self._graph.nodes.values():
            deps = []
            for dep_node in self._graph.nodes.values():
                if (
                    self._grammar.contains(head_node["word"], dep_node["word"])
                    and head_node["word"] != dep_node["word"]
                ):
                    deps.append(dep_node["address"])
            head_node["deps"] = deps

        # Create lattice of possible heads
        roots = []
        possible_heads = []
        for i, word in enumerate(tokens):
            heads = []
            for j, head in enumerate(tokens):
                if (i != j) and self._grammar.contains(head, word):
                    heads.append(j)
            if len(heads) == 0:
                roots.append(i)
            possible_heads.append(heads)

        # Set roots to attempt
        if len(roots) < 2:
            if len(roots) == 0:
                for i in range(len(tokens)):
                    roots.append(i)

            # Traverse lattice
            analyses = []
            for _ in roots:
                stack = []
                analysis = [[] for i in range(len(possible_heads))]
            deadline = (
                None if self._max_time is None else time.perf_counter() + self._max_time
            )
            i = 0
            forward = True
            while i >= 0:
                if deadline is not None and time.perf_counter() > deadline:
                    raise TimeoutError(
                        f"NonprojectiveDependencyParser exceeded its "
                        f"{self._max_time}s time limit; the grammar may be too "
                        "dense for this many tokens. Pass max_time=None to "
                        "disable the limit."
                    )
                if forward:
                    if len(possible_heads[i]) == 1:
                        analysis[i] = possible_heads[i][0]
                    elif len(possible_heads[i]) == 0:
                        analysis[i] = -1
                    else:
                        head = possible_heads[i].pop()
                        analysis[i] = head
                        stack.append([i, head])
                if not forward:
                    index_on_stack = False
                    for stack_item in stack:
                        if stack_item[0] == i:
                            index_on_stack = True
                    orig_length = len(possible_heads[i])

                    if index_on_stack and orig_length == 0:
                        for j in range(len(stack) - 1, -1, -1):
                            stack_item = stack[j]
                            if stack_item[0] == i:
                                possible_heads[i].append(stack.pop(j)[1])

                    elif index_on_stack and orig_length > 0:
                        head = possible_heads[i].pop()
                        analysis[i] = head
                        stack.append([i, head])
                        forward = True

                if i + 1 == len(possible_heads):
                    analyses.append(analysis[:])
                    forward = False
                if forward:
                    i += 1
                else:
                    i -= 1

        # Filter parses
        # ensure 1 root, every thing has 1 head
        for analysis in analyses:
            if analysis.count(-1) > 1:
                # there are several root elements!
                continue

            graph = DependencyGraph()
            graph.root = graph.nodes[analysis.index(-1) + 1]

            for address, (token, head_index) in enumerate(
                zip(tokens, analysis), start=1
            ):
                head_address = head_index + 1

                node = graph.nodes[address]
                node.update({"word": token, "address": address})

                if head_address == 0:
                    rel = "ROOT"
                else:
                    rel = ""
                graph.nodes[head_index + 1]["deps"][rel].append(address)

            # TODO: check for cycles
            yield graph


#################################################################
# Demos
#################################################################


def demo():
    # hall_demo()
    nonprojective_conll_parse_demo()
    rule_based_demo()


def hall_demo():
    npp = ProbabilisticNonprojectiveParser()
    npp.train([], DemoScorer())
    for parse_graph in npp.parse(["v1", "v2", "v3"], [None, None, None]):
        print(parse_graph)


def nonprojective_conll_parse_demo():
    from nltk.parse.dependencygraph import conll_data2

    graphs = [DependencyGraph(entry) for entry in conll_data2.split("\n\n") if entry]
    npp = ProbabilisticNonprojectiveParser()
    npp.train(graphs, NaiveBayesDependencyScorer())
    for parse_graph in npp.parse(
        ["Cathy", "zag", "hen", "zwaaien", "."], ["N", "V", "Pron", "Adj", "N", "Punc"]
    ):
        print(parse_graph)


def rule_based_demo():
    from nltk.grammar import DependencyGrammar

    grammar = DependencyGrammar.fromstring(
        """
    'taught' -> 'play' | 'man'
    'man' -> 'the' | 'in'
    'in' -> 'corner'
    'corner' -> 'the'
    'play' -> 'golf' | 'dachshund' | 'to'
    'dachshund' -> 'his'
    """
    )
    print(grammar)
    ndp = NonprojectiveDependencyParser(grammar)
    graphs = ndp.parse(
        [
            "the",
            "man",
            "in",
            "the",
            "corner",
            "taught",
            "his",
            "dachshund",
            "to",
            "play",
            "golf",
        ]
    )
    print("Graphs:")
    for graph in graphs:
        print(graph)


if __name__ == "__main__":
    demo()
