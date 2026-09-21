#!/usr/bin/env python
"""Morphing a set of surface contacts into another set, along the forearm.

Shared by ``interpolate_contact_points.py`` (which morphs contact positions) and
``interpolate_depth_field.py`` (which morphs the per-vertex penetration depth
that sits on those positions). Both face the same problem: the kinect samples
one frame per ~33 rows of the neural recording, and the rows in between have to
be filled by carrying one measured frame into the next *across the surface*,
never through the arm and never off it.

The pieces, in the order a caller uses them:

    vertices = load_forearm_vertices(ply)
    graph    = build_surface_graph(vertices, k)          # kNN, euclidean weights
    dist, pred = precompute_geodesics(graph)             # all-pairs, once
    tree     = cKDTree(vertices)

    mode  = detect_contact_mode(frames, ...)             # 'single' or 'multi'
    paths = morph_paths(vertices, dist, pred, tree, start, end, cluster_eps)
    v     = walk_fraction(paths[0], vertices, 0.5)       # vertex halfway along

``morph_paths`` is the core. It returns one vertex path per matched contact,
running source -> target over the surface, so ``path[0]`` is the source vertex
and ``path[-1]`` the target. That is enough to interpolate the position (walk
the path) *and* any per-vertex quantity measured at the two endpoints (blend
between ``path[0]``'s value and ``path[-1]``'s).

Contact mode (one finger vs whole hand)
---------------------------------------
A one-finger touch puts a single blob on the arm; a whole hand puts down 2-4
separate finger patches with gaps between them. Matching contacts globally would
let points drift from one finger into the next, smearing material across those
gaps. So contacts are first clustered (single linkage at ``cluster_eps``) and the
morph is confined to *corresponding* clusters.

Which regime a block is in is decided **once per block** from its early frames
and then held fixed:

* ``single`` - the whole patch is one group (a lone finger patch can briefly
  break in two from sensor noise; clustering it would only add jitter).
* ``multi``  - every frame is clustered and clusters are paired before matching.
  When two clusters merge into one (or one splits in two) the shared cluster is
  divided between its claimants along a geodesic nearest-cluster boundary, so a
  merge closes the gap from both sides instead of teleporting points across it.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

DEFAULT_K = 10
_NO_PATH = -9999

# Clustering. Contact points sit ~1 mm apart within a patch and the cloud's own
# vertices ~0.6 mm apart, while the gaps between fingers are several mm — 4 mm
# splits fingers reliably without fragmenting a single patch.
DEFAULT_CLUSTER_EPS = 4.0  # in the coordinate units of the data (mm here)
DEFAULT_MIN_CLUSTER = 3  # smaller specks are absorbed into their nearest neighbour cluster
DETECT_SAMPLE = 200  # early anchor frames inspected when auto-detecting the block's mode
DETECT_MIN_SHARE = 0.05  # a cluster must hold this share of a frame's points to count
DETECT_MIN_FRAC = 0.25  # ...in this fraction of the inspected frames -> "multi"

Point = Tuple[float, float, float]


# --------------------------------------------------------------------- format
# Matches the pipeline's own serialisation: "[[x y z] [x y z] ...]" / "[]".

def parse_contact_points(cell) -> List[Point]:
    """Parse one ``contact_points`` cell into a list of (x, y, z) tuples."""
    if not isinstance(cell, str):
        return []
    stripped = cell.strip()
    if not stripped or stripped == "[]":
        return []

    points: List[Point] = []
    for match in re.findall(r"\[([^\]]+)\]", stripped):
        # lstrip("[") drops the outer bracket captured with the first triplet.
        parts = match.strip().lstrip("[").replace(",", " ").split()
        if len(parts) == 3:
            try:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return points


def serialize_contact_points(points: Sequence[Point], precision: Optional[int] = None) -> str:
    """Re-serialise (x, y, z) tuples back into the pipeline's cell format.

    If ``precision`` is given, coordinates are rounded to that many decimals —
    keeps the output small and matches the input's ~0.1 mm precision.
    """
    if len(points) == 0:
        return "[]"
    if precision is None:
        inner = " ".join(f"[{x} {y} {z}]" for x, y, z in points)
    else:
        inner = " ".join(
            f"[{round(x, precision)} {round(y, precision)} {round(z, precision)}]"
            for x, y, z in points
        )
    return f"[{inner}]"


def load_forearm_vertices(ply_path: Path) -> np.ndarray:
    """Load the forearm point cloud S as an (N, 3) array."""
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(ply_path))
    vertices = np.asarray(pcd.points, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"Forearm PLY contains no points: {ply_path}")
    logger.info("Loaded forearm cloud: %d points from %s", len(vertices), ply_path.name)
    return vertices


# ---------------------------------------------------------------- surface graph

def build_surface_graph(vertices: np.ndarray, k: int = DEFAULT_K) -> csr_matrix:
    """Symmetric kNN graph over S, edge weight = euclidean distance."""
    n_points = len(vertices)
    k_eff = min(k, n_points - 1)
    if k_eff < 1:
        raise ValueError("Forearm cloud needs at least 2 points.")

    distances, indices = cKDTree(vertices).query(vertices, k=k_eff + 1)  # col 0 == self
    rows = np.repeat(np.arange(n_points), k_eff)
    cols = indices[:, 1:].ravel()
    weights = distances[:, 1:].ravel()

    graph = csr_matrix((weights, (rows, cols)), shape=(n_points, n_points))
    return graph.maximum(graph.T)  # undirected


def precompute_geodesics(graph: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """All-pairs geodesic distances + predecessors over S, computed once.

    For an N-vertex cloud this is an N x N float64 distance matrix and an N x N
    predecessor matrix (~165 MB at N=3718). Doing it once is far cheaper than a
    fresh Dijkstra per frame-pair, since contact patches carry tens-to-hundreds
    of source vertices each.
    """
    distances, predecessors = dijkstra(graph, directed=False, return_predecessors=True)
    logger.info("Precomputed all-pairs geodesics over %d vertices.", graph.shape[0])
    return distances, predecessors


def snap_to_surface(points: Sequence[Point], tree: cKDTree) -> np.ndarray:
    """Indices of the nearest surface vertex for each contact point."""
    if len(points) == 0:
        return np.empty(0, dtype=int)
    _, indices = tree.query(np.asarray(points, dtype=np.float64))
    return np.atleast_1d(indices).astype(int)


# ------------------------------------------------------------------- clusters
# One finger leaves a single patch; a whole hand leaves several with gaps in
# between. Clustering is done on the *raw* contact points (finer than the cloud)
# and the labels are carried over to the snapped vertices.

def _compact(labels: np.ndarray) -> np.ndarray:
    """Relabel to a contiguous 0..k-1 range."""
    _, compacted = np.unique(labels, return_inverse=True)
    return compacted.astype(int)


def cluster_points(
    points: Sequence[Point],
    eps: float = DEFAULT_CLUSTER_EPS,
    min_size: int = DEFAULT_MIN_CLUSTER,
) -> np.ndarray:
    """Label contact points into contiguous patches (single linkage at ``eps``).

    Two points join the same patch when they are within ``eps`` of each other,
    transitively — i.e. connected components of the eps-neighbourhood graph.
    Patches holding fewer than ``min_size`` points are absorbed into the nearest
    surviving patch so a stray speck never becomes a group of its own.
    """
    n_points = len(points)
    if n_points == 0:
        return np.empty(0, dtype=int)

    coords = np.asarray(points, dtype=np.float64)
    pairs = np.asarray(list(cKDTree(coords).query_pairs(eps)), dtype=int).reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_points, n_points)
    )
    _, labels = connected_components(graph, directed=False)

    sizes = np.bincount(labels)
    keep = np.flatnonzero(sizes >= min_size)
    if len(keep) == 0:
        return np.zeros(n_points, dtype=int)  # nothing substantial -> one group
    if len(keep) == len(sizes):
        return _compact(labels)

    stray = ~np.isin(labels, keep)
    solid = np.flatnonzero(~stray)
    _, nearest = cKDTree(coords[solid]).query(coords[stray])
    labels = labels.copy()
    labels[stray] = labels[solid[np.atleast_1d(nearest)]]
    return _compact(labels)


def cluster_vertex_sets(
    points: Sequence[Point],
    tree: cKDTree,
    eps: Optional[float],
    min_size: int = DEFAULT_MIN_CLUSTER,
) -> List[np.ndarray]:
    """Snap contacts onto S, grouped by patch: one unique-vertex array per patch.

    ``eps=None`` disables clustering — the whole contact set becomes one group,
    which is the single-finger behaviour.
    """
    if len(points) == 0:
        return []
    snapped = snap_to_surface(points, tree)
    if eps is None:
        return [np.unique(snapped)]
    labels = cluster_points(points, eps, min_size)
    # Distinct patches can share a vertex where the cloud is coarser than the
    # gap; harmless, the vertex is simply walked along both patches' paths.
    return [np.unique(snapped[labels == g]) for g in range(int(labels.max()) + 1)]


def detect_contact_mode(
    frames: Sequence[Sequence[Point]],
    eps: float = DEFAULT_CLUSTER_EPS,
    min_size: int = DEFAULT_MIN_CLUSTER,
    sample: int = DETECT_SAMPLE,
) -> str:
    """Decide ``'single'`` vs ``'multi'`` for a whole block from its early frames.

    Deliberately *not* re-decided per frame: the regime is a property of the
    block (one finger tip / whole hand), and a whole-hand frame legitimately
    collapses to one cluster at the start and end of a tap, when only part of
    the hand is down. Sampling many early frames rides over that.
    """
    inspected = [points for points in frames if len(points)][:sample]
    if not inspected:
        return "single"

    n_multi = 0
    for points in inspected:
        sizes = np.bincount(cluster_points(points, eps, min_size))
        floor = max(min_size, DETECT_MIN_SHARE * len(points))
        if np.count_nonzero(sizes >= floor) >= 2:
            n_multi += 1

    share = n_multi / len(inspected)
    mode = "multi" if share >= DETECT_MIN_FRAC else "single"
    logger.info(
        "  cluster detection: %d/%d early frames hold >=2 patches (%.0f%%) -> %s",
        n_multi, len(inspected), 100 * share, mode,
    )
    return mode


# ------------------------------------------------------------------ geodesics

def reconstruct_path(predecessors: np.ndarray, source: int, target: int) -> Optional[List[int]]:
    """Vertex path source -> target from a Dijkstra predecessor row."""
    if target == source:
        return [source]

    path = [target]
    limit = len(predecessors) + 1
    while path[-1] != source:
        previous = predecessors[path[-1]]
        if previous < 0:  # unreachable (includes _NO_PATH)
            return None
        path.append(int(previous))
        if len(path) > limit:  # defensive: malformed predecessor chain
            return None
    path.reverse()
    return path


def walk_fractions(
    path: Sequence[int], vertices: np.ndarray, fractions: Sequence[float]
) -> List[int]:
    """Vertices at several fractions along ``path``, measured by arc length.

    Arc length is computed once for the path rather than once per fraction — a
    span is walked at ~32 fractions (one per row of the gap), so recomputing it
    each time dominates the runtime.
    """
    if len(path) == 1:
        return [path[0]] * len(fractions)

    points = vertices[list(path)]
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    total = cumulative[-1]
    if total <= 0:
        return [path[0]] * len(fractions)

    last = len(path) - 1
    out = []
    for t in fractions:
        position = int(np.searchsorted(cumulative, t * total, side="left"))
        out.append(path[min(max(position, 0), last)])
    return out


def walk_fraction(path: Sequence[int], vertices: np.ndarray, t: float) -> int:
    """Vertex a fraction ``t`` along ``path``, measured by arc length."""
    return walk_fractions(path, vertices, (t,))[0]


def match_sets(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Pair every row with a column and every column with a row.

    Hungarian gives the optimal one-to-one core; whatever is left over on the
    larger side is attached to its cheapest counterpart, which reproduces the
    split/merge behaviour of optimal transport without the extra dependency.
    """
    n_rows, n_cols = cost.shape
    finite = np.isfinite(cost)
    fallback = cost[finite].max() * 10.0 if finite.any() else 1.0
    workable = np.where(finite, cost, fallback)

    rows, cols = linear_sum_assignment(workable)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    for i in set(range(n_rows)) - set(rows.tolist()):
        pairs.append((i, int(np.argmin(workable[i, :]))))
    for j in set(range(n_cols)) - set(cols.tolist()):
        pairs.append((int(np.argmin(workable[:, j])), j))
    return pairs


def nearest_share(
    members: np.ndarray,
    claimants: Sequence[np.ndarray],
    which: int,
    geo_dist: np.ndarray,
) -> np.ndarray:
    """The part of ``members`` geodesically closest to ``claimants[which]``.

    Used when several clusters are paired with the same counterpart (two fingers
    merging into one patch, or one splitting in two): the shared patch is divided
    along a geodesic nearest-cluster boundary so each claimant morphs only into
    its own side of it.
    """
    # Distance from every member to each claimant cluster = min over that cluster.
    distances = np.stack(
        [geo_dist[np.ix_(cluster, members)].min(axis=0) for cluster in claimants]
    )
    part = members[np.argmin(distances, axis=0) == which]
    if len(part) == 0:  # every claimant must still morph somewhere
        part = members[[int(np.argmin(distances[which]))]]
    return part


def pair_clusters(
    sources: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    geo_dist: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pair start clusters with end clusters, splitting any that are shared."""
    if len(sources) == 1 and len(targets) == 1:
        return [(sources[0], targets[0])]

    # Cluster-level cost = mean geodesic distance between the two vertex sets.
    # Mean (rather than min) tracks patch *identity*: it stays large for two
    # fingers that happen to touch, so they are not confused with each other.
    cost = np.empty((len(sources), len(targets)), dtype=np.float64)
    for a, source in enumerate(sources):
        for b, target in enumerate(targets):
            block = geo_dist[np.ix_(source, target)]
            finite = block[np.isfinite(block)]
            cost[a, b] = finite.mean() if finite.size else np.inf

    pairs = match_sets(cost)
    source_claims = Counter(a for a, _ in pairs)
    target_claims = Counter(b for _, b in pairs)

    paired: List[Tuple[np.ndarray, np.ndarray]] = []
    for a, b in pairs:
        source, target = sources[a], targets[b]
        if target_claims[b] > 1:  # merge: several fingers -> one patch
            claimants = [x for x, y in pairs if y == b]
            target = nearest_share(
                target, [sources[x] for x in claimants], claimants.index(a), geo_dist
            )
        if source_claims[a] > 1:  # split: one patch -> several fingers
            claimants = [y for x, y in pairs if x == a]
            source = nearest_share(
                source, [targets[y] for y in claimants], claimants.index(b), geo_dist
            )
        paired.append((source, target))
    return paired


# ----------------------------------------------------------------- the morph

def morph_paths(
    vertices: np.ndarray,
    geo_dist: np.ndarray,
    geo_pred: np.ndarray,
    tree: cKDTree,
    start_points: Sequence[Point],
    end_points: Sequence[Point],
    cluster_eps: Optional[float] = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER,
) -> List[List[int]]:
    """One surface path per matched contact, carrying ``start`` into ``end``.

    ``path[0]`` is the source vertex and ``path[-1]`` the target, so a caller can
    interpolate both the position (walk the path) and anything measured at the
    two endpoints (blend the two endpoint values).

    With ``cluster_eps`` set (whole-hand blocks) each endpoint's contacts are
    split into patches and matching is confined to paired patches, so no point
    crosses the gap between two fingers. ``None`` treats the contacts as a single
    patch, which is the single-finger behaviour.

    Returns [] when either side has no contacts.
    """
    # A contact patch is a *set* of surface vertices — dedupe before matching.
    sources = [c for c in cluster_vertex_sets(start_points, tree, cluster_eps, min_cluster_size)
               if len(c)]
    targets = [c for c in cluster_vertex_sets(end_points, tree, cluster_eps, min_cluster_size)
               if len(c)]
    if not sources or not targets:
        return []

    paths: List[List[int]] = []
    for source, target in pair_clusters(sources, targets, geo_dist):
        for i, j in match_sets(geo_dist[np.ix_(source, target)]):
            s, t = int(source[i]), int(target[j])
            path = reconstruct_path(geo_pred[s], s, t)
            paths.append(path if path is not None else [s])
    return paths
