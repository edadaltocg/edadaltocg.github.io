#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Reference implementations and figures for the algorithms and data structures post."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

rng = np.random.default_rng(0)
plt.rcParams["svg.fonttype"] = "none"


@dataclass
class Node:
    value: int
    next: "Node | None" = None


class LinkedList:
    def __init__(self) -> None:
        self.head: Node | None = None

    def push_front(self, value: int) -> None:
        self.head = Node(value, self.head)

    def push_back(self, value: int) -> None:
        if self.head is None:
            self.head = Node(value)
            return
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)

    def pop_front(self) -> int:
        if self.head is None:
            raise IndexError("pop from empty list")
        value, self.head = self.head.value, self.head.next
        return value

    def find(self, value: int) -> Node | None:
        cur = self.head
        while cur is not None and cur.value != value:
            cur = cur.next
        return cur

    def reverse(self) -> None:
        prev, cur = None, self.head
        while cur is not None:
            cur.next, prev, cur = prev, cur, cur.next
        self.head = prev


@dataclass
class DNode:
    value: int
    prev: "DNode | None" = None
    next: "DNode | None" = None


@dataclass
class BSTNode:
    value: int
    left: "BSTNode | None" = None
    right: "BSTNode | None" = None


def bst_insert(root: BSTNode | None, value: int) -> BSTNode:
    if root is None:
        return BSTNode(value)
    if value < root.value:
        root.left = bst_insert(root.left, value)
    elif value > root.value:
        root.right = bst_insert(root.right, value)
    return root


def bst_search(root: BSTNode | None, value: int) -> BSTNode | None:
    while root is not None and root.value != value:
        root = root.left if value < root.value else root.right
    return root


def inorder(root: BSTNode | None) -> list[int]:
    if root is None:
        return []
    return inorder(root.left) + [root.value] + inorder(root.right)


class MinHeap:
    def __init__(self, items: list[int] | None = None) -> None:
        self.a: list[int] = list(items or [])
        for i in range(len(self.a) // 2 - 1, -1, -1):
            self._sift_down(i)

    def push(self, x: int) -> None:
        self.a.append(x)
        self._sift_up(len(self.a) - 1)

    def pop(self) -> int:
        top = self.a[0]
        last = self.a.pop()
        if self.a:
            self.a[0] = last
            self._sift_down(0)
        return top

    def _sift_up(self, i: int) -> None:
        while i > 0 and self.a[(i - 1) // 2] > self.a[i]:
            parent = (i - 1) // 2
            self.a[i], self.a[parent] = self.a[parent], self.a[i]
            i = parent

    def _sift_down(self, i: int) -> None:
        n = len(self.a)
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            smallest = i
            if left < n and self.a[left] < self.a[smallest]:
                smallest = left
            if right < n and self.a[right] < self.a[smallest]:
                smallest = right
            if smallest == i:
                return
            self.a[i], self.a[smallest] = self.a[smallest], self.a[i]
            i = smallest


class Graph:
    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self.adj: dict[int, list[tuple[int, float]]] = {}

    def add_node(self, u: int) -> None:
        self.adj.setdefault(u, [])

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self.adj.setdefault(u, []).append((v, w))
        self.adj.setdefault(v, [])
        if not self.directed:
            self.adj[v].append((u, w))

    def neighbours(self, u: int) -> list[tuple[int, float]]:
        return self.adj.get(u, [])

    def to_matrix(self) -> tuple[list[int], np.ndarray]:
        nodes = sorted(self.adj)
        idx = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        M = np.full((n, n), np.inf)
        np.fill_diagonal(M, 0.0)
        for u, es in self.adj.items():
            for v, w in es:
                M[idx[u], idx[v]] = w
        return nodes, M


def binary_search(a: list[int], target: int) -> int:
    """Return an index of `target` in sorted `a`, or -1 if absent."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1


def lomuto_partition(a: list[int], lo: int, hi: int) -> int:
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i


def quicksort(a: list[int], lo: int = 0, hi: int | None = None) -> None:
    if hi is None:
        hi = len(a) - 1
    if lo >= hi:
        return
    k = int(rng.integers(lo, hi + 1))  # randomised pivot
    a[k], a[hi] = a[hi], a[k]
    p = lomuto_partition(a, lo, hi)
    quicksort(a, lo, p - 1)
    quicksort(a, p + 1, hi)


def bfs(g: Graph, start: int) -> list[int]:
    seen, order, q = {start}, [], deque([start])
    while q:
        u = q.popleft()
        order.append(u)
        for v, _ in g.neighbours(u):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return order


def dfs_iterative(g: Graph, start: int) -> list[int]:
    seen, order, stack = set(), [], [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for v, _ in reversed(g.neighbours(u)):
            if v not in seen:
                stack.append(v)
    return order


def dfs_recursive(g: Graph, start: int) -> list[int]:
    order, seen = [], set()

    def visit(u: int) -> None:
        seen.add(u)
        order.append(u)
        for v, _ in g.neighbours(u):
            if v not in seen:
                visit(v)

    visit(start)
    return order


def dijkstra(g: Graph, source: int) -> tuple[dict[int, float], dict[int, int | None]]:
    dist = {u: float("inf") for u in g.adj}
    prev: dict[int, int | None] = {u: None for u in g.adj}
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:  # stale entry
            continue
        for v, w in g.neighbours(u):
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    return dist, prev


def astar(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """Shortest 4-connected path on a grid; cells equal to 1 are free, 0 blocked."""
    H, W = grid.shape

    def h(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    g_score: dict[tuple[int, int], int] = {start: 0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[int, int, tuple[int, int]]] = [(h(start), 0, start)]
    while pq:
        _, g, u = heapq.heappop(pq)
        if u == goal:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return path[::-1]
        if g > g_score.get(u, 10**9):
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < H and 0 <= v[1] < W) or grid[v] == 0:
                continue
            ng = g + 1
            if ng < g_score.get(v, 10**9):
                g_score[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + h(v), ng, v))
    return []


def fib_naive(n: int) -> int:
    if n < 2:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n: int, cache: dict[int, int] | None = None) -> int:
    if cache is None:
        cache = {}
    if n < 2:
        return n
    if n not in cache:
        cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)
    return cache[n]


def fib_tab(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def knapsack(weights: list[int], values: list[int], W: int) -> int:
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w_i, v_i = weights[i - 1], values[i - 1]
        for c in range(W + 1):
            dp[i][c] = dp[i - 1][c]
            if w_i <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w_i] + v_i)
    return dp[n][W]


def lcs(x: str, y: str) -> str:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i, j, out = m, n, []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            out.append(x[i - 1])
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def _node_circle(ax, xy, label, radius=0.35, fc="white", ec="black", fontsize=11):
    ax.add_patch(Circle(xy, radius=radius, fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.text(
        xy[0], xy[1], str(label), ha="center", va="center", fontsize=fontsize, zorder=3
    )


def _arrow(ax, p0, p1, color="black", style="-|>", lw=1.2, mutation_scale=14, rad=0.0):
    cs = f"arc3,rad={rad}" if rad else "arc3"
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
            connectionstyle=cs,
            zorder=1,
        )
    )


fig, axes = plt.subplots(2, 1, figsize=(8, 3.6))
for ax in axes:
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

# singly
ax = axes[0]
ax.text(-0.3, 0.0, "head", ha="right", va="center", fontsize=10)
_arrow(ax, (-0.2, 0.0), (0.35, 0.0))
for i, v in enumerate([7, 3, 9, 2, 5]):
    cx = 0.9 + 1.6 * i
    ax.add_patch(Rectangle((cx, -0.4), 1.2, 0.8, fc="white", ec="black", lw=1.2))
    ax.plot([cx + 0.8, cx + 0.8], [-0.4, 0.4], color="black", lw=1.0)
    ax.text(cx + 0.4, 0.0, str(v), ha="center", va="center", fontsize=11)
    ax.text(cx + 1.0, 0.0, "•" if i < 4 else "⌀", ha="center", va="center", fontsize=12)
    if i < 4:
        _arrow(ax, (cx + 1.0, 0.0), (cx + 1.6 - 0.05, 0.0))
ax.set_title("Singly linked list", fontsize=11, loc="left")

# doubly
ax = axes[1]
ax.text(-0.3, 0.0, "head", ha="right", va="center", fontsize=10)
_arrow(ax, (-0.2, 0.1), (0.35, 0.1))
for i, v in enumerate([7, 3, 9, 2, 5]):
    cx = 0.9 + 1.6 * i
    ax.add_patch(Rectangle((cx, -0.4), 1.2, 0.8, fc="white", ec="black", lw=1.2))
    ax.plot([cx + 0.3, cx + 0.3], [-0.4, 0.4], color="black", lw=1.0)
    ax.plot([cx + 0.9, cx + 0.9], [-0.4, 0.4], color="black", lw=1.0)
    ax.text(cx + 0.6, 0.0, str(v), ha="center", va="center", fontsize=11)
    if i < 4:
        _arrow(ax, (cx + 0.9, 0.1), (cx + 1.6 + 0.3 - 0.05, 0.1))
        _arrow(ax, (cx + 1.6 + 0.3, -0.1), (cx + 0.9 + 0.05, -0.1))
ax.set_title("Doubly linked list", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig("linked_list.svg", bbox_inches="tight")
plt.close()


def _draw_tree(ax, root, positions):
    for node, (x, y) in positions.items():
        _node_circle(ax, (x, y), node)
    for parent, kids in root.items():
        for k in kids:
            if k in positions:
                _arrow(ax, positions[parent], positions[k], style="-", lw=1.0)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
balanced_edges = {8: [4, 12], 4: [2, 6], 12: [10, 14], 2: [], 6: [], 10: [], 14: []}
balanced_pos = {
    8: (3, 3),
    4: (1.5, 2),
    12: (4.5, 2),
    2: (0.5, 1),
    6: (2.5, 1),
    10: (3.5, 1),
    14: (5.5, 1),
}
ax = axes[0]
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(0.2, 3.8)
ax.set_aspect("equal")
ax.axis("off")
_draw_tree(ax, balanced_edges, balanced_pos)
ax.set_title("Balanced BST: search is $O(\\log n)$", fontsize=11)

degen_edges = {2: [4], 4: [6], 6: [8], 8: [10], 10: [12], 12: [14], 14: []}
degen_pos = {
    v: (i + 0.5, 3.5 - 0.5 * i) for i, v in enumerate([2, 4, 6, 8, 10, 12, 14])
}
ax = axes[1]
ax.set_xlim(-0.2, 7)
ax.set_ylim(-0.2, 3.8)
ax.set_aspect("equal")
ax.axis("off")
_draw_tree(ax, degen_edges, degen_pos)
ax.set_title("Degenerate BST: search is $O(n)$", fontsize=11)
plt.tight_layout()
plt.savefig("binary_tree.svg", bbox_inches="tight")
plt.close()


heap_vals = [1, 3, 2, 7, 5, 4, 9]
positions = {
    0: (3.5, 3.0),
    1: (1.75, 2.0),
    2: (5.25, 2.0),
    3: (0.875, 1.0),
    4: (2.625, 1.0),
    5: (4.375, 1.0),
    6: (6.125, 1.0),
}
parent_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.3, 7.5)
ax.set_ylim(-1.3, 3.6)
ax.set_aspect("equal")
ax.axis("off")
for p, c in parent_edges:
    _arrow(ax, positions[p], positions[c], style="-", lw=1.0)
for i, v in enumerate(heap_vals):
    _node_circle(ax, positions[i], v)
# array view below
cell_w, y0 = 0.9, -0.7
for i, v in enumerate(heap_vals):
    x = 0.4 + i * cell_w
    ax.add_patch(Rectangle((x, y0), cell_w, 0.8, fc="white", ec="black", lw=1.2))
    ax.text(x + cell_w / 2, y0 + 0.4, str(v), ha="center", va="center", fontsize=11)
    ax.text(
        x + cell_w / 2,
        y0 - 0.25,
        f"[{i}]",
        ha="center",
        va="top",
        fontsize=9,
        color="gray",
    )
    # connector from tree node to array cell
    ax.plot(
        [positions[i][0], x + cell_w / 2],
        [positions[i][1] - 0.35, y0 + 0.8],
        color="lightgray",
        lw=0.8,
        ls=":",
        zorder=0,
    )
ax.text(
    3.5,
    -1.2,
    "parent of i is (i-1)//2;  children of i are 2i+1 and 2i+2",
    ha="center",
    fontsize=10,
    color="gray",
)
plt.tight_layout()
plt.savefig("heap_array.svg", bbox_inches="tight")
plt.close()


fig = plt.figure(figsize=(10, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.2])
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect("equal")
ax.axis("off")
g_pos = {1: (0.5, 2.5), 2: (2.0, 3.0), 3: (3.2, 1.8), 4: (1.8, 0.6), 5: (0.4, 1.0)}
g_edges = [(1, 2), (1, 5), (2, 3), (3, 4), (4, 5), (2, 4)]
for u, v in g_edges:
    _arrow(ax, g_pos[u], g_pos[v], style="-", lw=1.0)
for u, p in g_pos.items():
    _node_circle(ax, p, u)
ax.set_title("Graph $G$", fontsize=11)

ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_title("Adjacency list", fontsize=11, loc="left")
adj_lines = [
    "1: [2, 5]",
    "2: [1, 3, 4]",
    "3: [2, 4]",
    "4: [2, 3, 5]",
    "5: [1, 4]",
]
for i, line in enumerate(adj_lines):
    ax.text(0.0, 0.9 - i * 0.18, line, fontsize=11, family="monospace")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax = fig.add_subplot(gs[0, 2])
ax.set_title("Adjacency matrix", fontsize=11, loc="left")
n = 5
M = np.zeros((n, n), dtype=int)
for u, v in g_edges:
    M[u - 1, v - 1] = 1
    M[v - 1, u - 1] = 1
ax.imshow(M, cmap="Greys", vmin=0, vmax=1.5)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(range(1, n + 1))
ax.set_yticklabels(range(1, n + 1))
for i in range(n):
    for j in range(n):
        ax.text(
            j,
            i,
            M[i, j],
            ha="center",
            va="center",
            fontsize=10,
            color="white" if M[i, j] else "black",
        )
plt.tight_layout()
plt.savefig("graph_repr.svg", bbox_inches="tight")
plt.close()


a = [1, 3, 4, 7, 9, 11, 14, 18, 22, 25, 29, 31, 35, 40, 44, 51]
target = 22
intervals = []
lo, hi = 0, len(a)
while lo < hi:
    mid = (lo + hi) // 2
    intervals.append((lo, hi, mid))
    if a[mid] == target:
        break
    if a[mid] < target:
        lo = mid + 1
    else:
        hi = mid

fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(intervals)))
ax.set_xlim(-0.5, len(a))
ax.set_ylim(-len(intervals) - 0.5, 1)
ax.axis("off")
for i, v in enumerate(a):
    ax.text(i, 0.4, str(v), ha="center", va="center", fontsize=10, color="black")
for row, (lo_i, hi_i, mid) in enumerate(intervals):
    y = -row - 0.2
    ax.add_patch(
        Rectangle(
            (lo_i - 0.45, y - 0.3),
            (hi_i - lo_i) - 0.1,
            0.6,
            fc="#e8eef6",
            ec="#456",
            lw=1.0,
        )
    )
    for j in range(lo_i, hi_i):
        ax.text(
            j,
            y,
            str(a[j]),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            weight="bold" if j == mid else "normal",
        )
    ax.text(
        -0.6, y, f"step {row + 1}", ha="right", va="center", fontsize=9, color="gray"
    )
    ax.add_patch(Circle((mid, y), 0.32, fc="none", ec="crimson", lw=1.4, zorder=2))
ax.text(-0.6, 0.4, "array", ha="right", va="center", fontsize=9, color="gray")
ax.set_title(
    f"Binary search for {target}: the live interval halves each step", fontsize=11
)
plt.tight_layout()
plt.savefig("binary_search.svg", bbox_inches="tight")
plt.close()


def _partition_trace(arr):
    snapshots = [list(arr)]
    pivots = [len(arr) - 1]
    a = list(arr)
    lo, hi = 0, len(a) - 1
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
            snapshots.append(list(a))
            pivots.append(hi)
    a[i], a[hi] = a[hi], a[i]
    snapshots.append(list(a))
    pivots.append(i)
    return snapshots, pivots, i


arr = [7, 2, 8, 1, 6, 4, 5]
snaps, pivs, final = _partition_trace(arr)
fig, ax = plt.subplots(figsize=(9, 0.5 + 0.55 * len(snaps)))
ax.set_xlim(-0.6, len(arr))
ax.set_ylim(-len(snaps) - 0.4, 0.8)
ax.axis("off")
for row, (snap, piv) in enumerate(zip(snaps, pivs)):
    y = -row - 0.1
    ax.text(-0.6, y, f"step {row}", ha="right", va="center", fontsize=9, color="gray")
    for j, v in enumerate(snap):
        fc = "#fde6e6" if j == piv else "white"
        ax.add_patch(
            Rectangle((j - 0.4, y - 0.32), 0.8, 0.64, fc=fc, ec="black", lw=1.0)
        )
        ax.text(j, y, str(v), ha="center", va="center", fontsize=11)
ax.text(
    -0.6,
    0.4,
    "pivot in red; final pivot index is " + str(final),
    ha="left",
    va="center",
    fontsize=10,
    color="black",
)
ax.set_title(
    "Lomuto partition trace on [7, 2, 8, 1, 6, 4, 5] with pivot = 5", fontsize=11
)
plt.tight_layout()
plt.savefig("quicksort_trace.svg", bbox_inches="tight")
plt.close()


g = Graph()
edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (5, 8)]
for u, v in edges:
    g.add_edge(u, v)
pos = {
    1: (3, 3),
    2: (1.5, 2),
    3: (4.5, 2),
    4: (0.5, 1),
    5: (2.5, 1),
    6: (3.7, 1),
    7: (5.5, 1),
    8: (2.5, 0),
}
bfs_order = bfs(g, 1)
dfs_order = dfs_recursive(g, 1)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, order, name in zip(axes, [bfs_order, dfs_order], ["BFS", "DFS"]):
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for u, v in edges:
        _arrow(ax, pos[u], pos[v], style="-", lw=1.0, color="gray")
    for u, p in pos.items():
        _node_circle(ax, p, u)
        rank = order.index(u) + 1
        ax.text(p[0] + 0.45, p[1] + 0.35, f"#{rank}", fontsize=10, color="crimson")
    ax.set_title(f"{name} from node 1: visit order in red", fontsize=11)
plt.tight_layout()
plt.savefig("bfs_dfs.svg", bbox_inches="tight")
plt.close()


g = Graph()
weighted_edges = [
    (1, 2, 7),
    (1, 3, 9),
    (1, 6, 14),
    (2, 3, 10),
    (2, 4, 15),
    (3, 4, 11),
    (3, 6, 2),
    (4, 5, 6),
    (5, 6, 9),
]
for u, v, w in weighted_edges:
    g.add_edge(u, v, w)
pos = {
    1: (0.4, 2.0),
    2: (2.0, 3.4),
    3: (2.6, 1.5),
    4: (4.6, 2.8),
    5: (5.4, 0.8),
    6: (2.8, -0.2),
}
dist, prev = dijkstra(g, 1)
sp_edges = {(prev[v], v) for v in prev if prev[v] is not None}

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.set_xlim(-0.6, 6.4)
ax.set_ylim(-1, 4)
ax.set_aspect("equal")
ax.axis("off")
for u, v, w in weighted_edges:
    is_sp = (u, v) in sp_edges or (v, u) in sp_edges
    _arrow(
        ax,
        pos[u],
        pos[v],
        style="-",
        lw=2.2 if is_sp else 1.0,
        color="crimson" if is_sp else "gray",
    )
    mid = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
    ax.text(
        mid[0],
        mid[1] + 0.1,
        str(w),
        fontsize=9,
        color="black",
        bbox=dict(fc="white", ec="none", pad=1.0),
    )
for u, p in pos.items():
    _node_circle(ax, p, u)
    ax.text(
        p[0], p[1] - 0.6, f"d={int(dist[u])}", ha="center", fontsize=9, color="crimson"
    )
ax.set_title(
    "Dijkstra from node 1: shortest-path tree in red, final distances below each node",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("dijkstra.svg", bbox_inches="tight")
plt.close()


H, W = 10, 14
grid = np.ones((H, W), dtype=int)
walls = (
    [(r, 4) for r in range(2, 8)]
    + [(7, c) for c in range(4, 11)]
    + [(r, 10) for r in range(2, 8)]
)
for r, c in walls:
    grid[r, c] = 0
start, goal = (1, 1), (1, 12)


# Re-run A* but track expanded nodes for visualisation
def astar_with_expansions(grid, start, goal):
    H, W = grid.shape

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    g_score = {start: 0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    pq = [(h(start), 0, start)]
    expanded: list[tuple[int, int]] = []
    while pq:
        _, g, u = heapq.heappop(pq)
        if u in expanded:
            continue
        expanded.append(u)
        if u == goal:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return path[::-1], expanded
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < H and 0 <= v[1] < W) or grid[v] == 0:
                continue
            ng = g + 1
            if ng < g_score.get(v, 10**9):
                g_score[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + h(v), ng, v))
    return [], expanded


path, expanded = astar_with_expansions(grid, start, goal)
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-0.5, W - 0.5)
ax.set_ylim(H - 0.5, -0.5)
ax.set_aspect("equal")
ax.axis("off")
for r in range(H):
    for c in range(W):
        if grid[r, c] == 0:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fc="black"))
        else:
            ax.add_patch(
                Rectangle((c - 0.5, r - 0.5), 1, 1, fc="white", ec="#ccc", lw=0.4)
            )
for cell in expanded:
    ax.add_patch(
        Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, fc="#bcd2f0", ec="#ccc", lw=0.4)
    )
for cell in path:
    ax.add_patch(
        Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, fc="#f0a0a0", ec="#ccc", lw=0.4)
    )
ax.add_patch(Circle((start[1], start[0]), 0.3, fc="#2a7", ec="black", lw=1.0))
ax.add_patch(Circle((goal[1], goal[0]), 0.3, fc="crimson", ec="black", lw=1.0))
ax.text(start[1], start[0] - 0.65, "start", ha="center", fontsize=9)
ax.text(goal[1], goal[0] - 0.65, "goal", ha="center", fontsize=9)
ax.set_title(
    "A* on a grid with Manhattan heuristic: blue cells expanded, red cells on shortest path",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("astar.svg", bbox_inches="tight")
plt.close()


x, y = "GATTACA", "GCATGCU"
m, n = len(x), len(y)
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if x[i - 1] == y[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
# traceback
i, j = m, n
back: set[tuple[int, int]] = set()
while i > 0 and j > 0:
    back.add((i, j))
    if x[i - 1] == y[j - 1]:
        i, j = i - 1, j - 1
    elif dp[i - 1][j] >= dp[i][j - 1]:
        i -= 1
    else:
        j -= 1

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_xlim(-0.5, n + 1.5)
ax.set_ylim(m + 1.5, -0.5)
ax.set_aspect("equal")
ax.axis("off")
for j_ in range(n + 1):
    label = " " if j_ == 0 else y[j_ - 1]
    ax.text(j_ + 1, -0.2, label, ha="center", fontsize=11, weight="bold", color="#456")
for i_ in range(m + 1):
    label = " " if i_ == 0 else x[i_ - 1]
    ax.text(
        -0.2, i_ + 0.4, label, ha="center", fontsize=11, weight="bold", color="#456"
    )
for i_ in range(m + 1):
    for j_ in range(n + 1):
        fc = "#fde6e6" if (i_, j_) in back else "white"
        ax.add_patch(
            Rectangle((j_ + 0.55, i_ - 0.05), 0.9, 0.9, fc=fc, ec="black", lw=0.8)
        )
        ax.text(
            j_ + 1.0, i_ + 0.4, str(dp[i_][j_]), ha="center", va="center", fontsize=11
        )
ax.set_title(
    f"LCS table for '{x}' vs '{y}': red cells trace back the LCS '{lcs(x, y)}'",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("dp_table.svg", bbox_inches="tight")
plt.close()
