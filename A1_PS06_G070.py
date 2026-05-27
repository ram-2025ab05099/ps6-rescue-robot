"""
AIMLCZG557/AECLZG557 - Assignment 1 (PS6)
Drone Path Planning using A* Search with:
1) Euclidean heuristic
2) Obstacle-aware heuristic

"""

from __future__ import annotations

import ast
import argparse
import heapq
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]
Point = Tuple[int, int]


class InputFormatError(Exception):
    """Raised when input file format is invalid."""


class QueueFullError(Exception):
    """Raised when priority queue insert is attempted on a full queue."""


class QueueEmptyError(Exception):
    """Raised when delete operation is attempted on an empty queue."""


@dataclass
class SearchResult:
    """Container for A* run result used by both heuristics."""
    path: List[Point]
    path_cost: float
    nodes_expanded: int
    success: bool
    message: str


class BoundedPriorityQueue:
    """
    Priority queue with a fixed capacity and clear full/empty behavior.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize bounded queue to explicitly support full/empty checks."""
        if capacity <= 0:
            raise ValueError("Queue capacity must be positive.")
        self.capacity = capacity
        self._heap: List[Tuple[float, float, int, Point]] = []

    def insert(self, item: Tuple[float, float, int, Point]) -> None:
        """Insert one item; raise error if queue is already full."""
        if len(self._heap) >= self.capacity:
            raise QueueFullError("Queue is full. Insert operation cannot be performed.")
        heapq.heappush(self._heap, item)

    def delete_min(self) -> Tuple[float, float, int, Point]:
        """Remove and return smallest-priority item; raise error if empty."""
        if not self._heap:
            raise QueueEmptyError("Queue is empty. Delete operation cannot be performed.")
        return heapq.heappop(self._heap)

    def is_empty(self) -> bool:
        """Helper for A* loop termination."""
        return len(self._heap) == 0


class DroneAStarPlanner:
    """A* planner for 8-directional drone navigation in a binary grid."""

    # Directions: East, West, North, South, NE, NW, SE, SW
    MOVES: List[Tuple[int, int]] = [
        (0, 1),
        (0, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (-1, -1),
        (1, 1),
        (1, -1),
    ]

    def __init__(self, terrain: Grid, alpha: float = 0.35) -> None:
        """Store terrain and obstacle penalty factor for custom heuristic."""
        self.terrain = terrain
        self.rows = len(terrain)
        self.cols = len(terrain[0]) if self.rows > 0 else 0
        self.alpha = alpha

    def in_bounds(self, r: int, c: int) -> bool:
        """Check whether cell index is inside grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r: int, c: int) -> bool:
        """A cell is traversable only when inside grid and value is 1."""
        return self.in_bounds(r, c) and self.terrain[r][c] == 1

    def move_cost(self, current: Point, nxt: Point) -> float:
        """Uniform movement cost chosen for sample-style shortest-step paths."""
        # Uniform move cost keeps path scoring consistent with the assignment samples.
        return 1.0

    def euclidean(self, node: Point, goal: Point) -> float:
        """Traditional heuristic required in the problem statement."""
        return math.dist(node, goal)

    def nearby_obstacles(self, node: Point) -> int:
        """Compute O(n): nearby obstacle count used by custom heuristic."""
        r, c = node
        obstacles = 0

        # Count blocked cells in the 8-neighborhood around the current cell.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.terrain[nr][nc] == 0:
                    obstacles += 1
        return obstacles

    def obstacle_aware(self, node: Point, goal: Point) -> float:
        """Problem-specific heuristic: h'(n) = h(n) + alpha * O(n)."""
        return self.euclidean(node, goal) + self.alpha * self.nearby_obstacles(node)

    def is_diagonal_move_safe(self, current: Point, neighbor: Point) -> bool:
        """Prevents moving diagonally through blocked corners."""
        dr = neighbor[0] - current[0]
        dc = neighbor[1] - current[1]

        if abs(dr) != 1 or abs(dc) != 1:
            return True

        # For diagonal move, forbid only when both side-adjacent cells are blocked.
        # This avoids squeezing through a fully closed corner while still allowing
        # practical diagonal motion in partially open spaces.
        side_1 = (current[0] + dr, current[1])
        side_2 = (current[0], current[1] + dc)
        return self.is_free(*side_1) or self.is_free(*side_2)

    def reconstruct_path(self, parent: Dict[Point, Point], end: Point) -> List[Point]:
        """Backtrack from goal to start using parent map."""
        path = [end]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def _validate_inputs(self, start: Point, end: Point) -> Optional[SearchResult]:
        """Run basic checks before starting A*."""
        if self.rows == 0 or self.cols == 0:
            return SearchResult([], 0.0, 0, False, "Error: terrain matrix is empty.")

        if not self.in_bounds(*start) or not self.in_bounds(*end):
            return SearchResult([], 0.0, 0, False, "Error: start or end index is out of bounds.")

        if not self.is_free(*start) or not self.is_free(*end):
            return SearchResult([], 0.0, 0, False, "Error: start or end lies on blocked terrain.")

        return None

    def plan(self, start: Point, end: Point, heuristic_type: str = "euclidean") -> SearchResult:
        """
        Run A* search using selected heuristic (Euclidean or obstacle-aware).
        """
        validation_error = self._validate_inputs(start, end)
        if validation_error is not None:
            return validation_error

        if start == end:
            return SearchResult([start], 0.0, 0, True, "Start is the destination.")

        heuristic = self.euclidean if heuristic_type == "euclidean" else self.obstacle_aware

        open_heap = BoundedPriorityQueue(max(1, self.rows * self.cols * 8))
        push_count = 0

        g_score: Dict[Point, float] = {start: 0.0}
        parent: Dict[Point, Point] = {}
        closed_set = set()

        start_h = heuristic(start, end)
        try:
            open_heap.insert((start_h, start_h, push_count, start))
        except QueueFullError as exc:
            return SearchResult([], 0.0, 0, False, f"Error: {exc}")
        push_count += 1

        nodes_expanded = 0

        while not open_heap.is_empty():
            try:
                _, _, _, current = open_heap.delete_min()
            except QueueEmptyError as exc:
                return SearchResult([], 0.0, nodes_expanded, False, f"Error: {exc}")

            if current in closed_set:
                continue

            nodes_expanded += 1

            if current == end:
                path = self.reconstruct_path(parent, end)
                return SearchResult(path, g_score[end], nodes_expanded, True, "Path found.")

            closed_set.add(current)

            for dr, dc in self.MOVES:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if not self.is_free(nr, nc):
                    continue
                if neighbor in closed_set:
                    continue
                if not self.is_diagonal_move_safe(current, neighbor):
                    continue

                tentative_g = g_score[current] + self.move_cost(current, neighbor)

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    h = heuristic(neighbor, end)
                    f = tentative_g + h
                    try:
                        open_heap.insert((f, h, push_count, neighbor))
                    except QueueFullError as exc:
                        return SearchResult([], 0.0, nodes_expanded, False, f"Error: {exc}")
                    push_count += 1

        return SearchResult([], 0.0, nodes_expanded, False, "No path exists.")


def parse_input_file(file_path: str) -> Tuple[Grid, Point, Point]:
    """
    Parse input file.
    Expected format:
    terrain = [[...], [...], ...]
    start = (r, c)
    end = (r, c)
    Notes:
    - Supports multiline terrain matrices.
    - Keeps run file-driven instead of hardcoded.
    """
    text = Path(file_path).read_text(encoding="utf-8").strip()
    if not text:
        raise InputFormatError("Input file is empty.")

    # Supports multiline terrain matrix and single-line start/end tuples.
    terrain_match = re.search(r"terrain\s*=\s*(\[[\s\S]*?\])\s*start\s*=", text)
    if terrain_match:
        terrain_str = terrain_match.group(1)
    else:
        terrain_line = re.search(r"terrain\s*=\s*(\[[^\n]*\])", text)
        terrain_str = terrain_line.group(1) if terrain_line else None

    start_match = re.search(r"start\s*=\s*(\([^\n]*\))", text)
    end_match = re.search(r"end\s*=\s*(\([^\n]*\))", text)

    terrain = ast.literal_eval(terrain_str) if terrain_str else None
    start = tuple(ast.literal_eval(start_match.group(1))) if start_match else None
    end = tuple(ast.literal_eval(end_match.group(1))) if end_match else None

    if terrain is None or start is None or end is None:
        raise InputFormatError("Input file must contain terrain, start, and end.")

    if not isinstance(terrain, list) or not terrain or not isinstance(terrain[0], list):
        raise InputFormatError("Terrain must be a non-empty 2D list.")

    cols = len(terrain[0])
    if any(len(row) != cols for row in terrain):
        raise InputFormatError("All terrain rows must have the same length.")

    if any(cell not in (0, 1) for row in terrain for cell in row):
        raise InputFormatError("Terrain values must be only 0 or 1.")

    if len(start) != 2 or len(end) != 2:
        raise InputFormatError("Start and end must be index pairs.")

    return terrain, start, end


def write_output_file(file_path: str, euclidean_result: SearchResult, obstacle_result: SearchResult) -> None:
    """
    Write output in a compact fixed format.
    """
    # Keep output in strict, minimal syntax for evaluation.
    if euclidean_result.success:
        euclidean_line = f"Euclidean_Path = {euclidean_result.path}"
    else:
        euclidean_line = f"Euclidean_Path = []  # {euclidean_result.message}"

    if obstacle_result.success:
        obstacle_line = f"ObstacleAware_Path = {obstacle_result.path}"
    else:
        obstacle_line = f"ObstacleAware_Path = []  # {obstacle_result.message}"

    Path(file_path).write_text(f"{euclidean_line}\n{obstacle_line}\n", encoding="utf-8")


def run_from_files(input_file: str, output_file: str, alpha: float = 0.35) -> None:
    """
    End-to-end execution wrapper for assignment evaluation.
    Reads input -> runs both heuristics -> writes output with guarded error messages.
    """
    try:
        terrain, start, end = parse_input_file(input_file)
        planner = DroneAStarPlanner(terrain, alpha=alpha)

        euclidean_result = planner.plan(start, end, heuristic_type="euclidean")
        obstacle_result = planner.plan(start, end, heuristic_type="obstacle")

        write_output_file(output_file, euclidean_result, obstacle_result)
        print(f"Output written to: {output_file}")

    except FileNotFoundError:
        print(f"Error: input file not found -> {input_file}")
    except InputFormatError as exc:
        print(f"Input format error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    # Use script folder as default base so command works from any current directory.
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="A* Drone Path Planner for PS6 assignment")
    parser.add_argument(
        "--input",
        dest="input_file",
        default=str(base_dir / "inputPS06.txt"),
        help="Path to input file containing terrain, start, and end.",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        default=str(base_dir / "outputPS06.txt"),
        help="Path to output file where result will be written.",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=0.35,
        # Keep alpha configurable for easy tuning of obstacle penalty.
        help="Obstacle penalty factor for obstacle-aware heuristic.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # If user passes relative paths, resolve them against script directory.
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    if not output_path.is_absolute():
        output_path = base_dir / output_path

    run_from_files(str(input_path), str(output_path), alpha=args.alpha)
