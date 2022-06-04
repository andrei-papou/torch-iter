import math
import typing as t

from torch_iter.index import Index
from torch_iter.metric import LargerIsBetterCriteria
from torch_iter.planners.subset import FixedSubsetIterPlanner, FixedSubsetIterPlannerBuilder, \
    EpochBasedSubsetIterPlanner, EpochBasedSubsetIterPlannerBuilder, \
    MetricBasedSubsetIterPlanner, MetricBasedSubsetIterPlannerBuilder
from torch_iter.subset_size import NatSubsetSize

T1, T2 = t.TypeVar('T1', bound=t.SupportsFloat), t.TypeVar('T2', bound=t.SupportsFloat)


def pair_is_close(lhs: t.Tuple[T1, T2], rhs: t.Tuple[T1, T2]) -> bool:
    return math.isclose(lhs[0], rhs[0]) and math.isclose(lhs[1], rhs[1])


def is_empty(it: t.Iterator[t.Any]) -> bool:
    return not any(True for _ in it)


class RangeDataLoader:

    def __init__(self, size: int) -> None:
        self._size = size

    def __iter__(self) -> t.Iterator[int]:
        return iter(range(self._size))

    def __len__(self) -> int:
        return self._size


def test_fixed_subset_iter_planner():
    builder = FixedSubsetIterPlannerBuilder(subset_size=NatSubsetSize(2))
    planner = builder.build(data_loader=RangeDataLoader(5))
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (0, 5)), 0)
    assert next(it) == (Index(0, (1, 5)), 1)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 3
    assert next(it) == (Index(0, (2, 5)), 2)
    assert next(it) == (Index(0, (3, 5)), 3)
    assert next(it) == (Index(0, (4, 5)), 4)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(1, (0, 5)), 0)
    assert next(it) == (Index(1, (1, 5)), 1)
    assert is_empty(it)

    planner = FixedSubsetIterPlanner(
        data_loader=RangeDataLoader(5),
        subset_size=2,
        stop_at_epoch_end=False)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (0, 5)), 0)
    assert next(it) == (Index(0, (1, 5)), 1)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (2, 5)), 2)
    assert next(it) == (Index(0, (3, 5)), 3)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (4, 5)), 4)
    assert next(it) == (Index(1, (0, 5)), 0)
    assert is_empty(it)


def test_epoch_based_subset_iter_planner():
    builder = EpochBasedSubsetIterPlannerBuilder(
        epoch_to_subset_size_mapping={
            0: NatSubsetSize(2),
            1: NatSubsetSize(3),
        })
    planner = builder.build(data_loader=RangeDataLoader(5))
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (0, 5)), 0)
    assert next(it) == (Index(0, (1, 5)), 1)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 3
    assert next(it) == (Index(0, (2, 5)), 2)
    assert next(it) == (Index(0, (3, 5)), 3)
    assert next(it) == (Index(0, (4, 5)), 4)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 5
    assert next(it) == (Index(1, (0, 5)), 0)
    assert next(it) == (Index(1, (1, 5)), 1)
    assert next(it) == (Index(1, (2, 5)), 2)
    assert next(it) == (Index(1, (3, 5)), 3)
    assert next(it) == (Index(1, (4, 5)), 4)
    assert is_empty(it)

    planner = EpochBasedSubsetIterPlanner(
        data_loader=RangeDataLoader(5),
        epoch_to_subset_size_mapping={
            0: 2,
            1: 3,
        },
        stop_at_epoch_end=False)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (0, 5)), 0)
    assert next(it) == (Index(0, (1, 5)), 1)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (2, 5)), 2)
    assert next(it) == (Index(0, (3, 5)), 3)
    assert is_empty(it)
    it = planner.get_next_iter()
    assert len(it) == 2
    assert next(it) == (Index(0, (4, 5)), 4)
    assert next(it) == (Index(1, (0, 5)), 0)
    it = planner.get_next_iter()
    assert len(it) == 3
    assert next(it) == (Index(1, (1, 5)), 1)
    assert next(it) == (Index(1, (2, 5)), 2)
    assert next(it) == (Index(1, (3, 5)), 3)
    assert is_empty(it)


def test_metric_based_subset_iter_planner():
    builder = MetricBasedSubsetIterPlannerBuilder(
        metric_criteria=LargerIsBetterCriteria(),
        metric_threshold_to_subset_size_mapping={
            0.0: NatSubsetSize(1),
            0.5: NatSubsetSize(3),
        })
    planner = builder.build(data_loader=RangeDataLoader(5))
    it = planner.get_next_iter()
    assert len(it) == 1
    assert next(it) == (Index(0, (0, 5)), 0)
    assert is_empty(it)
    it = planner.get_next_iter(0.5)
    assert len(it) == 4
    assert next(it) == (Index(0, (1, 5)), 1)
    assert next(it) == (Index(0, (2, 5)), 2)
    assert next(it) == (Index(0, (3, 5)), 3)
    assert next(it) == (Index(0, (4, 5)), 4)
    assert is_empty(it)

    planner = MetricBasedSubsetIterPlanner(
        data_loader=RangeDataLoader(5),
        metric_criteria=LargerIsBetterCriteria(),
        metric_threshold_to_subset_size_mapping={
            0.0: 1,
            0.25: 2,
            0.5: 3,
        },
        stop_at_epoch_end=False)
    it = planner.get_next_iter()
    assert len(it) == 1
    assert next(it) == (Index(0, (0, 5)), 0)
    assert is_empty(it)
    it = planner.get_next_iter(0.5)
    assert len(it) == 3
    assert next(it) == (Index(0, (1, 5)), 1)
    assert next(it) == (Index(0, (2, 5)), 2)
    assert next(it) == (Index(0, (3, 5)), 3)
    assert is_empty(it)
    it = planner.get_next_iter(0.3)
    assert len(it) == 2
    assert next(it) == (Index(0, (4, 5)), 4)
    assert next(it) == (Index(1, (0, 5)), 0)
