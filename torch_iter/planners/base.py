
import typing as t

from torch_iter.index import Index
from torch_iter.sized_iter import SizedIter
from torch_iter.torch_types import DataLoader

_T = t.TypeVar('_T', covariant=True)


class IterPlanner(t.Generic[_T]):

    def __init__(self, data_loader: DataLoader[_T]) -> None:
        self._data_loader = data_loader

    def get_next_iter(self, val_metric: t.Optional[float] = None) -> SizedIter[t.Tuple[Index, _T]]:
        raise NotImplementedError()


class IterPlannerBuilder(t.Generic[_T]):

    def build(self, data_loader: DataLoader[_T]) -> IterPlanner[_T]:
        raise NotImplementedError()
