from types import EllipsisType
from collections.abc import Iterator
from typing import TypeAlias, Protocol, overload

from matplotlib.axes import Axes

_Slice: TypeAlias = slice | EllipsisType


class OneDimSubPlots(Protocol):

    def ravel(self) -> 'OneDimSubPlots': ...
    @overload
    def __getitem__(self, index: int) -> Axes: ...
    @overload
    def __getitem__(self, index: _Slice) -> 'OneDimSubPlots': ...
    def __getitem__(self, index: int | _Slice) -> 'Axes | OneDimSubPlots': ...
    def __iter__(self) -> Iterator[Axes]: ...


class TwoDimSubPlots(Protocol):

    def ravel(self) -> OneDimSubPlots: ...
    @overload
    def __getitem__(self, index: tuple[int, int]) -> Axes: ...
    @overload
    def __getitem__(self, index: int | tuple[_Slice, int] | tuple[int, _Slice]) -> OneDimSubPlots: ...
    @overload
    def __getitem__(self, index: _Slice | tuple[_Slice, _Slice]) -> 'TwoDimSubPlots': ...
    def __getitem__(
        self,
        index: int
            | _Slice
            | tuple[int, int]
            | tuple[_Slice, int]
            | tuple[int, _Slice]
            | tuple[_Slice, _Slice]
        ) -> 'Axes | OneDimSubPlots | TwoDimSubPlots': ...
    def __iter__(self) -> Iterator[OneDimSubPlots]: ...
