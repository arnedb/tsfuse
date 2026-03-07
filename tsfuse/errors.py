"""Exceptions and warnings for TSFuse."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsfuse.computation.nodes import Transformer


class InvalidTagError(Exception):
    """Error that is raised when an invalid tag is created."""


class InvalidPreconditionError(Exception):
    """Error that is raised when a precondition is not satisfied."""

    def __init__(self, transformer: Transformer) -> None:
        self.transformer = transformer

    def __str__(self) -> str:
        return f"Not all preconditions for {self.transformer.__class__.__name__} are satisfied."
