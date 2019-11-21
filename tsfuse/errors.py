# -*- coding: UTF-8 -*-
"""
tsfuse.errors
=============

Exceptions and warnings.
"""


class InvalidTagError(Exception):
    """Error that is raised when an invalid tag is created."""
    pass


class InvalidPreconditionError(Exception):
    """Warning that is raised when a precondition is not satisfied."""

    def __init__(self, transformer):
        self.transformer = transformer

    def __str__(self):
        return "Not all preconditions for {} are satisfied." \
            .format(self.transformer.__class__.__name__)
