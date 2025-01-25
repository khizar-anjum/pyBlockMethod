#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class PDESolver(ABC):
    """Abstract base class for PDE solvers."""
    
    @abstractmethod
    def solve(self, verbose=False):
        """Solve the PDE."""
        pass

    @abstractmethod
    def _initialize_solution(self):
        """Initialize solution data structures."""
        pass 