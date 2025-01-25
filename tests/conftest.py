#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from src.pyBlockGrid.core.polygon import polygon

@pytest.fixture
def sample_polygon():
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    return polygon(vertices) 