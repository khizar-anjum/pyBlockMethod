#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class block:
    def __init__(self, center : np.ndarray, angle : float, length : float, r0 : float, block_kind : int,
                 id_ : int = None, edge_i_index : int = None, edge_j_index : int = None):
        """
        Initialize the block class. Blocks are the basic building blocks for the block grid method. They can be of three kinds:
        1. First kind: sectors extending from the vertex
        2. Second kind: half-disks extending from the edge
        3. Third kind: disks inside the polygon

        Parameters:
        center (np.ndarray): The center of the block
        angle (float): The angle of the block
        length (float): The length of the block
        block_kind (int): The kind of the block, either 1, 2, or 3
        id_ (int): The id of the block, used for indexing, and finding overlaps between blocks
        edge_i_index (int): The index of the edge vector from which the block extends. For blocks of second kind,
            this parameter shows the edge that they lie on, None for blocks of third kind.
        edge_j_index (int): The index of the edge vector to which the block extends. This parameter is None for blocks
            of second and third kind.

        Returns:
        None
        """
        self.center = center
        self.angle = angle
        self.length = length # radius of the basic block! (radius r)
        self.r0 = r0 # radius of the extended block! length (r) < r0
        self.block_kind = block_kind
        self.id_ = id_ # id of the block, used for indexing, and finding overlaps between blocks
        self.edge_i_index = edge_i_index # index of the edge vector from which the block extends, None for third kind blocks
        self.edge_j_index = edge_j_index # index of the edge vector to which the block extends, None for second and third kind blocks
        self.inner_points = [] # inner points of the block from the meshgrid, in global coordinates

    def is_inside(self, point):
        # Vector from block center to point
        v = point - self.center
        dist = np.linalg.norm(v)
        if dist > self.length:
            return False
        return True