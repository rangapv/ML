#!/usr/bin/env python3
#author:rangapv@yahoo.com
#03-02-26

from cuopt import routing
from cuopt import distance_engine
import cudf
import numpy as np
import pandas as pd
factory_open_time = 0
factory_close_time = 100

offsets = np.array([0, 1, 3, 7, 9, 11, 13, 15, 17, 20, 22])
edges = np.array(
    [2, 2, 4, 0, 1, 3, 5, 2, 6, 1, 7, 2, 8, 3, 9, 4, 8, 5, 7, 9, 6, 8]
)
weights = np.array(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2]
)

target_locations = np.array([0, 4, 5, 6])

waypoint_graph = distance_engine.WaypointMatrix(offsets, edges, weights)
cost_matrix = waypoint_graph.compute_cost_matrix(target_locations)
transit_time_matrix = cost_matrix.copy(deep=True)
target_map = {v: k for k, v in enumerate(target_locations)}
index_map = {k: v for k, v in enumerate(target_locations)}
print(f"Waypoint graph node to time matrix index mapping \n{target_map}\n")
print(cost_matrix)
