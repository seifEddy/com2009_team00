#!/usr/bin/env python3
import  numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

@lru_cache
def reduce_points_tuple(points_tuple, radius, min_points=8):
    points = np.array(points_tuple)
    reduced_pts = []
    reduced_points = None
    for i in range(points.shape[0]):
        num_close_points = 0
        for j in range(i + 1, points.shape[0]):
            if np.sqrt((points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2) < radius:
            # if abs(points[i, 0] - points[j, 0]) < radius and abs(points[i, 1] - points[j, 1]) < radius:
                num_close_points += 1
        if num_close_points >= min_points:
            reduced_pts.append(points[i].tolist())
            reduced_points = np.array(reduced_pts)
    return tuple(map(tuple, reduced_points)) if reduced_points is not None else None

def reduce_points(points, radius, min_points=8):
    reduced_pts = []
    reduced_points = None
    for i in range(points.shape[0]):
        num_close_points = 0
        for j in range(i + 1, points.shape[0]):
            if abs(points[i, 0] - points[j, 0]) < radius and abs(points[i, 0] - points[j, 0]) < radius:
                num_close_points += 1
        if num_close_points >= min_points:
            # input(str(points[i].shape))
            reduced_pts.append(points[i].tolist())
            reduced_points = np.array(reduced_pts)
    return reduced_points


if __name__ == '__main__':
    group1_center = np.array([10, 10])  # Center of the first group
    group2_center = np.array([30, 30])  # Center of the second group
    group3_center = np.array([50, 50])  # Center of the third group

    group1_points = group1_center + 2.0 * np.random.randn(500, 2)  # 3 points around group1_center
    group2_points = group2_center + 2.0 * np.random.randn(500, 2)  # 8 points around group2_center
    group3_points = group3_center + 2.0 * np.random.randn(500, 2)  # 10 points around group3_center

    # Concatenate all points into a single array
    # points = np.concatenate([group1_points, group2_points, group3_points])
    points = np.loadtxt('/home/seghiri/abdulla_ws/src/com2009_team26/scripts/array', dtype=float)
    input(str(points.shape[0]))
    reduced_points = reduce_points(points, 0.01, 10)
    input(str(reduced_points.shape[0]))
    # Plot each group of points
    # plt.scatter(group1_points[:, 0], group1_points[:, 1], color='red', label='Group 1')
    # plt.scatter(group2_points[:, 0], group2_points[:, 1], color='blue', label='Group 2')
    # plt.scatter(group3_points[:, 0], group3_points[:, 1], color='green', label='Group 3')
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
    plt.scatter(reduced_points[:, 0], reduced_points[:, 1], color='yellow', label='Reduced')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Distribution of Points')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


