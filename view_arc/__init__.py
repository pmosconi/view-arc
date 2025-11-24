"""
View Arc Obstacle Detection
============================

Public API for finding the obstacle with largest visible angular coverage
within a field-of-view arc from a viewer point.
"""

from view_arc.api import find_largest_obstacle, ObstacleResult

__all__ = ['find_largest_obstacle', 'ObstacleResult']
__version__ = '0.1.0'
