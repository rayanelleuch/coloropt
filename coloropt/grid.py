"""
Grid-based and greedy sampling methods for color optimization.
"""
import colorsys
import random

import numpy as np

from .core import euclidean_distance


def farthest_point_sampling_2d(points, n, prior_points=None):
    """
    Greedy farthest point sampling in 2D.
    
    Parameters:
      points: a list of candidate 2D points (tuples or lists)
      n: total number of points desired (including any prior points)
      prior_points: a list of points that must be included
      
    Returns:
      A list of n points chosen from 'points'
    """
    # Start with the prior points (if any)
    if prior_points is None:
        chosen = []
    else:
        chosen = list(prior_points)
    # Remove any candidate that is already in the chosen set
    remaining = [p for p in points if p not in chosen]
    
    # If no point has been chosen yet, choose one arbitrarily (here, we pick the first candidate)
    if not chosen and remaining:
        chosen.append(remaining.pop(0))
    
    # Iteratively add the point that is farthest from the chosen set
    while len(chosen) < n and remaining:
        best_point = None
        best_distance = -1
        for p in remaining:
            # Compute distance from candidate p to its closest point in 'chosen'
            d = min(np.linalg.norm(np.array(p) - np.array(c)) for c in chosen)
            if d > best_distance:
                best_distance = d
                best_point = p
        chosen.append(best_point)
        remaining.remove(best_point)
    return chosen


def farthest_point_sampling_rgb(n, sample_size=50000, prior_colors=None):
    """
    Pick `n` colors by greedily maximizing mutual distances from a random sample
    of the 8-bit RGB cube, ensuring that all `prior_colors` are included.
    
    Parameters:
      n: int
          Total number of colors you want in the final set
      sample_size: int
          Number of random RGB points to generate as the candidate pool
      prior_colors: list of (R, G, B)
          A list of colors (in [0..255]) that must be included in the final set
    
    Returns:
      chosen: list of (R, G, B)
          The list of `n` colors, which includes all `prior_colors`
    """
    if prior_colors is None:
        prior_colors = []

    # 1) Randomly sample points from the RGB space
    candidates = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(sample_size)
    ]

    # 2) Start with the prior colors
    chosen = list(prior_colors)  # copy so we don't mutate the original list
    
    # 3) Figure out how many more colors we need
    needed = n - len(chosen)
    if needed < 0:
        raise ValueError("Number of prior_colors exceeds n.")
    
    # If no prior colors were provided, select a random first color
    if not chosen and candidates:
        chosen.append(candidates.pop(random.randint(0, len(candidates) - 1)))
        needed -= 1

    # 4) Iteratively pick the candidate farthest from all chosen so far
    for _ in range(needed):
        best_color = None
        best_dist = -1
        for c in candidates:
            dist_to_closest = min(euclidean_distance(c, ch) for ch in chosen)
            if dist_to_closest > best_dist:
                best_dist = dist_to_closest
                best_color = c
        chosen.append(best_color)
        # Optional: remove the chosen color from candidates to avoid duplicates
        if best_color in candidates:
            candidates.remove(best_color)

    return chosen


def get_hsv_colors(n, saturation=1.0, value=1.0):
    """
    Returns n colors spaced evenly around the HSV hue,
    converting to (R,G,B) in [0..255].
    
    Parameters:
      n: int
          Number of colors to generate
      saturation: float
          Saturation value in [0, 1]
      value: float
          Value/brightness in [0, 1]
    
    Returns:
      colors: list of (R, G, B)
          The list of evenly spaced colors
    """
    colors = []
    for i in range(n):
        hue = i / n  # Evenly spaced hues in [0,1)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors
