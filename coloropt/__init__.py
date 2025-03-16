"""
Color Optimization Package - Tools for optimal color selection and placement
"""

__version__ = "0.1.0"

# Import main functions to make them available at the package level
from .core import (calculate_min_distance, compute_min_distances,
                   euclidean_distance)
from .gradient import optimize_points_gd, optimize_points_with_prior_gd, optimize_points_3d
from .grid import farthest_point_sampling_2d, farthest_point_sampling_rgb, get_hsv_colors
from .visualization import (plot_2d_point_distribution, plot_color_list,
                            plot_distance_distributions,
                            plot_optimization_results, plot_colors_3d,
                            convert_points_to_rgb)
