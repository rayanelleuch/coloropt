"""
Core functionality for color optimization.
"""
import math

import numpy as np


def euclidean_distance(c1, c2):
    """
    Calculate Euclidean distance between two points/colors.
    
    Parameters:
      c1, c2: arrays/lists/tuples of same length
      
    Returns:
      float: the Euclidean distance
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def calculate_min_distance(points):
    """
    Calculate the minimum pairwise distance between points.
    
    Parameters:
      points: numpy array of shape (n, d) representing n d-dimensional points
      
    Returns:
      float: minimum distance between any pair of points
    """
    n = len(points)
    min_dist = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            min_dist = min(min_dist, dist)
    
    return min_dist


def compute_min_distances(points):
    """
    Compute the minimum distance between any pair of points and all pairwise distances.
    
    Parameters:
      points: numpy array of shape (n, d) representing n d-dimensional points
      
    Returns:
      min_dist: minimum distance between any pair of points
      all_dists: sorted array of all pairwise distances
    """
    n = len(points)
    all_dists = []
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            all_dists.append(dist)
    
    all_dists = np.array(all_dists)
    all_dists.sort()
    
    return all_dists[0] if len(all_dists) > 0 else 0, all_dists


def evaluate_point_distribution(points, title=""):
    """
    Evaluate the distribution of points by analyzing minimum distances.
    
    Parameters:
      points: numpy array of shape (n, d)
      title: title for the evaluation
      
    Returns:
      min_dist: minimum distance
      all_dists: all pairwise distances (sorted)
    """
    min_dist, all_dists = compute_min_distances(points)
    
    print(f"--- Evaluation for {title} ---")
    print(f"Number of points: {len(points)}")
    print(f"Minimum distance: {min_dist:.4f}")
    if len(all_dists) > 0:
        print(f"Mean distance: {np.mean(all_dists):.4f}")
        print(f"Median distance: {np.median(all_dists):.4f}")
        print(f"Max distance: {np.max(all_dists):.4f}")
    print()
    
    return min_dist, all_dists


def compare_and_select_best_palette(n_colors, method_params=None, prior_colors=None, 
                                   criterion='min_dist', methods=None, verbose=False):
    """
    Compare different color palette generation methods and select the best one based on specified criteria.
    
    Parameters:
        n_colors (int): Number of colors to generate
        method_params (dict): Dictionary of parameters for each method
        prior_colors (list): Optional list of prior colors to include
        criterion (str): Criterion for selecting the best method ('min_dist', 'mean_dist', 'time')
        methods (list): List of methods to compare (defaults to ['grid', 'gd', 'hsv'])
        verbose (bool): Whether to print verbose output
        
    Returns:
        tuple: (best_method, best_colors, metrics)
            - best_method (str): Name of the best method
            - best_colors (list): List of RGB colors from the best method
            - metrics (dict): Performance metrics for the best method
    """
    import time
    import numpy as np
    from .grid import farthest_point_sampling_rgb, get_hsv_colors
    from .gradient import optimize_points_3d
    from .visualization import convert_points_to_rgb
    
    if methods is None:
        methods = ['grid', 'gd', 'hsv']
        
    if method_params is None:
        method_params = {
            'grid': {'sample_size': 100000},
            'gd': {'alpha': 40.0, 'lr': 0.01, 'n_iters': 1500, 'num_runs': 2},
            'hsv': {'saturation': 0.85, 'value': 0.85}
        }
    
    results = {}
    
    if verbose:
        print(f"Comparing {len(methods)} methods for generating {n_colors} colors")
    
    for method in methods:
        if verbose:
            print(f"Running {method} method...")
        
        start_time = time.time()
        params = method_params.get(method, {})
        
        if method == 'grid':
            # Grid-based sampling
            sample_size = params.get('sample_size', 100000)
            colors = farthest_point_sampling_rgb(n=n_colors, sample_size=sample_size, prior_colors=prior_colors)
            points_3d = np.array(colors) / 255.0  # Normalize to [0,1]
            
        elif method == 'gd':
            # Gradient descent optimization
            alpha = params.get('alpha', 40.0)
            lr = params.get('lr', 0.01)
            n_iters = params.get('n_iters', 1500)
            num_runs = params.get('num_runs', 2)
            
            # Handle prior colors for GD
            if prior_colors:
                prior_points = np.array(prior_colors) / 255.0  # Normalize to [0,1]
                n_free = n_colors - len(prior_colors)
                # Note: optimize_points_3d needs to be extended to support prior points
                # This is a placeholder for that functionality
                points_3d = optimize_points_3d(n_colors, alpha=alpha, lr=lr, n_iters=n_iters, num_runs=num_runs)
            else:
                points_3d = optimize_points_3d(n_colors, alpha=alpha, lr=lr, n_iters=n_iters, num_runs=num_runs)
            
            colors = convert_points_to_rgb(points_3d)
            
        elif method == 'hsv':
            # HSV color space sampling
            saturation = params.get('saturation', 0.85)
            value = params.get('value', 0.85)
            colors = get_hsv_colors(n=n_colors, saturation=saturation, value=value, 
            # prior_colors=prior_colors
            )
            points_3d = np.array(colors) / 255.0  # Normalize to [0,1]
        
        elapsed_time = time.time() - start_time
        
        # Evaluate the color distribution
        min_dist, all_dists = compute_min_distances(points_3d)
        mean_dist = np.mean(all_dists)
        median_dist = np.median(all_dists)
        
        results[method] = {
            'colors': colors,
            'points': points_3d,
            'min_dist': min_dist,
            'mean_dist': mean_dist,
            'median_dist': median_dist,
            'all_dists': all_dists,
            'time': elapsed_time
        }
        
        if verbose:
            print(f"  - {method}: min_dist={min_dist:.4f}, mean_dist={mean_dist:.4f}, time={elapsed_time:.2f}s")
    
    # Select best method based on criterion
    if criterion == 'min_dist':
        # Select method with highest minimum distance
        best_method = max(results.keys(), key=lambda m: results[m]['min_dist'])
    elif criterion == 'mean_dist':
        # Select method with highest mean distance
        best_method = max(results.keys(), key=lambda m: results[m]['mean_dist'])
    elif criterion == 'time':
        # Select fastest method
        best_method = min(results.keys(), key=lambda m: results[m]['time'])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    best_colors = results[best_method]['colors']
    best_metrics = {k: v for k, v in results[best_method].items() if k not in ['colors', 'points', 'all_dists']}
    
    if verbose:
        print(f"\nBest method: {best_method.upper()}")
        print(f"  - min_dist: {best_metrics['min_dist']:.4f}")
        print(f"  - mean_dist: {best_metrics['mean_dist']:.4f}")
        print(f"  - time: {best_metrics['time']:.2f}s")
    
    return best_method, best_colors, best_metrics
