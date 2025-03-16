"""
Visualization utilities for color optimization.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_color_list(color_list, title, save_path=None):
    """
    Plot a row of colored markers for the given (R,G,B) color list.
    Each color is shown as a circle on the x-axis.
    
    Parameters:
      color_list: list of (R, G, B) values in [0, 255]
      title: title for the plot
      save_path: path to save the figure (or None to display)
    """
    # plt.figure()
    for i, color in enumerate(color_list):
        # Normalize for matplotlib [0..1]
        r, g, b = color[0]/255, color[1]/255, color[2]/255
        plt.plot(i, 0, marker='o', markersize=20,
                 markerfacecolor=(r, g, b),
                 markeredgecolor=(r, g, b))
    plt.xlim(-1, len(color_list))
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.title(title)
    
    # if save_path:
    #     plt.savefig(save_path)
    #     plt.close()
    # else:
    #     plt.show()


def plot_2d_point_distribution(points, title="Point Distribution", prior_indices=None, 
                             annotate=True, save_path=None, domain=(0, 1)):
    """
    Visualize a 2D point distribution.
    
    Parameters:
      points: numpy array of shape (n, 2)
      title: title for the plot
      prior_indices: indices of prior points (to color differently)
      annotate: whether to annotate points with their indices
      save_path: path to save the figure (or None to display)
      domain: tuple (min, max) defining the domain range
    """
    plt.figure(figsize=(8, 8))
    
    # Plot regular points
    if prior_indices is None:
        plt.scatter(points[:, 0], points[:, 1], color='red', s=80)
        if annotate:
            for i, p in enumerate(points):
                plt.text(p[0] + 0.02, p[1] + 0.02, f"{i+1}", fontsize=10)
    else:
        # Plot non-prior points
        regular_mask = np.ones(len(points), dtype=bool)
        regular_mask[prior_indices] = False
        plt.scatter(points[regular_mask, 0], points[regular_mask, 1], color='red', s=80, label='Regular Points')
        
        # Plot prior points
        plt.scatter(points[prior_indices, 0], points[prior_indices, 1], color='green', s=100, label='Prior Points')
        
        if annotate:
            for i, p in enumerate(points):
                plt.text(p[0] + 0.02, p[1] + 0.02, f"{i+1}", fontsize=10)
        
        plt.legend()
    
    # Calculate the min distance and add to title
    from .core import calculate_min_distance
    min_dist = calculate_min_distance(points)
    title = f"{title}\nMin dist: {min_dist:.4f}"
    
    plt.title(title)
    # domain_min, domain_max = domain
    # margin = 0.1 * (domain_max - domain_min)
    # plt.xlim(domain_min - margin, domain_max + margin)
    # plt.ylim(domain_min - margin, domain_max + margin)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_optimization_results(points, loss_history, title="Optimization Results", 
                            prior_indices=None, save_path=None, domain=(0, 1)):
    """
    Plot optimization results: loss history and final point positions.
    
    Parameters:
      points: numpy array of shape (n, 2) with final point positions
      loss_history: list of loss values during optimization
      title: base title for the plot
      prior_indices: indices of prior points (to color differently)
      save_path: path to save the figure (or None to display)
      domain: tuple (min, max) defining the domain range
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss history
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss over iterations")
    plt.grid(True, alpha=0.3)
    
    # Plot final point positions
    plt.subplot(1, 2, 2)
    
    if prior_indices is None:
        # No prior points - all points are the same color
        plt.scatter(points[:, 0], points[:, 1], s=100, c='red')
        for i, p in enumerate(points):
            plt.text(p[0] + 0.02, p[1] + 0.02, f"{i+1}", color='blue', fontsize=10)
    else:
        # With prior points - plot in different colors
        regular_mask = np.ones(len(points), dtype=bool)
        regular_mask[prior_indices] = False
        
        # Plot regular points
        plt.scatter(points[regular_mask, 0], points[regular_mask, 1], 
                  s=100, c='red', label='Optimized')
        
        # Plot prior points
        plt.scatter(points[prior_indices, 0], points[prior_indices, 1], 
                  s=120, c='green', label='Prior')
        
        # Annotations
        for i, p in enumerate(points):
            plt.text(p[0] + 0.02, p[1] + 0.02, f"{i+1}", color='blue', fontsize=10)
        
        plt.legend()
    
    # Add min distance to title
    from .core import calculate_min_distance
    min_dist = calculate_min_distance(points)
    plt.title(f"{title} - Final positions\nMin dist: {min_dist:.4f}")
    
    # Set axis limits
    # domain_min, domain_max = domain
    # margin = 0.1 * (domain_max - domain_min)
    # plt.xlim(domain_min - margin, domain_max + margin)
    # plt.ylim(domain_min - margin, domain_max + margin)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_distance_distributions(greedy_dists, gd_dists, title="Distance Distributions", save_path=None):
    """
    Plot histograms of the pairwise distance distributions for both methods.
    
    Parameters:
      greedy_dists: array of distances from the greedy approach
      gd_dists: array of distances from the gradient descent approach
      title: title for the plot
      save_path: path to save the figure (or None to display)
    """
    plt.figure(figsize=(12, 6))
    
    plt.hist(greedy_dists, bins=20, alpha=0.5, label=f'Greedy (min={greedy_dists[0]:.4f})')
    plt.hist(gd_dists, bins=20, alpha=0.5, label=f'GD (min={gd_dists[0]:.4f})')
    
    plt.xlabel('Pairwise Distance')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def convert_points_to_rgb(points, domain=(0, 1)):
    """Convert 3D points to RGB colors [0-255]
    
    Parameters:
    -----------
    points : numpy.ndarray
        3D points with shape (n, 3)
    domain : tuple, default=(0, 1)
        Domain range of the points as (min, max)
        
    Returns:
    --------
    list
        List of RGB tuples, each with values in range [0, 255]
    """
    import numpy as np
    
    domain_min, domain_max = domain
    domain_range = domain_max - domain_min
    
    # Normalize to [0,1] then scale to [0,255]
    normalized = (points - domain_min) / domain_range
    rgb_colors = (normalized * 255).astype(int)
    
    # Clip values to valid RGB range
    rgb_colors = np.clip(rgb_colors, 0, 255)
    
    return [tuple(color) for color in rgb_colors]


def plot_colors_3d(colors, ax=None, title=None):
    """
    Plot colors in 3D RGB space.
    
    Parameters:
      colors: list of (R, G, B) values in [0, 255]
      ax: matplotlib 3D axis (optional)
      title: title for the plot (optional)
      
    Returns:
      matplotlib axis
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to [0,1] for plotting
    rgb_normalized = np.array(colors) / 255.0
    
    # Plot points
    ax.scatter(rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2],
               c=rgb_normalized, s=100)
    
    # Set labels and limits
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    if title:
        ax.set_title(title)
    
    return ax
