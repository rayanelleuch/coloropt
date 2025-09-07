"""
Gradient descent optimization methods for color/point placement.
"""
import numpy as np
import torch


def compute_loss(points, alpha):
    """
    Compute the loss for a set of points.
    
    Parameters:
      points: tensor of shape (n,d) where d is the dimension
      alpha: positive scalar controlling the sharpness of the log-sum-exp
      
    Returns:
      loss: scalar loss (to be minimized)
    """
    n = points.shape[0]
    # Compute all pairwise distances
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # shape (n, n, d)
    dists = torch.norm(diff, dim=2)  # shape (n, n)
    # Consider only pairs with i < j (upper triangle)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    selected_dists = dists[mask]
    # Surrogate: f = -1/alpha * log(sum(exp(-alpha * d)))
    f = -1.0 / alpha * torch.log(torch.sum(torch.exp(-alpha * selected_dists)) + 1e-8)
    # We want to maximize f, so define loss = -f
    loss = -f
    return loss


def compute_loss_with_prior(free_points, prior_points, alpha):
    """
    Compute loss when some points (prior_points) are fixed.
    
    Parameters:
      free_points: tensor of shape (m,2) to optimize
      prior_points: tensor of shape (k,2) (fixed)
      alpha: positive scalar controlling the sharpness of the log-sum-exp
    
    Returns:
      loss: scalar loss (to be minimized)
    """
    # Concatenate free and prior points
    all_points = torch.cat([free_points, prior_points], dim=0)
    n = all_points.shape[0]
    diff = all_points.unsqueeze(0) - all_points.unsqueeze(1)
    dists = torch.norm(diff, dim=2)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    selected_dists = dists[mask]
    f = -1.0 / alpha * torch.log(torch.sum(torch.exp(-alpha * selected_dists)) + 1e-8)
    loss = -f
    return loss


def optimize_points_gd(n_points, alpha=50.0, lr=0.01, n_iters=2000, num_runs=1, domain=(0, 1), verbose=True, dim=2):
    """
    Optimize point positions to maximize the minimum distance between any pair of points
    using gradient descent.
    
    Args:
        n_points: Number of points to optimize
        alpha: Sharpness parameter for the loss function
        lr: Learning rate for optimization
        n_iters: Number of iterations
        num_runs: Number of optimization runs (best result will be returned)
        domain: Tuple (min, max) defining the domain bounds
        verbose: Whether to print progress
        dim: Dimensionality of points (default: 2 for 2D)
        
    Returns:
        Tuple of (optimized point positions, loss history)
    """
    best_points = None
    best_min_dist = -1
    best_loss_history = []
    domain_min, domain_max = domain
    domain_range = domain_max - domain_min
    
    for run in range(num_runs):
        # Initialize n_points randomly in the domain with specified dimensionality
        points = torch.rand(n_points, dim) * domain_range + domain_min
        points.requires_grad = True

        optimizer = torch.optim.Adam([points], lr=lr)
        loss_history = []
        
        for it in range(n_iters):
            optimizer.zero_grad()
            
            # Calculate loss using the compute_loss function
            loss = compute_loss(points, alpha)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Project points back to domain
            with torch.no_grad():
                points.data.clamp_(domain_min, domain_max)
            
            # Record loss
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Optional progress reporting
            if verbose and (it + 1) % 100 == 0:
                print(f"Run {run+1}, Iteration {it+1}/{n_iters}, Loss: {current_loss:.4f}")
        
        # Calculate minimum distance for this run
        final_points = points.detach().cpu().numpy()
        from .core import calculate_min_distance
        min_dist = calculate_min_distance(final_points)
        
        if verbose:
            print(f"Run {run+1} completed - Min distance: {min_dist:.4f}")
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = final_points
            best_loss_history = loss_history
            if verbose:
                print(f"New best configuration found: {min_dist:.4f}")
    
    return best_points, best_loss_history


def optimize_points_with_prior_gd(n_free, prior_coords, alpha=50.0, lr=0.01, n_iters=2000, 
                                  num_runs=1, domain=(0, 1), verbose=False):
    """
    Optimize point positions with some fixed prior points.
    
    Args:
        n_free: Number of free points to optimize
        prior_coords: List of (x,y) coordinates of fixed prior points
        alpha: Sharpness parameter for the loss function
        lr: Learning rate for optimization
        n_iters: Number of iterations
        num_runs: Number of optimization runs (best result will be returned)
        domain: Tuple (min, max) defining the domain bounds
        verbose: Whether to print progress
        
    Returns:
        Tuple of (all points including both optimized and prior, loss history, indices of prior points)
    """
    prior_points = torch.tensor(prior_coords, dtype=torch.float32)
    n_prior = prior_points.shape[0]
    best_points = None
    best_min_dist = -1
    best_loss_history = []
    domain_min, domain_max = domain
    domain_range = domain_max - domain_min
    
    for run in range(num_runs):
        # Initialize n_free points randomly in the domain
        free_points = torch.rand(n_free, 2) * domain_range + domain_min
        free_points.requires_grad = True
        optimizer = torch.optim.Adam([free_points], lr=lr)
        loss_history = []
        
        for it in range(n_iters):
            optimizer.zero_grad()
            
            # Combine free and prior points for loss computation
            all_points = torch.cat([free_points, prior_points])
            
            # Calculate loss using the compute_loss function
            loss = compute_loss(all_points, alpha)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Project points back to domain
            with torch.no_grad():
                free_points.data.clamp_(domain_min, domain_max)
            
            # Record loss
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Optional progress reporting
            if verbose and (it + 1) % 100 == 0:
                print(f"Run {run+1}, Iteration {it+1}/{n_iters}, Loss: {current_loss:.4f}")
        
        # Calculate minimum distance for this run
        final_free_points = free_points.detach().cpu().numpy()
        final_all_points = np.vstack([final_free_points, prior_points.numpy()])
        from .core import calculate_min_distance
        min_dist = calculate_min_distance(final_all_points)
        
        if verbose:
            print(f"Run {run+1} completed - Min distance: {min_dist:.4f}")
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = final_all_points
            best_loss_history = loss_history
            if verbose:
                print(f"New best configuration found: {min_dist:.4f}")
    
    # Return all points and indices of the prior points
    prior_indices = list(range(n_free, n_free + n_prior))
    return best_points, best_loss_history, prior_indices


def optimize_points_3d_with_prior(n_free, prior_coords, alpha=50.0, lr=0.01, n_iters=2000,
                                  num_runs=1, domain=(0, 1), verbose=False):
    """
    Optimize 3D point positions with some fixed prior points.

    Args:
        n_free: Number of free points to optimize
        prior_coords: List of (x,y,z) coordinates of fixed prior points
        alpha: Sharpness parameter for the loss function
        lr: Learning rate for optimization
        n_iters: Number of iterations
        num_runs: Number of optimization runs (best result will be returned)
        domain: Tuple (min, max) defining the domain bounds
        verbose: Whether to print progress

    Returns:
        Tuple of (all points including both optimized and prior, loss history, indices of prior points)
    """
    prior_points = torch.tensor(prior_coords, dtype=torch.float32)
    n_prior = prior_points.shape[0]
    dim = 3 # Ensure 3D
    if prior_points.shape[1] != dim:
        raise ValueError(f"prior_coords must be 3-dimensional, got shape {prior_points.shape}")

    best_points = None
    best_min_dist = -1
    best_loss_history = []
    domain_min, domain_max = domain
    domain_range = domain_max - domain_min

    for run in range(num_runs):
        # Initialize n_free points randomly in the 3D domain
        free_points = torch.rand(n_free, dim) * domain_range + domain_min
        free_points.requires_grad = True
        optimizer = torch.optim.Adam([free_points], lr=lr)
        loss_history = []

        for it in range(n_iters):
            optimizer.zero_grad()

            # Combine free and prior points for loss computation
            all_points = torch.cat([free_points, prior_points], dim=0)

            # Calculate loss using the compute_loss function
            loss = compute_loss(all_points, alpha)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Project points back to domain
            with torch.no_grad():
                free_points.data.clamp_(domain_min, domain_max)

            # Record loss
            current_loss = loss.item()
            loss_history.append(current_loss)

            # Optional progress reporting
            if verbose and (it + 1) % 100 == 0:
                print(f"Run {run+1}, Iteration {it+1}/{n_iters}, Loss: {current_loss:.4f}")

        # Calculate minimum distance for this run
        final_free_points = free_points.detach().cpu().numpy()
        final_all_points = np.vstack([final_free_points, prior_points.numpy()])
        from .core import calculate_min_distance
        min_dist = calculate_min_distance(final_all_points)

        if verbose:
            print(f"Run {run+1} completed - Min distance: {min_dist:.4f}")

        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best_points = final_all_points
            best_loss_history = loss_history
            if verbose:
                print(f"New best configuration found: {min_dist:.4f}")

    # Return all points and indices of the prior points
    prior_indices = list(range(n_free, n_free + n_prior))
    return best_points, best_loss_history, prior_indices


def optimize_points_3d(n_points, alpha=50.0, lr=0.01, n_iters=2000, num_runs=1, domain=(0, 1)):
    """Run 3D point optimization for RGB colors
    
    Parameters:
    -----------
    n_points : int
        Number of points to optimize
    alpha : float, default=50.0
        Sharpness parameter for the softmin approximation
    lr : float, default=0.01
        Learning rate for the optimizer
    n_iters : int, default=2000
        Number of optimization iterations
    num_runs : int, default=1
        Number of optimization runs to perform (best result is returned)
    domain : tuple, default=(0, 1)
        Domain range for the points as (min, max)
        
    Returns:
    --------
    numpy.ndarray
        Optimized 3D points with shape (n_points, 3)
    """
    # Reuse the optimize_points_gd function with dim=3
    points, _ = optimize_points_gd(
        n_points=n_points,
        alpha=alpha,
        lr=lr,
        n_iters=n_iters,
        num_runs=num_runs,
        domain=domain,
        verbose=False,
        dim=3
    )
    
    return points
