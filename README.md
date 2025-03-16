# Color Optimization Package

This package provides tools for generating well-distributed color palettes using various algorithms, with a focus on maximizing the minimum distance between colors in RGB space. It offers multiple approaches with different trade-offs between optimization quality and computational efficiency.

## Table of Contents

- [Installation](#installation)
- [Package Goals](#package-goals)
- [Optimization Methods](#optimization-methods)
- [Usage](#usage)
- [Examples](#examples)
- [Algorithms Comparison](#algorithms-comparison)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the package, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd color-optimization-pkg
pip install -r requirements.txt
```

Alternatively, you can install the package directly using:

```bash
pip install .
```

## Package Goals

The primary goal of this package is to generate visually distinct color palettes for data visualization, UI design, and other applications where color differentiation is important. The package:

- Maximizes the minimum distance between colors in RGB space
- Provides multiple optimization approaches with different trade-offs
- Supports constraints like fixed/prior colors
- Works in both 2D (for visual experimentation) and 3D color spaces
- Offers evaluation metrics and visualization tools

## Optimization Methods

The package implements three main approaches for color optimization:

1. **Grid-based Sampling**: Uses farthest point sampling on a dense grid of points in RGB space
2. **Gradient Descent**: Directly optimizes point positions using gradient-based optimization
3. **HSV Sampling**: Samples colors evenly in the HSV color space

## Usage

### Basic Usage

```python
from coloropt.grid import farthest_point_sampling_rgb
from coloropt.gradient import optimize_points_3d
from coloropt.visualization import plot_color_list

# Generate 8 colors using grid sampling
grid_colors = farthest_point_sampling_rgb(n=8)

# Generate 8 colors using gradient descent
optimized_points = optimize_points_3d(n_colors=8)
gd_colors = convert_points_to_rgb(optimized_points)

# Visualize the resulting palette
plot_color_list(grid_colors, title="Grid Sampled Colors")
```

### Finding the Best Palette

```python
from coloropt.core import compare_and_select_best_palette

# Compare methods and automatically select the best palette
best_method, best_colors, metrics = compare_and_select_best_palette(
    n_colors=8,
    method_params={
        'grid': {'sample_size': 10000},
        'gd': {'alpha': 40.0, 'lr': 0.01, 'n_iters': 1500},
        'hsv': {'saturation': 0.85, 'value': 0.85}
    },
    criterion='min_dist'
)
```

## Examples

Check the `examples` directory for detailed usage examples:

- `color_gd_example.ipynb`: Demonstrates gradient descent optimization in both 2D and 3D spaces
- `color_comparison_example.ipynb`: Compares different optimization methods and their results
- `compare_with_distinctipy.ipynb`: Benchmarks our methods against the distinctipy package

## Algorithms Comparison

### Gradient Descent

**Pros:**
- Often produces the largest minimum distance between colors
- Directly optimizes the objective function
- Works well in both 2D and 3D spaces

**Cons:**
- Most computationally expensive (O(n²) per iteration * number of iterations)
- Can get stuck in local minima
- Requires parameter tuning (learning rate, alpha, iterations)

### Grid-based Sampling

**Pros:**
- Generally produces good distributions
- More predictable results than gradient descent
- Simpler implementation

**Cons:**
- Quality depends on the grid density (sample size)
- Can be very computationally intensive with large sample sizes (O(n·m) where m is sample size)
- Memory intensive for high-density grids
- May actually be slower than gradient descent when high sample sizes are used

### HSV Sampling

**Pros:**
- Fastest method (O(n) complexity)
- Perceptually intuitive
- Works well for small palettes

**Cons:**
- Does not maximize minimum distance in RGB space
- Limited control over color distribution
- Less optimal for larger palettes

### When to Use Each Method

- **Gradient Descent**: When quality is critical and computation time is not a constraint
- **Grid Sampling**: For a good balance between quality and computation time when using moderate sample sizes
- **HSV Sampling**: When speed is paramount or when working with small palettes
