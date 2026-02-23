import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class GradientDescent3DVisualizer:
    def __init__(self, function_type='styblinski_tang_3d'):
        """
        Initialize the visualizer with challenging but visualizable 3D function types
        """
        self.function_type = function_type
        self.setup_function()
        
    def setup_function(self):
        """Setup the optimization function and its gradient"""

        if self.function_type == 'styblinski_tang_3d':
            # Styblinski-Tang function: highly multimodal with visible global minimum
            # f(x,y,z) = 0.5 * sum(xi^4 - 16*xi^2 + 5*xi)
            self.f = lambda x: 0.5 * sum(xi**4 - 16*xi**2 + 5*xi for xi in x)
            self.grad_f = lambda x: 0.5 * np.array([4*xi**3 - 32*xi + 5 for xi in x])
            self.minimum = np.array([-2.903534, -2.903534, -2.903534])  # Global minimum
            
        elif self.function_type == 'rastrigin_3d':
            # 3D Rastrigin: highly multimodal, many local minima, global min at origin
            # f(x,y,z) = A*n + sum(xi^2 - A*cos(2*pi*xi))
            A = 10
            self.f = lambda x: (A * len(x) + 
                               sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x))
            self.grad_f = lambda x: np.array([2*xi + 2*A*np.pi*np.sin(2*np.pi*xi) for xi in x])
            self.minimum = np.array([0.0, 0.0, 0.0])  # Global minimum at origin
            
        elif self.function_type == 'griewank_3d':
            # Griewank function: multimodal with global structure, global min at origin
            # f(x,y,z) = 1 + sum(xi^2)/4000 - prod(cos(xi/sqrt(i+1)))
            self.f = lambda x: (1 + sum(xi**2 for xi in x) / 4000 - 
                               np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)]))
            
            def griewank_grad(x):
                grad = np.zeros_like(x)
                cos_prod = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
                for i, xi in enumerate(x):
                    # Derivative of sum term
                    grad[i] = xi / 2000
                    # Derivative of product term
                    if abs(cos_prod) > 1e-10:
                        grad[i] += (cos_prod / np.cos(xi / np.sqrt(i+1))) * np.sin(xi / np.sqrt(i+1)) / np.sqrt(i+1)
                return grad
            
            self.grad_f = griewank_grad
            self.minimum = np.array([0.0, 0.0, 0.0])  # Global minimum at origin


class BatchGradientDescent3D:
    """Standard Batch Gradient Descent for 3D"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.name = "Batch GD"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        return x - self.lr * grad

class MomentumGD3D:
    """Gradient Descent with Momentum for 3D"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.name = "Momentum"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return x - self.velocity

class AdamOptimizer3D:
    """Adam Optimizer for 3D"""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        self.name = "Adam"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        self.t += 1
        
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSpropOptimizer3D:
    """RMSprop Optimizer for 3D"""
    def __init__(self, learning_rate=0.01, gamma=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.v = None
        self.name = "RMSprop"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        
        if self.v is None:
            self.v = np.zeros_like(x)
        
        self.v = self.gamma * self.v + (1 - self.gamma) * (grad ** 2)
        return x - self.lr * grad / (np.sqrt(self.v) + self.epsilon)

class NesterovMomentum3D:
    """Nesterov Accelerated Gradient for 3D"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.name = "Nesterov"
        
    def step(self, x, grad_func, **kwargs):
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        
        # Look-ahead point
        x_lookahead = x - self.momentum * self.velocity
        grad = grad_func(x_lookahead)
        
        self.velocity = self.momentum * self.velocity + self.lr * grad
        return x - self.velocity

def run_optimization_3d(optimizer, visualizer, start_point, max_iterations=250):
    """Run optimization and return the path"""
    
    # Standard iterative optimization
    path = [start_point.copy()]
    x = start_point.copy()
    
    for i in range(max_iterations):
        x_new = optimizer.step(x, visualizer.grad_f)
        path.append(x_new.copy())
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        
        # Prevent divergence (more lenient bounds)
        if np.linalg.norm(x) > 50:
            break
    
    return np.array(path)

def animate_optimization_3d(function_type='styblinski_tang_3d', start_point=None):
    """Create 3D animation comparing optimization methods"""
    visualizer = GradientDescent3DVisualizer(function_type)

    if start_point is None:
        # Reasonable starting points that allow visualization of global minimum
        start_points = {
            'styblinski_tang_3d': np.array([2.0, -1.0, 1.5]),
            'rastrigin_3d': np.array([3.0, -2.5, 2.0]),
            'griewank_3d': np.array([8.0, -6.0, 5.0])
        }
        start_point = start_points.get(function_type, np.array([2.0, -1.0, 1.5]))

    # Initialize optimizers with tuned learning rates
    optimizers = [
        BatchGradientDescent3D(learning_rate=0.02),
        MomentumGD3D(learning_rate=0.015, momentum=0.9),
        NesterovMomentum3D(learning_rate=0.015, momentum=0.9),
        AdamOptimizer3D(learning_rate=0.08),
        RMSpropOptimizer3D(learning_rate=0.06)
    ]
    
    # Run optimizations
    paths = {}
    for opt in optimizers:
        # Reset optimizer state
        if hasattr(opt, 'velocity'):
            opt.velocity = None
        if hasattr(opt, 'm'):
            opt.m = None
            if hasattr(opt, 'v'):
                opt.v = None
            opt.t = 0
        
        print(f"Running {opt.name} on {function_type}...")
        path = run_optimization_3d(opt, visualizer, start_point)
        paths[opt.name] = path
        print(f"  Completed with {len(path)} iterations")
    
    # Create the 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Always plot the global minimum with large, visible marker
    ax.scatter(visualizer.minimum[0], visualizer.minimum[1], visualizer.minimum[2], 
              c='red', s=200, marker='*', label='Global Minimum', 
              edgecolors='black', linewidth=2, alpha=1.0)
    
    # Plot starting point
    ax.scatter(start_point[0], start_point[1], start_point[2],
              c='green', s=150, marker='s', label='Start Point',
              edgecolors='black', linewidth=1, alpha=0.8)
    
    # Colors for different optimizers
    colors = ['#FF0000', '#00AA00', '#0000FF', '#FF8000', '#FF00FF']
    
    # Initialize line objects
    lines = {}
    points = {}
    for i, (name, path) in enumerate(paths.items()):
        line, = ax.plot([], [], [], color=colors[i], linewidth=2, 
                       label=f'{name} ({len(path)} steps)', alpha=0.9)
        point, = ax.plot([], [], [], 'o', color=colors[i], markersize=5, 
                        markeredgecolor='black', markeredgewidth=0.5)
        lines[name] = line
        points[name] = point
    
    # Set axis limits to ensure global minimum is always visible
    all_points = np.vstack([path for path in paths.values()])
    
    # Include the global minimum in bounds calculation
    all_points_with_min = np.vstack([all_points, visualizer.minimum.reshape(1, -1), start_point.reshape(1, -1)])
    
    # Add some margin
    margin = np.std(all_points_with_min, axis=0) * 0.3
    
    x_range = [all_points_with_min[:, 0].min() - margin[0], all_points_with_min[:, 0].max() + margin[0]]
    y_range = [all_points_with_min[:, 1].min() - margin[1], all_points_with_min[:, 1].max() + margin[1]]
    z_range = [all_points_with_min[:, 2].min() - margin[2], all_points_with_min[:, 2].max() + margin[2]]
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Unconstrained Optimization: {function_type.replace("_", " ").title()}', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set a good viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Animation function
    max_length = max(len(path) for path in paths.values())
    
    def animate(frame):
        for name, path in paths.items():
            if frame < len(path):
                # Update trajectory
                lines[name].set_data_3d(path[:frame+1, 0], 
                                       path[:frame+1, 1], 
                                       path[:frame+1, 2])
                # Update current position
                points[name].set_data_3d([path[frame, 0]], 
                                        [path[frame, 1]], 
                                        [path[frame, 2]])
            else:
                # Keep showing the full path if this optimizer has converged
                lines[name].set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                points[name].set_data_3d([path[-1, 0]], 
                                        [path[-1, 1]], 
                                        [path[-1, 2]])
        
        return list(lines.values()) + list(points.values())
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=min(max_length, frames),
                                 interval=interval, blit=False, repeat=True)
    
    plt.tight_layout()
    return fig, anim, paths

# Example usage
if __name__ == "__main__":
    interval = 100  # Faster animation for more detail (lower = faster/more detailed)
    tol = 1e-6
    frames = 200
    
    # Only functions where global minimum is easily visible
    visualizable_functions = [
        'styblinski_tang_3d',  # Global min at (-2.9, -2.9, -2.9)
        'rastrigin_3d',        # Global min at (0, 0, 0) 
        'griewank_3d'          # Global min at (0, 0, 0)
    ]
    
    for function_to_animate in visualizable_functions:
        print(f"\n{'='*60}")
        print(f"Testing function: {function_to_animate}")
        print(f"{'='*60}")
        
        # Create and show 3D animation
        fig_anim, anim, paths = animate_optimization_3d(function_to_animate)
        
        # Print convergence info
        visualizer = GradientDescent3DVisualizer(function_to_animate)
        print(f"\nResults for {function_to_animate}:")
        print(f"Global minimum at: {visualizer.minimum}")
        
        for name, path in paths.items():
            final_point = path[-1]
            final_value = visualizer.f(final_point)
            distance_to_optimum = np.linalg.norm(final_point - visualizer.minimum)
            print(f"  {name:12s}: {len(path):3d} iters, f={final_value:8.3f}, dist={distance_to_optimum:6.3f}")
                
        # Save animation with higher quality (more FPS = more detail)
        try:
            anim.save(f'detailed_3d_{function_to_animate}.mp4', writer='ffmpeg', fps=15, bitrate=3000)
            print(f"Saved: detailed_3d_{function_to_animate}.mp4")
        except Exception as e:
            print(f"Could not save animation: {e}")