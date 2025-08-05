import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class GradientDescentVisualizer:
    def __init__(self, function_type='quadratic'):
        """
        Initialize the visualizer with different function types
        """
        self.function_type = function_type
        self.setup_function()
        
    def setup_function(self):
        """Setup the optimization function and its gradient"""

        if self.function_type == 'quadratic':
            # Simple quadratic bowl: f(x,y) = x^2 + y^2
            self.f = lambda x: x[0]**2 + x[1]**2
            self.grad_f = lambda x: np.array([2*x[0], 2*x[1]])
            self.minimum = np.array([0.0, 0.0])
            
        elif self.function_type == 'rosenbrock':
            # Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
            a, b = 1, 100
            self.f = lambda x: (a - x[0])**2 + b*(x[1] - x[0]**2)**2
            self.grad_f = lambda x: np.array([
                -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2),
                2*b*(x[1] - x[0]**2)
            ])
            self.minimum = np.array([a, a**2])
            
        elif self.function_type == 'himmelblau':
            # Himmelblau's function: f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
            self.f = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
            self.grad_f = lambda x: np.array([
                4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
                2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
            ])
            # One of the minima
            self.minimum = np.array([3.0, 2.0])

        elif self.function_type == 'beale':
            # Beale function: very narrow valley, tests optimizer robustness
            self.f = lambda x: ((1.5 - x[0] + x[0]*x[1])**2 + 
                            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
                            (2.625 - x[0] + x[0]*x[1]**3)**2)
            self.grad_f = lambda x: np.array([
                2*(1.5 - x[0] + x[0]*x[1])*(-1 + x[1]) + 
                2*(2.25 - x[0] + x[0]*x[1]**2)*(-1 + x[1]**2) + 
                2*(2.625 - x[0] + x[0]*x[1]**3)*(-1 + x[1]**3),
                2*(1.5 - x[0] + x[0]*x[1])*(x[0]) + 
                2*(2.25 - x[0] + x[0]*x[1]**2)*(2*x[0]*x[1]) + 
                2*(2.625 - x[0] + x[0]*x[1]**3)*(3*x[0]*x[1]**2)
            ])
            self.minimum = np.array([3.0, 0.5])

        elif self.function_type == 'rastrigin':
            # Rastrigin function: highly multimodal with many local minima
            A = 10
            self.f = lambda x: (2*A + x[0]**2 - A*np.cos(2*np.pi*x[0]) + 
                            x[1]**2 - A*np.cos(2*np.pi*x[1]))
            self.grad_f = lambda x: np.array([
                2*x[0] + 2*A*np.pi*np.sin(2*np.pi*x[0]),
                2*x[1] + 2*A*np.pi*np.sin(2*np.pi*x[1])
            ])
            self.minimum = np.array([0.0, 0.0])


class BatchGradientDescent:
    """Standard Batch Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.name = "Batch GD"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        return x - self.lr * grad

class StochasticGradientDescent:
    """Stochastic Gradient Descent (simulated with noise)"""
    def __init__(self, learning_rate=0.01, noise_scale=0.1):
        self.lr = learning_rate
        self.noise_scale = noise_scale
        self.name = "SGD"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        # Add noise to simulate stochastic behavior
        noise = np.random.normal(0, self.noise_scale, grad.shape)
        return x - self.lr * (grad + noise)

class MiniBatchGradientDescent:
    """Mini-batch Gradient Descent (simulated with reduced noise)"""
    def __init__(self, learning_rate=0.01, noise_scale=0.05):
        self.lr = learning_rate
        self.noise_scale = noise_scale
        self.name = "Mini-batch GD"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        # Add smaller noise to simulate mini-batch behavior
        noise = np.random.normal(0, self.noise_scale, grad.shape)
        return x - self.lr * (grad + noise)

class MomentumGD:
    """Gradient Descent with Momentum"""
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

class AdamOptimizer:
    """Adam Optimizer"""
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

class RMSpropOptimizer:
    """RMSprop Optimizer"""
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

class AdaGradOptimizer:
    """AdaGrad Optimizer"""
    def __init__(self, learning_rate=0.1, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.G = None
        self.name = "AdaGrad"
        
    def step(self, x, grad_func, **kwargs):
        grad = grad_func(x)
        
        if self.G is None:
            self.G = np.zeros_like(x)
        
        self.G += grad ** 2
        return x - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

def run_optimization(optimizer, visualizer, start_point, max_iterations=1000):
    """Run optimization and return the path"""
    path = [start_point.copy()]
    x = start_point.copy()
    
    for i in range(max_iterations):
        x_new = optimizer.step(x, visualizer.grad_f)
        path.append(x_new.copy())
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        
        # Prevent divergence
        if np.linalg.norm(x) > 10:
            break
    
    return np.array(path)

def create_contour_plot(visualizer, xlim=(-3, 3), ylim=(-3, 3)):
    """Create contour plot of the function"""
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = visualizer.f(np.array([X[i, j], Y[i, j]]))
    
    return X, Y, Z

def animate_optimization(function_type='quadratic', start_point=None):
    """Create animation comparing all optimization methods"""
    visualizer = GradientDescentVisualizer(function_type)



    if start_point is None:
        if function_type == 'quadratic':
            start_point = np.array([2.5, 2.0])
        elif function_type == 'rosenbrock':
            start_point = np.array([-1.0, 1.0])
        elif function_type == 'beale':
            start_point = np.array([1.0, 1.0])
        elif function_type == 'rastrigin':
            start_point = np.array([3.0, 3.0])
        else: # himmelblau
            start_point = np.array([0.0, 0.0])

    # Set up contour plot
    if function_type == 'rosenbrock':
        xlim, ylim = (-2, 2), (-1, 3)
    elif function_type == 'himmelblau':
        xlim, ylim = (-5, 5), (-5, 5)
    elif function_type == 'beale':
        xlim, ylim = (-1, 5), (-1, 2)
    elif function_type == 'rastrigin':
        xlim, ylim = (-5, 5), (-5, 5)
    else:
        xlim, ylim = (-3, 3), (-3, 3)



    # Initialize optimizers
    optimizers = [
        BatchGradientDescent(learning_rate=0.1),
        StochasticGradientDescent(learning_rate=0.1, noise_scale=0.1),
        MiniBatchGradientDescent(learning_rate=0.1, noise_scale=0.05),
        MomentumGD(learning_rate=0.05, momentum=0.9),
        AdamOptimizer(learning_rate=0.1),
        RMSpropOptimizer(learning_rate=0.1),
        AdaGradOptimizer(learning_rate=0.5)
    ]
    
    # Run optimizations
    paths = {}
    for opt in optimizers:
        # Reset optimizer state
        if hasattr(opt, 'velocity'):
            opt.velocity = None
        if hasattr(opt, 'm'):
            opt.m = None
            opt.v = None
            opt.t = 0
        if hasattr(opt, 'G'):
            opt.G = None
        
        path = run_optimization(opt, visualizer, start_point)
        paths[opt.name] = path
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))
        
    X, Y, Z = create_contour_plot(visualizer, xlim, ylim)
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.2, colors='gray')
    
    # Plot minimum
    ax.plot(visualizer.minimum[0], visualizer.minimum[1], 'k*', 
            markersize=6, label='Global Minimum')
    
    # Colors for different optimizers (high contrast)
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FFD700', '#FF1493']
    
    # Initialize line objects
    lines = {}
    points = {}
    for i, (name, path) in enumerate(paths.items()):
        line, = ax.plot([], [], color=colors[i], linewidth=1, 
                       label=name, alpha=0.8)
        point, = ax.plot([], [], 'o', color=colors[i], markersize=4)
        lines[name] = line
        points[name] = point
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Gradient Descent Variants on {function_type.title()} Function')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Animation function
    max_length = max(len(path) for path in paths.values())
    
    def animate(frame):
        for name, path in paths.items():
            if frame < len(path):
                # Update trajectory
                lines[name].set_data(path[:frame+1, 0], path[:frame+1, 1])
                # Update current position
                points[name].set_data([path[frame, 0]], [path[frame, 1]])
            else:
                # Keep showing the full path if this optimizer has converged
                lines[name].set_data(path[:, 0], path[:, 1])
                points[name].set_data([path[-1, 0]], [path[-1, 1]])
        
        return list(lines.values()) + list(points.values())
    
    # Create animation (much slower for detailed observation)
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                 interval=interval, blit=True, repeat=False)
    
    plt.tight_layout()
    return fig, anim, paths

def plot_convergence_comparison(paths, visualizer):
    """Plot convergence curves for all optimizers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (name, path) in enumerate(paths.items()):
        # Calculate function values
        func_values = [visualizer.f(point) for point in path]
        distances = [np.linalg.norm(point - visualizer.minimum) for point in path]
        
        ax1.semilogy(func_values, color=colors[i], label=name, linewidth=2)
        ax2.semilogy(distances, color=colors[i], label=name, linewidth=2)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function Value (log scale)')
    ax1.set_title('Convergence: Function Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Distance to Minimum (log scale)')
    ax2.set_title('Convergence: Distance to Minimum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Example usage - just show animations
if __name__ == "__main__":

    interval = 250
    tol = 1e-6
    frames=500

    # Choose which function to animate ('quadratic', 'rosenbrock', or 'himmelblau' or 'beale' or 'rastrigin')
    function_to_animate = 'rastrigin'  
    
    # Create and show only the animation
    fig_anim, anim, paths = animate_optimization(function_to_animate)
    plt.show()
    
    anim.save(f'gradient_descent_{function_to_animate}.mp4', writer='ffmpeg', fps=10)
