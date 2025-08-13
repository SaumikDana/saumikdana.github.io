import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesRegression:
    """Time series regression problem for optimization visualization"""
    
    def __init__(self, data, is_levels=True):
        self.data = np.array(data)
        self.is_levels = is_levels
        self.setup_regression()
        
    def setup_regression(self):
        """Create autoregressive time series regression: Y_t = β₀ + β₁*Y_{t-1} + β₂*trend + ε_t"""
        n = len(self.data)
        
        # Create features and targets
        self.Y = self.data[1:]  # Y_t
        self.X = np.column_stack([
            np.ones(n-1),           # intercept
            self.data[:-1],         # Y_{t-1} (lag)
            np.arange(1, n)         # trend
        ])
        
        # Calculate condition number of X'X
        XtX = self.X.T @ self.X
        self.condition_number = np.linalg.cond(XtX)
        
        # True minimum (OLS solution) for visualization
        try:
            self.true_beta = np.linalg.solve(XtX, self.X.T @ self.Y)
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            self.true_beta = np.linalg.pinv(self.X) @ self.Y
            
    def objective(self, beta):
        """Mean squared error objective function"""
        predictions = self.X @ beta
        residuals = self.Y - predictions
        return 0.5 * np.sum(residuals**2)
    
    def gradient(self, beta):
        """Gradient of MSE"""
        predictions = self.X @ beta
        residuals = self.Y - predictions
        return -self.X.T @ residuals



class AdamOptimizer:
    """Adam optimizer with bias correction"""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.name = "Adam"
        
    def reset(self):
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, beta, problem):
        grad = problem.gradient(beta)
        self.t += 1
        
        if self.m is None:
            self.m = np.zeros_like(beta)
            self.v = np.zeros_like(beta)
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return beta - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)



# ADD MISSING OPTIMIZERS
class RMSpropOptimizer:
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.v = None
        self.name = "RMSprop"
        
    def reset(self):
        self.v = None
        
    def step(self, beta, problem):
        grad = problem.gradient(beta)
        
        if self.v is None:
            self.v = np.zeros_like(beta)
        
        # Update moving average of squared gradients
        self.v = self.decay * self.v + (1 - self.decay) * (grad ** 2)
        
        return beta - self.lr * grad / (np.sqrt(self.v) + self.epsilon)

class AdaGradOptimizer:
    """AdaGrad optimizer"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.G = None
        self.name = "AdaGrad"
        
    def reset(self):
        self.G = None
        
    def step(self, beta, problem):
        grad = problem.gradient(beta)
        
        if self.G is None:
            self.G = np.zeros_like(beta)
        
        # Accumulate squared gradients
        self.G += grad ** 2
        
        return beta - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

def load_cpi_data(levels_file='CPIAUCSL.csv', changes_file='CPIAUCSL_PCH.csv'):
    """Load CPI data from CSV files"""
    try:
        # Load levels data
        levels_df = pd.read_csv(levels_file)
        levels_data = levels_df.iloc[:, 1].values  # Second column
        
        # Load changes data  
        changes_df = pd.read_csv(changes_file)
        changes_data = changes_df.iloc[:, 1].values  # Second column
        
        print(f"Loaded CPI data: {len(levels_data)} observations")
        print(f"Levels range: {levels_data.min():.1f} to {levels_data.max():.1f}")
        print(f"Changes range: {changes_data.min():.3f}% to {changes_data.max():.3f}%")
        print(f"Scale difference: {levels_data.max() / changes_data.max():.0f}x")
        
        return levels_data, changes_data
        
    except FileNotFoundError:
        print("CSV files not found. Generating synthetic CPI-like data...")
        return generate_synthetic_cpi_data()

def generate_synthetic_cpi_data(n=540):
    """Generate synthetic CPI data that mimics real characteristics"""
    np.random.seed(42)  # For reproducibility
    
    levels_data = []
    changes_data = []
    
    cpi = 78.0  # Starting value like 1980
    base_growth = 0.003  # ~3.6% annual inflation
    
    for i in range(n):
        levels_data.append(cpi)
        
        # Add realistic components
        noise = np.random.normal(0, 0.002)
        cyclical = 0.001 * np.sin(i * 0.1)
        
        # Economic shocks
        shock = 0
        if i in [120, 240]:  # Simulate crises
            shock = np.random.normal(0, 0.01)
        if i > 480:  # Recent inflation surge
            shock += 0.003
            
        monthly_change = base_growth + noise + cyclical + shock
        pct_change = monthly_change * 100
        
        if i > 0:
            changes_data.append(pct_change)
            
        cpi *= (1 + monthly_change)
    
    levels_data = np.array(levels_data)
    changes_data = np.array(changes_data)
    
    print(f"Generated synthetic CPI data: {len(levels_data)} observations")
    print(f"Levels range: {levels_data.min():.1f} to {levels_data.max():.1f}")
    print(f"Changes range: {changes_data.min():.3f}% to {changes_data.max():.3f}%")
    print(f"Scale difference: {levels_data.max() / changes_data.max():.0f}x")
    
    return levels_data, changes_data

def run_optimization(optimizer, problem, start_beta, max_iterations=500, tolerance=1e-10):
    """Run optimization and return path and metrics"""
    
    beta = start_beta.copy()
    path = [beta.copy()]
    losses = [problem.objective(beta)]
    
    converged = False
    consecutive_errors = 0
    
    for i in range(max_iterations):
        try:
            new_beta = optimizer.step(beta, problem)
            loss = problem.objective(new_beta)
            
            # Check for numerical issues but be more permissive
            if not np.isfinite(loss):
                print(f"Non-finite loss at iteration {i}, using previous beta")
                consecutive_errors += 1
                if consecutive_errors > 10:
                    print(f"Too many consecutive errors, stopping")
                    break
                continue
            
            if loss > 1e25:  # Much higher threshold
                print(f"Very large loss {loss:.2e} at iteration {i}, clipping")
                # Clip the step size and continue
                step = new_beta - beta
                new_beta = beta + 0.1 * step
                loss = problem.objective(new_beta)
            
            consecutive_errors = 0  # Reset error counter
                
            # Much more relaxed convergence for better visualization
            beta_change = np.linalg.norm(new_beta - beta)
            if beta_change < tolerance and i > 200:  # Force at least 200 iterations
                converged = True
                break
                
            beta = new_beta
            path.append(beta.copy())
            losses.append(loss)
            
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            consecutive_errors += 1
            if consecutive_errors > 10:
                print(f"Too many consecutive errors, stopping")
                break
            # Try to continue with a smaller step
            try:
                grad = problem.gradient(beta)
                if hasattr(optimizer, 'lr'):
                    small_step = beta - 0.001 * optimizer.lr * grad
                else:
                    small_step = beta - 0.00001 * grad
                beta = small_step
                path.append(beta.copy())
                losses.append(problem.objective(beta))
            except:
                break
    
    print(f"Optimization finished with {len(path)} iterations")
    
    return {
        'path': np.array(path),
        'losses': np.array(losses),
        'converged': converged,
        'iterations': len(path),
        'final_loss': losses[-1] if losses else float('inf')
    }

def create_visualization(levels_data, changes_data, optimizer_name='adam'):
    """Create side-by-side optimization visualization"""
    
    # Create regression problems
    levels_problem = TimeSeriesRegression(levels_data, is_levels=True)
    changes_problem = TimeSeriesRegression(changes_data, is_levels=False)
    
    print(f"\nCondition Numbers:")
    print(f"Levels: {levels_problem.condition_number:.2e}")
    print(f"Changes: {changes_problem.condition_number:.2f}")
    
    # Create optimizers with different learning rates
    if optimizer_name == 'adam':
        levels_optimizer = AdamOptimizer(learning_rate=0.0001)  # Much smaller for levels
        changes_optimizer = AdamOptimizer(learning_rate=0.01)
    elif optimizer_name == 'rmsprop':
        levels_optimizer = RMSpropOptimizer(learning_rate=0.001, decay=0.9)
        changes_optimizer = RMSpropOptimizer(learning_rate=0.01, decay=0.9)
    elif optimizer_name == 'adagrad':
        levels_optimizer = AdaGradOptimizer(learning_rate=0.01)
        changes_optimizer = AdaGradOptimizer(learning_rate=0.1)
    
    # Starting points
    levels_start = np.array([100.0, 0.5, 0.001])  # Conservative for levels
    changes_start = np.array([0.1, 0.1, 0.001])   # Normal for changes
    
    # Run optimizations with more iterations
    max_iters = 500
    
    print(f"\nRunning {optimizer_name.title()} optimization...")
    levels_result = run_optimization(levels_optimizer, levels_problem, levels_start, max_iterations=max_iters)
    changes_result = run_optimization(changes_optimizer, changes_problem, changes_start, max_iterations=max_iters)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'CPI Time Series Optimization: {optimizer_name.title()} Algorithm', fontsize=16, fontweight='bold')
    
    # Colors
    levels_color = '#e74c3c'
    changes_color = '#27ae60'
    
    # Plot optimization paths (β₁ vs β₂)
    levels_path = levels_result['path']
    changes_path = changes_result['path']
    
    # Levels path (often chaotic)
    ax1.plot(levels_path[:, 1], levels_path[:, 2], 'o-', color=levels_color, 
             linewidth=2, markersize=3, alpha=0.7, label='Optimization Path')
    ax1.plot(levels_problem.true_beta[1], levels_problem.true_beta[2], 
             '*', color='black', markersize=15, label='True Minimum')
    ax1.plot(levels_path[0, 1], levels_path[0, 2], 'o', color='blue', 
             markersize=8, label='Start')
    ax1.set_title('CPI Levels: Optimization Chaos', fontsize=14, fontweight='bold', color=levels_color)
    ax1.set_xlabel('β₁ (Lag Coefficient)')
    ax1.set_ylabel('β₂ (Trend Coefficient)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Changes path (should be smooth)
    ax2.plot(changes_path[:, 1], changes_path[:, 2], 'o-', color=changes_color, 
             linewidth=2, markersize=3, alpha=0.7, label='Optimization Path')
    ax2.plot(changes_problem.true_beta[1], changes_problem.true_beta[2], 
             '*', color='black', markersize=15, label='True Minimum')
    ax2.plot(changes_path[0, 1], changes_path[0, 2], 'o', color='blue', 
             markersize=8, label='Start')
    ax2.set_title('CPI Changes: Smooth Convergence', fontsize=14, fontweight='bold', color=changes_color)
    ax2.set_xlabel('β₁ (Lag Coefficient)')
    ax2.set_ylabel('β₂ (Trend Coefficient)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot loss curves
    ax3.semilogy(levels_result['losses'], color=levels_color, linewidth=3, label='Levels Loss')
    ax3.set_title('CPI Levels: Loss vs Iteration', fontsize=14, fontweight='bold', color=levels_color)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (log scale)')
    ax3.grid(True, alpha=0.3)
    
    ax4.semilogy(changes_result['losses'], color=changes_color, linewidth=3, label='Changes Loss')
    ax4.set_title('CPI Changes: Loss vs Iteration', fontsize=14, fontweight='bold', color=changes_color)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Loss (log scale)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print results
    print(f"\nOptimization Results:")
    print(f"{'Metric':<25} {'Levels':<15} {'Changes':<15}")
    print(f"{'-'*55}")
    print(f"{'Condition Number':<25} {levels_problem.condition_number:<15.2e} {changes_problem.condition_number:<15.2f}")
    print(f"{'Converged':<25} {levels_result['converged']:<15} {changes_result['converged']:<15}")
    print(f"{'Iterations':<25} {levels_result['iterations']:<15} {changes_result['iterations']:<15}")
    print(f"{'Final Loss':<25} {levels_result['final_loss']:<15.2e} {changes_result['final_loss']:<15.6f}")
    
    return fig, levels_result, changes_result

def create_animated_comparison(levels_data, changes_data, optimizer_name='adam'):
    """Create animated comparison showing optimization in real-time"""
    
    # Setup problems and optimizers
    levels_problem = TimeSeriesRegression(levels_data, is_levels=True)
    changes_problem = TimeSeriesRegression(changes_data, is_levels=False)
    
    if optimizer_name == 'adam':
        levels_optimizer = AdamOptimizer(learning_rate=0.0001)
        changes_optimizer = AdamOptimizer(learning_rate=0.01)
    elif optimizer_name == 'rmsprop':
        # RMSprop - should handle ill-conditioning well like Adam
        levels_optimizer = RMSpropOptimizer(learning_rate=0.001, decay=0.9)
        changes_optimizer = RMSpropOptimizer(learning_rate=0.01, decay=0.9)
    else:  # adagrad
        # AdaGrad - adaptive learning rates but gets more conservative over time
        levels_optimizer = AdaGradOptimizer(learning_rate=0.01)
        changes_optimizer = AdaGradOptimizer(learning_rate=0.1)
    
    # Run optimizations to get full paths
    levels_start = np.array([100.0, 0.5, 0.001])
    changes_start = np.array([0.1, 0.1, 0.001])
    
    # Increase max_iterations for longer animations
    max_iters = 500
    
    levels_result = run_optimization(levels_optimizer, levels_problem, levels_start, max_iterations=max_iters)
    changes_result = run_optimization(changes_optimizer, changes_problem, changes_start, max_iterations=max_iters)
    
    # Create animated plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Animated Optimization Comparison: {optimizer_name.title()}', fontsize=16)
    
    # Setup axes
    levels_path = levels_result['path']
    changes_path = changes_result['path']
    
    # FIXED: Set consistent axis limits that include both optimization paths AND true optima
    # For levels plot
    levels_x_min = min(levels_path[:, 1].min(), levels_problem.true_beta[1]) - 0.1
    levels_x_max = max(levels_path[:, 1].max(), levels_problem.true_beta[1]) + 0.1
    levels_y_min = min(levels_path[:, 2].min(), levels_problem.true_beta[2]) - 0.0001
    levels_y_max = max(levels_path[:, 2].max(), levels_problem.true_beta[2]) + 0.0001
    
    ax1.set_xlim(levels_x_min, levels_x_max)
    ax1.set_ylim(levels_y_min, levels_y_max)
    ax1.set_title('CPI Levels (Chaos)', fontweight='bold', color='#e74c3c')
    ax1.set_xlabel('β₁ (Lag Coefficient)')
    ax1.set_ylabel('β₂ (Trend Coefficient)')
    ax1.grid(True, alpha=0.3)
    
    # For changes plot
    changes_x_min = min(changes_path[:, 1].min(), changes_problem.true_beta[1]) - 0.02
    changes_x_max = max(changes_path[:, 1].max(), changes_problem.true_beta[1]) + 0.02
    changes_y_min = min(changes_path[:, 2].min(), changes_problem.true_beta[2]) - 0.0002
    changes_y_max = max(changes_path[:, 2].max(), changes_problem.true_beta[2]) + 0.0002
    
    ax2.set_xlim(changes_x_min, changes_x_max)
    ax2.set_ylim(changes_y_min, changes_y_max)
    ax2.set_title('CPI Changes (Smooth)', fontweight='bold', color='#27ae60')
    ax2.set_xlabel('β₁ (Lag Coefficient)')
    ax2.set_ylabel('β₂ (Trend Coefficient)')
    ax2.grid(True, alpha=0.3)
    
    # Plot true minima - now guaranteed to be visible!
    ax1.plot(levels_problem.true_beta[1], levels_problem.true_beta[2], '*', 
             color='black', markersize=15, label='True Minimum')
    ax2.plot(changes_problem.true_beta[1], changes_problem.true_beta[2], '*', 
             color='black', markersize=15, label='True Minimum')
    
    # Initialize line objects
    levels_line, = ax1.plot([], [], 'o-', color='#e74c3c', linewidth=2, markersize=4)
    changes_line, = ax2.plot([], [], 'o-', color='#27ae60', linewidth=2, markersize=4)
    
    levels_point, = ax1.plot([], [], 'o', color='blue', markersize=8)
    changes_point, = ax2.plot([], [], 'o', color='blue', markersize=8)
    
    def animate(frame):
        # Update levels path
        if frame < len(levels_path):
            levels_line.set_data(levels_path[:frame+1, 1], levels_path[:frame+1, 2])
            levels_point.set_data([levels_path[frame, 1]], [levels_path[frame, 2]])
        
        # Update changes path
        if frame < len(changes_path):
            changes_line.set_data(changes_path[:frame+1, 1], changes_path[:frame+1, 2])
            changes_point.set_data([changes_path[frame, 1]], [changes_path[frame, 2]])
        
        return levels_line, changes_line, levels_point, changes_point
    
    max_frames = max(len(levels_path), len(changes_path))
    
    print(f"Animation will have {max_frames} frames for {optimizer_name}")
    if max_frames < 50:
        print(f"Warning: Only {max_frames} frames, animation may be very short")
    
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                 interval=150, blit=True, repeat=False)  # Slower interval for better visibility
    
    return fig, anim

# Main execution
if __name__ == "__main__":
    print("🚨 CPI Time Series Optimization Disaster Visualization 🚨")
    print("=" * 60)
    
    # Load or generate CPI data
    levels_data, changes_data = load_cpi_data()
    
    print("\n" + "=" * 60)
    print("Creating optimization comparison...")
    
    # Try different optimizers
    optimizers = ['adam', 'rmsprop', 'adagrad']  # Fixed order and removed invalid optimizer
    
    for optimizer in optimizers:
        print(f"\n--- Processing {optimizer.upper()} ---")
        fig, levels_result, changes_result = create_visualization(
            levels_data, changes_data, optimizer
        )
                
        print("Creating animated version...")
        fig_anim, anim = create_animated_comparison(levels_data, changes_data, optimizer)

        # Save animation as MP4 (commented out to avoid ffmpeg dependency issues)
        print(f"Saving animation as 'cpi_optimization_{optimizer}.mp4'...")
        anim.save(f'cpi_optimization_{optimizer}.mp4', writer='ffmpeg', fps=10, bitrate=1800)
        print("Animation saved!")