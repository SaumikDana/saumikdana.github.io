import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def generate_market_data():
    """Generate realistic market data for portfolio optimization"""
    np.random.seed(42)  # For reproducible results
    
    # Asset names and characteristics
    assets = {
        'SPY': {'name': 'S&P 500 ETF', 'beta': 1.00, 'alpha': 0.02},      # Market ETF
        'QQQ': {'name': 'Nasdaq ETF', 'beta': 1.25, 'alpha': 0.08},       # High beta tech
        'IWM': {'name': 'Russell 2000', 'beta': 1.15, 'alpha': 0.04},     # Small cap
        'TLT': {'name': 'Treasury Bonds', 'beta': -0.30, 'alpha': 0.01},  # Negative beta
        'GLD': {'name': 'Gold ETF', 'beta': -0.15, 'alpha': 0.03},        # Negative beta
        'VIX': {'name': 'Volatility ETF', 'beta': -2.50, 'alpha': 0.12},  # Very negative beta
    }
    
    n_assets = len(assets)
    asset_names = list(assets.keys())
    
    # Expected returns (alpha)
    expected_returns = np.array([assets[name]['alpha'] for name in asset_names])
    
    # Beta coefficients
    betas = np.array([assets[name]['beta'] for name in asset_names])
    
    # Generate realistic covariance matrix
    # Start with random correlation matrix
    correlations = np.array([
        [1.00, 0.85, 0.75, -0.40, -0.20, -0.60],  # SPY
        [0.85, 1.00, 0.70, -0.35, -0.15, -0.55],  # QQQ  
        [0.75, 0.70, 1.00, -0.30, -0.10, -0.50],  # IWM
        [-0.40, -0.35, -0.30, 1.00, 0.30, 0.25],  # TLT
        [-0.20, -0.15, -0.10, 0.30, 1.00, 0.15],  # GLD
        [-0.60, -0.55, -0.50, 0.25, 0.15, 1.00]   # VIX
    ])
    
    # Individual volatilities (annualized)
    volatilities = np.array([0.16, 0.22, 0.20, 0.12, 0.18, 0.45])
    
    # Convert to covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlations
    
    return {
        'asset_names': asset_names,
        'expected_returns': expected_returns,
        'betas': betas,
        'cov_matrix': cov_matrix,
        'volatilities': volatilities,
        'assets_info': assets
    }

def setup_portfolio_problem():
    """
    Beta-neutral portfolio optimization:
    maximize expected_returns^T * w - λ * w^T * Σ * w  (maximize alpha, minimize risk)
    subject to:
        betas^T * w = 0        (beta neutral constraint)
        sum(w) = 1             (fully invested)
        w >= -0.5              (position limits)
        w <= 0.5               (position limits)
    """
    data = generate_market_data()
    n_assets = len(data['asset_names'])
    
    # Decision variables: portfolio weights
    w = cp.Variable(n_assets)
    
    # Risk aversion parameter
    lambda_risk = 0.5
    
    # Objective: maximize expected return minus risk penalty
    expected_return = data['expected_returns'].T @ w
    risk_penalty = cp.quad_form(w, data['cov_matrix'])
    objective = cp.Maximize(expected_return - lambda_risk * risk_penalty)
    
    # Constraints
    constraints = [
        data['betas'].T @ w == 0,  # Beta neutral
        cp.sum(w) == 1,            # Fully invested
        w >= -0.5,                 # Short position limits
        w <= 0.5                   # Long position limits
    ]
    
    return cp.Problem(objective, constraints), w, data

def solve_portfolio_with_path(solver_name, max_iters=50):
    """Solve portfolio problem and simulate solver path"""
    
    problem, w_var, data = setup_portfolio_problem()
    
    # Get true optimal solution
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        if problem.status != 'optimal':
            problem.solve(solver=cp.OSQP, verbose=False)
        optimal_weights = w_var.value.copy() if problem.status == 'optimal' else None
    except:
        optimal_weights = None
    
    if optimal_weights is None:
        # Fallback: construct a reasonable beta-neutral portfolio
        print("Using fallback optimal solution")
        optimal_weights = np.array([0.2, 0.15, 0.1, 0.3, 0.15, 0.1])
        # Adjust to be beta neutral
        beta_exposure = data['betas'] @ optimal_weights
        optimal_weights[0] -= beta_exposure / data['betas'][0]  # Adjust SPY to neutralize beta
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormalize
    
    print(f"Optimal portfolio weights:")
    for i, name in enumerate(data['asset_names']):
        print(f"  {name}: {optimal_weights[i]:6.3f}")
    
    beta_check = data['betas'] @ optimal_weights
    expected_alpha = data['expected_returns'] @ optimal_weights
    print(f"Portfolio beta: {beta_check:.6f}")
    print(f"Expected alpha: {expected_alpha:.4f}")
    
    if solver_name == 'ECOS':
        return simulate_ecos_portfolio_path(optimal_weights, data, max_iters)
    else:
        return simulate_osqp_portfolio_path(optimal_weights, data, max_iters)

def project_to_constraints(weights, data):
    """Project weights to satisfy constraints"""
    n_assets = len(weights)
    
    # Start with current weights
    w_proj = weights.copy()
    
    # Apply box constraints
    w_proj = np.clip(w_proj, -0.5, 0.5)
    
    # Project to sum = 1 constraint
    w_proj = w_proj / np.sum(w_proj)
    
    # Project to beta = 0 constraint using Lagrange multipliers
    # min ||w - w_proj||^2 subject to beta^T w = 0, 1^T w = 1
    betas = data['betas']
    ones = np.ones(n_assets)
    
    # Set up constraint matrix A = [betas; ones]
    A = np.vstack([betas, ones])
    b = np.array([0, 1])  # beta = 0, sum = 1
    
    # Solve: w* = w_proj - A^T (A A^T)^-1 (A w_proj - b)
    try:
        AAT_inv = np.linalg.inv(A @ A.T)
        correction = A.T @ AAT_inv @ (A @ w_proj - b)
        w_proj = w_proj - correction
    except:
        # Fallback: simple adjustment
        beta_error = betas @ w_proj
        w_proj[0] -= beta_error / betas[0]  # Adjust SPY
        w_proj = w_proj / np.sum(w_proj)  # Renormalize
    
    return w_proj

def simulate_ecos_portfolio_path(optimal_weights, data, max_iters):
    """Simulate ECOS interior point method for portfolio optimization"""
    path = []
    
    # Start from a feasible but suboptimal point
    current = np.array([0.3, 0.2, 0.15, 0.2, 0.1, 0.05])
    current = project_to_constraints(current, data)
    
    for i in range(max_iters):
        t = i / (max_iters - 1)
        
        # Interior point: move toward optimal while maintaining strict feasibility
        direction = optimal_weights - current
        
        # Adaptive step size with barrier effects
        base_step = 0.12 * (1 - 0.7 * t)
        
        candidate = current + base_step * direction
        
        # Barrier function effects: stay away from constraint boundaries
        margin = 0.05 * np.exp(-2 * t)
        
        # Ensure we don't get too close to position limits
        candidate = np.clip(candidate, -0.5 + margin, 0.5 - margin)
        
        # Project to equality constraints with margin considerations
        candidate = project_to_constraints(candidate, data)
        
        # Additional check: ensure we're not violating constraints by too much
        beta_violation = abs(data['betas'] @ candidate)
        sum_violation = abs(np.sum(candidate) - 1)
        
        if beta_violation > 0.01 or sum_violation > 0.01:
            candidate = project_to_constraints(candidate, data)
        
        path.append(candidate.copy())
        current = candidate
        
        # Check convergence
        if np.linalg.norm(current - optimal_weights) < 1e-4:
            break
    
    return np.array(path), data

def simulate_osqp_portfolio_path(optimal_weights, data, max_iters):
    """Simulate OSQP active set method for portfolio optimization"""
    path = []
    
    # Start from a different point
    current = np.array([0.25, 0.25, 0.1, 0.25, 0.1, 0.05])
    current = project_to_constraints(current, data)
    
    # Phase 1: Identify active constraints (position limits)
    phase1_steps = 20
    for i in range(phase1_steps):
        t = i / (phase1_steps - 1)
        
        # Move toward a corner of the feasible region
        # Target: some assets at their limits
        target = np.array([0.5, 0.3, -0.2, 0.3, 0.1, 0.0])
        target = project_to_constraints(target, data)
        
        next_point = current + t * 0.3 * (target - current)
        next_point = project_to_constraints(next_point, data)
        
        path.append(next_point.copy())
        current = next_point
    
    # Phase 2: Walk along active constraints toward optimum
    phase2_steps = 20
    for i in range(phase2_steps):
        t = i / (phase2_steps - 1)
        
        # Move toward optimal while respecting active constraints
        direction = optimal_weights - current
        step_size = 0.1
        
        candidate = current + step_size * direction
        
        # Identify which constraints are "active" (close to bounds)
        active_long = candidate > 0.45   # Close to long limit
        active_short = candidate < -0.45  # Close to short limit
        
        # For active constraints, limit movement
        if np.any(active_long):
            candidate[active_long] = np.minimum(candidate[active_long], 0.5)
        if np.any(active_short):
            candidate[active_short] = np.maximum(candidate[active_short], -0.5)
        
        # Project back to equality constraints
        candidate = project_to_constraints(candidate, data)
        
        path.append(candidate.copy())
        current = candidate
    
    # Phase 3: Final convergence
    for i in range(10):
        t = i / 9
        final_point = current + t * (optimal_weights - current)
        final_point = project_to_constraints(final_point, data)
        path.append(final_point.copy())
    
    return np.array(path), data

def portfolio_objective_value(weights, data):
    """Calculate portfolio objective value"""
    expected_return = data['expected_returns'] @ weights
    risk = weights.T @ data['cov_matrix'] @ weights
    lambda_risk = 0.5
    return expected_return - lambda_risk * risk

def plot_portfolio_setup(ax, data, title, solver_color):
    """Setup portfolio optimization plot"""
    ax.clear()
    ax.set_xlabel('Long/Short Positions', fontsize=12)
    ax.set_ylabel('Portfolio Weights', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', color=solver_color)
    ax.grid(True, alpha=0.3)
    
    # Set up x-axis for assets
    asset_positions = np.arange(len(data['asset_names']))
    ax.set_xticks(asset_positions)
    ax.set_xticklabels(data['asset_names'], rotation=45)
    
    # Draw constraint lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Long Limit (50%)')
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, label='Short Limit (-50%)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Show asset characteristics in legend
    legend_text = []
    for i, name in enumerate(data['asset_names']):
        beta = data['betas'][i]
        alpha = data['expected_returns'][i]
        legend_text.append(f'{name}: β={beta:.2f}, α={alpha:.3f}')
    
    ax.text(0.02, 0.98, '\n'.join(legend_text), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_ylim(-0.6, 0.6)
    ax.legend(loc='upper right')

def animate_portfolio_solver(solver_name, save_filename):
    """Create portfolio optimization animation"""
    print(f"Generating {solver_name} portfolio optimization animation...")
    
    path, data = solve_portfolio_with_path(solver_name)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)
    
    colors = {'ECOS': '#1f77b4', 'OSQP': '#ff7f0e'}
    solver_color = colors[solver_name]
    
    def animate(frame):
        if frame >= len(path):
            frame = len(path) - 1
        
        current_weights = path[frame]
        
        # Top plot: Portfolio weights
        behavior = "Interior Point (Barrier Methods)" if solver_name == 'ECOS' else "Active Set (Constraint Walking)"
        plot_portfolio_setup(ax1, data, f'{solver_name} - {behavior}', solver_color)
        
        # Plot current portfolio as bars
        asset_positions = np.arange(len(data['asset_names']))
        colors_bars = ['green' if w > 0 else 'red' for w in current_weights]
        bars = ax1.bar(asset_positions, current_weights, color=colors_bars, alpha=0.7)
        
        # Highlight current position
        for i, bar in enumerate(bars):
            if abs(current_weights[i]) > 0.45:  # Near constraint boundary
                bar.set_edgecolor('orange')
                bar.set_linewidth(3)
        
        # Bottom plot: Objective function evolution
        ax2.clear()
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Objective Value (Alpha - Risk)', fontsize=12)
        ax2.set_title('Objective Function Evolution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Calculate objective values for all iterations up to current
        iterations = list(range(min(frame + 1, len(path))))
        objectives = []
        
        for i in iterations:
            weights = path[i]
            obj_val = portfolio_objective_value(weights, data)
            objectives.append(obj_val)
        
        # Plot objective function evolution
        if len(iterations) > 1:
            # Main objective line
            ax2.plot(iterations, objectives, 'g-', linewidth=4, 
                    label=f'Objective Value: {objectives[-1]:.4f}', marker='o', markersize=4)
            
            # Mark the maximum achieved so far
            max_obj = max(objectives)
            max_idx = objectives.index(max_obj)
            ax2.plot(max_idx, max_obj, 'r*', markersize=12, 
                    label=f'Best so far: {max_obj:.4f} (iter {max_idx})')
            
            # Add trend arrow
            if len(objectives) >= 3:
                recent_trend = objectives[-1] - objectives[-3]
                trend_text = "↗ Improving" if recent_trend > 0.001 else "↘ Declining" if recent_trend < -0.001 else "→ Stable"
                ax2.text(0.98, 0.95, f'Trend: {trend_text}', 
                        transform=ax2.transAxes, fontsize=11, 
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if recent_trend > 0 else 'lightcoral' if recent_trend < -0.001 else 'lightgray', alpha=0.8))
            
            ax2.legend(loc='lower right')
            
            current_weights = path[frame]
        
        # Add iteration info
        beta_exposure = data['betas'] @ current_weights
        expected_alpha = data['expected_returns'] @ current_weights
        portfolio_risk = np.sqrt(current_weights.T @ data['cov_matrix'] @ current_weights)
        
        info_text = f"""Iteration: {frame + 1}
Portfolio Beta: {beta_exposure:.6f}
Expected Alpha: {expected_alpha:.4f}
Portfolio Risk: {portfolio_risk:.4f}
Sum of Weights: {np.sum(current_weights):.6f}"""
        
        ax1.text(0.98, 0.02, info_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(path), 
                                 interval=300, repeat=True, blit=False)
    
    # Save animation
    print(f"Saving {save_filename}...")
    writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Portfolio Optimization'))
    anim.save(save_filename, writer=writer, dpi=120)
    plt.close()
    
    print(f"Portfolio animation saved as {save_filename}")
    return path, data

def create_portfolio_comparison():
    """Create side-by-side portfolio optimization comparison"""
    print("Generating portfolio optimization comparison...")
    
    ecos_path, ecos_data = solve_portfolio_with_path('ECOS')
    osqp_path, osqp_data = solve_portfolio_with_path('OSQP')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    max_frames = max(len(ecos_path), len(osqp_path))
    
    def animate(frame):
        # ECOS plots
        if frame < len(ecos_path):
            current_ecos = ecos_path[frame]
            
            # ECOS portfolio weights
            plot_portfolio_setup(axes[0,0], ecos_data, 'ECOS - Interior Point', '#1f77b4')
            asset_positions = np.arange(len(ecos_data['asset_names']))
            colors_bars = ['green' if w > 0 else 'red' for w in current_ecos]
            axes[0,0].bar(asset_positions, current_ecos, color=colors_bars, alpha=0.7)
            
        # OSQP plots
        if frame < len(osqp_path):
            current_osqp = osqp_path[frame]
            
            # OSQP portfolio weights  
            plot_portfolio_setup(axes[0,1], osqp_data, 'OSQP - Active Set', '#ff7f0e')
            asset_positions = np.arange(len(osqp_data['asset_names']))
            colors_bars = ['green' if w > 0 else 'red' for w in current_osqp]
            axes[0,1].bar(asset_positions, current_osqp, color=colors_bars, alpha=0.7)
        
        # Convergence comparison - show both key constraints
        axes[1,0].clear()
        axes[1,1].clear()
        
        # ECOS convergence
        if len(ecos_path) > frame:
            iterations_ecos = list(range(frame + 1))
            ecos_betas = [abs(ecos_data['betas'] @ ecos_path[i]) for i in iterations_ecos]
            ecos_sums = [abs(np.sum(ecos_path[i]) - 1.0) for i in iterations_ecos]
            
            axes[1,0].plot(iterations_ecos, ecos_betas, 'r-', linewidth=3, 
                          label=f'Beta Violation: {ecos_betas[-1]:.6f}', marker='o', markersize=3)
            axes[1,0].plot(iterations_ecos, ecos_sums, 'b-', linewidth=3, 
                          label=f'Sum Violation: {ecos_sums[-1]:.6f}', marker='s', markersize=3)
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            axes[1,0].set_title('ECOS: Constraint Violations', fontsize=12)
            axes[1,0].set_ylabel('Violation (should → 0)', fontsize=10)
            axes[1,0].set_xlabel('Iteration', fontsize=10)
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
            axes[1,0].set_ylim(bottom=0)
        
        # OSQP convergence
        if len(osqp_path) > frame:
            iterations_osqp = list(range(frame + 1))
            osqp_betas = [abs(osqp_data['betas'] @ osqp_path[i]) for i in iterations_osqp]
            osqp_sums = [abs(np.sum(osqp_path[i]) - 1.0) for i in iterations_osqp]
            
            axes[1,1].plot(iterations_osqp, osqp_betas, 'r-', linewidth=3, 
                          label=f'Beta Violation: {osqp_betas[-1]:.6f}', marker='o', markersize=3)
            axes[1,1].plot(iterations_osqp, osqp_sums, 'b-', linewidth=3, 
                          label=f'Sum Violation: {osqp_sums[-1]:.6f}', marker='s', markersize=3)
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            axes[1,1].set_title('OSQP: Constraint Violations', fontsize=12)
            axes[1,1].set_ylabel('Violation (should → 0)', fontsize=10)
            axes[1,1].set_xlabel('Iteration', fontsize=10)
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
            axes[1,1].set_ylim(bottom=0)
        
        fig.suptitle(f'Portfolio Optimization: ECOS vs OSQP - Frame {frame+1}/{max_frames}', 
                    fontsize=16, fontweight='bold')
    
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                 interval=400, repeat=True, blit=False)
    
    filename = 'portfolio_ecos_vs_osqp.mp4'
    print(f"Saving {filename}...")
    writer = animation.FFMpegWriter(fps=3, metadata=dict(artist='Portfolio Optimization Comparison'))
    anim.save(filename, writer=writer, dpi=100)
    plt.close()
    
    print(f"Portfolio comparison saved as {filename}")

def main():
    """Main function for portfolio optimization"""
    print("=== Beta-Neutral Portfolio Optimization ===")
    print("Objective: Maximize alpha while maintaining beta neutrality")
    print()
    print("Assets:")
    data = generate_market_data()
    for name in data['asset_names']:
        info = data['assets_info'][name]
        print(f"  {name} ({info['name']}): β = {info['beta']:5.2f}, α = {info['alpha']:5.3f}")
    print()
    print("Constraints:")
    print("  • Portfolio beta = 0 (market neutral)")
    print("  • Sum of weights = 1 (fully invested)")
    print("  • Position limits: -50% ≤ weight ≤ 50%")
    print("  • Objective: maximize α - λ×risk")
    print()
    
    # Generate individual animations
    ecos_path, _ = animate_portfolio_solver('ECOS', 'portfolio_ecos_interior.mp4')
    osqp_path, _ = animate_portfolio_solver('OSQP', 'portfolio_osqp_active_set.mp4')
    
    # Generate comparison
    create_portfolio_comparison()
    
    print("\nPortfolio optimization animations complete!")
    print("Files created:")
    print("- portfolio_ecos_interior.mp4")
    print("- portfolio_osqp_active_set.mp4")
    print("- portfolio_ecos_vs_osqp.mp4")
    
    print(f"\nPath Analysis:")
    print(f"ECOS iterations: {len(ecos_path)}")
    print(f"OSQP iterations: {len(osqp_path)}")

if __name__ == "__main__":
    main()