import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def generate_extended_market_data():
    """Generate realistic market data for higher-dimensional portfolio optimization"""
    np.random.seed(42)  # For reproducible results
    
    # Extended asset universe - 12 assets across different sectors
    assets = {
        # Equity ETFs
        'SPY': {'name': 'S&P 500 ETF', 'beta': 1.00, 'alpha': 0.02, 'sector': 'Market'},
        'QQQ': {'name': 'Nasdaq ETF', 'beta': 1.25, 'alpha': 0.08, 'sector': 'Tech'},
        'IWM': {'name': 'Russell 2000', 'beta': 1.15, 'alpha': 0.04, 'sector': 'SmallCap'},
        'EFA': {'name': 'International', 'beta': 0.85, 'alpha': 0.03, 'sector': 'Intl'},
        'EEM': {'name': 'Emerging Markets', 'beta': 1.35, 'alpha': 0.06, 'sector': 'EM'},
        'XLF': {'name': 'Financials', 'beta': 1.45, 'alpha': 0.05, 'sector': 'Finance'},
        
        # Alternative assets
        'TLT': {'name': 'Treasury Bonds', 'beta': -0.30, 'alpha': 0.01, 'sector': 'Bonds'},
        'GLD': {'name': 'Gold ETF', 'beta': -0.15, 'alpha': 0.03, 'sector': 'Commodity'},
        'VIX': {'name': 'Volatility ETF', 'beta': -2.50, 'alpha': 0.12, 'sector': 'Volatility'},
        'USO': {'name': 'Oil ETF', 'beta': 0.45, 'alpha': 0.07, 'sector': 'Energy'},
        'REZ': {'name': 'Real Estate', 'beta': 0.95, 'alpha': 0.04, 'sector': 'REIT'},
        'DBA': {'name': 'Agriculture', 'beta': 0.25, 'alpha': 0.05, 'sector': 'Agri'},
    }
    
    n_assets = len(assets)
    asset_names = list(assets.keys())
    
    # Expected returns (alpha)
    expected_returns = np.array([assets[name]['alpha'] for name in asset_names])
    
    # Beta coefficients
    betas = np.array([assets[name]['beta'] for name in asset_names])
    
    # Generate realistic correlation matrix for 12 assets
    np.random.seed(42)
    
    # Start with block structure based on sectors
    base_corr = np.eye(n_assets)
    
    # High correlation within equity group (first 6)
    for i in range(6):
        for j in range(6):
            if i != j:
                base_corr[i, j] = 0.6 + 0.3 * np.random.random()
    
    # Medium correlation for alternatives with equities
    for i in range(6):
        for j in range(6, n_assets):
            base_corr[i, j] = base_corr[j, i] = -0.1 + 0.4 * np.random.random()
    
    # Low correlation among alternatives
    for i in range(6, n_assets):
        for j in range(i+1, n_assets):
            base_corr[i, j] = base_corr[j, i] = -0.2 + 0.4 * np.random.random()
    
    # Ensure positive definite
    eigenvals, eigenvecs = np.linalg.eigh(base_corr)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
    correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Normalize to correlation matrix
    diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(correlations)))
    correlations = diag_inv_sqrt @ correlations @ diag_inv_sqrt
    
    # Individual volatilities (annualized)
    volatilities = np.array([0.16, 0.22, 0.20, 0.18, 0.28, 0.24,  # Equities
                           0.12, 0.18, 0.45, 0.35, 0.20, 0.25])   # Alternatives
    
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

def load_fred_data():
    """Load and process FRED economic data for benchmark construction"""
    try:
        import pandas as pd
        
        # Load 3-Month Treasury data
        fred_data = pd.read_csv('DGS3MO.csv')
        fred_data['observation_date'] = pd.to_datetime(fred_data['observation_date'])
        fred_data = fred_data.dropna()  # Remove missing values
        
        # Calculate some derived indicators
        fred_data['risk_free_rate'] = fred_data['DGS3MO'] / 100  # Convert to decimal
        
        # Simple moving averages for regime identification
        fred_data['rate_ma_30'] = fred_data['risk_free_rate'].rolling(30).mean()
        fred_data['rate_ma_90'] = fred_data['risk_free_rate'].rolling(90).mean()
        
        # Rate change momentum
        fred_data['rate_change'] = fred_data['risk_free_rate'].diff(5)  # 5-day change
        
        # Simple regime indicators
        fred_data['rising_rate_regime'] = (fred_data['risk_free_rate'] > fred_data['rate_ma_30']).astype(float)
        fred_data['high_rate_regime'] = (fred_data['risk_free_rate'] > fred_data['risk_free_rate'].quantile(0.7)).astype(float)
        
        return fred_data.fillna(method='ffill').fillna(method='bfill')
        
    except Exception as e:
        print(f"Could not load FRED data: {e}")
        # Return dummy data if file not available
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        return pd.DataFrame({
            'observation_date': dates,
            'risk_free_rate': 0.02,  # 2% constant
            'rising_rate_regime': 0.5,
            'high_rate_regime': 0.3,
            'rate_change': 0.0
        })

def create_fed_policy_benchmark_weights(fred_data, asset_names):
    """Create Fed Policy benchmark weights based on economic indicators"""
    
    # Get latest economic indicators (use most recent data point)
    latest = fred_data.iloc[-1]
    
    risk_free_rate = latest['risk_free_rate']
    rising_rates = latest['rising_rate_regime']
    high_rates = latest['high_rate_regime']
    rate_momentum = latest['rate_change']
    
    # Asset mapping for our universe
    asset_map = {
        'SPY': 'equity_broad',
        'QQQ': 'equity_growth', 
        'IWM': 'equity_small',
        'EFA': 'equity_intl',
        'EEM': 'equity_em',
        'XLF': 'equity_financial',
        'TLT': 'bonds_long',
        'GLD': 'commodity_gold',
        'VIX': 'volatility',
        'USO': 'commodity_energy',
        'REZ': 'real_estate',
        'DBA': 'commodity_agri'
    }
    
    n_assets = len(asset_names)
    weights = np.zeros(n_assets)
    
    # Fed Policy Logic:
    # 1. Base allocation depends on rate level
    # 2. Rising rates favor financials, hurt duration
    # 3. High rates favor cash/short duration
    # 4. Rate momentum affects growth vs value
    
    for i, asset in enumerate(asset_names):
        asset_type = asset_map.get(asset, 'other')
        
        if asset_type == 'equity_broad':  # SPY
            # Reduce equity in high rate environment
            weights[i] = 0.25 * (1 - 0.3 * high_rates)
            
        elif asset_type == 'equity_growth':  # QQQ  
            # Growth sensitive to rates
            weights[i] = 0.15 * (1 - 0.5 * rising_rates)
            
        elif asset_type == 'equity_financial':  # XLF
            # Banks benefit from rising rates
            weights[i] = 0.05 + 0.1 * rising_rates
            
        elif asset_type == 'bonds_long':  # TLT
            # Long bonds hurt by rising rates
            weights[i] = 0.20 * (1 - 0.7 * rising_rates)
            
        elif asset_type == 'commodity_gold':  # GLD
            # Gold as inflation hedge in high rate environment
            weights[i] = 0.05 + 0.1 * high_rates
            
        elif asset_type == 'volatility':  # VIX
            # Volatility protection in uncertain rate environment
            weights[i] = 0.02 + 0.05 * abs(rate_momentum) * 10
            
        elif asset_type in ['equity_intl', 'equity_em', 'equity_small']:
            # International and small cap
            weights[i] = 0.08 * (1 - 0.2 * high_rates)
            
        elif asset_type in ['commodity_energy', 'commodity_agri']:
            # Commodities as inflation hedge
            weights[i] = 0.05 + 0.05 * high_rates
            
        elif asset_type == 'real_estate':  # REITs
            # REITs sensitive to rates
            weights[i] = 0.08 * (1 - 0.4 * rising_rates)
            
        else:
            weights[i] = 0.05  # Default small allocation
    
    # Normalize to sum to 1
    weights = weights / np.sum(weights)
    
    # Apply some randomness to avoid perfect tracking
    noise = np.random.normal(0, 0.01, n_assets)
    weights = weights + noise
    weights = np.clip(weights, 0, 0.5)  # Position limits
    weights = weights / np.sum(weights)  # Renormalize
    
    return weights

def setup_portfolio_problem_variants():
    """Create different variants of the portfolio problem to explore perturbations"""
    data = generate_extended_market_data()
    n_assets = len(data['asset_names'])
    
    # Load FRED data for benchmark
    fred_data = load_fred_data()
    fed_benchmark_weights = create_fed_policy_benchmark_weights(fred_data, data['asset_names'])
    
    variants = {}
    
    # Base case
    variants['base'] = {
        'name': 'Base Case',
        'lambda_risk': 0.5,
        'position_limits': (-0.3, 0.3),
        'beta_target': 0.0,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'Standard beta-neutral with moderate risk aversion',
        'objective_type': 'alpha_maximization'
    }
    
    # Risk aversion variants
    variants['low_risk'] = {
        'name': 'Low Risk Aversion',
        'lambda_risk': 0.1,
        'position_limits': (-0.3, 0.3),
        'beta_target': 0.0,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'More aggressive - prioritizes returns over risk',
        'objective_type': 'alpha_maximization'
    }
    
    variants['high_risk'] = {
        'name': 'High Risk Aversion',
        'lambda_risk': 2.0,
        'position_limits': (-0.3, 0.3),
        'beta_target': 0.0,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'Conservative - strongly penalizes risk',
        'objective_type': 'alpha_maximization'
    }
    
    # Position limit variants
    variants['tight_limits'] = {
        'name': 'Tight Position Limits',
        'lambda_risk': 0.5,
        'position_limits': (-0.15, 0.15),
        'beta_target': 0.0,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'Constrained positions - max 15% per asset',
        'objective_type': 'alpha_maximization'
    }
    
    variants['loose_limits'] = {
        'name': 'Loose Position Limits',
        'lambda_risk': 0.5,
        'position_limits': (-0.8, 0.8),
        'beta_target': 0.0,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'Relaxed constraints - allows concentrated positions',
        'objective_type': 'alpha_maximization'
    }
    
    # Beta target variants
    variants['beta_positive'] = {
        'name': 'Positive Beta Target',
        'lambda_risk': 0.5,
        'position_limits': (-0.3, 0.3),
        'beta_target': 0.5,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'Target 50% market exposure',
        'objective_type': 'alpha_maximization'
    }
    
    variants['beta_flexible'] = {
        'name': 'Flexible Beta',
        'lambda_risk': 0.5,
        'position_limits': (-0.3, 0.3),
        'beta_target': 0.0,
        'beta_tolerance': 0.2,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'description': 'Allow beta deviation ±20%',
        'objective_type': 'alpha_maximization'
    }
    
    # Expected returns variants (alpha scenarios)
    boosted_returns = data['expected_returns'].copy()
    boosted_returns[[1, 4, 8]] *= 2  # Boost QQQ, EEM, VIX alphas
    
    variants['alpha_boost'] = {
        'name': 'Boosted High-Beta Assets',
        'lambda_risk': 0.5,
        'position_limits': (-0.3, 0.3),
        'beta_target': 0.0,
        'beta_tolerance': 0.0,
        'leverage_limit': 1.0,
        'expected_returns': boosted_returns,
        'description': 'Enhanced alpha for volatile assets',
        'objective_type': 'alpha_maximization'
    }
    
    # Benchmark tracking variants
    variants['fed_benchmark'] = {
        'name': 'Fed Policy Benchmark',
        'lambda_risk': 1.0,  # High penalty for tracking error
        'position_limits': (-0.2, 0.5),  # Allow long bias
        'beta_target': None,  # No beta constraint
        'beta_tolerance': None,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'benchmark_weights': fed_benchmark_weights,
        'description': 'Track Fed policy-driven allocation',
        'objective_type': 'benchmark_tracking'
    }
    
    # 60/40 Traditional benchmark
    traditional_60_40 = np.zeros(n_assets)
    spy_idx = data['asset_names'].index('SPY') if 'SPY' in data['asset_names'] else 0
    tlt_idx = data['asset_names'].index('TLT') if 'TLT' in data['asset_names'] else 6
    traditional_60_40[spy_idx] = 0.6
    traditional_60_40[tlt_idx] = 0.4
    
    variants['60_40_tracker'] = {
        'name': '60/40 Tracker',
        'lambda_risk': 2.0,  # Very high penalty for tracking error
        'position_limits': (-0.1, 0.8),  # Allow concentrated positions
        'beta_target': None,  # No beta constraint
        'beta_tolerance': None,
        'leverage_limit': 1.0,
        'expected_returns': data['expected_returns'],
        'benchmark_weights': traditional_60_40,
        'description': 'Track classic 60% equity / 40% bonds',
        'objective_type': 'benchmark_tracking'
    }
    
    return variants, data

def solve_portfolio_variant(variant_config, data):
    """Solve a specific portfolio variant"""
    n_assets = len(data['asset_names'])
    
    # Decision variables
    w = cp.Variable(n_assets)
    
    # Determine objective type
    objective_type = variant_config.get('objective_type', 'alpha_maximization')
    
    if objective_type == 'benchmark_tracking':
        # Benchmark tracking objective: minimize tracking error
        benchmark_weights = variant_config['benchmark_weights']
        tracking_error = cp.sum_squares(w - benchmark_weights)
        objective = cp.Minimize(tracking_error)
        
    else:
        # Alpha maximization objective (default)
        expected_return = variant_config['expected_returns'].T @ w
        risk_penalty = cp.quad_form(w, data['cov_matrix'])
        objective = cp.Maximize(expected_return - variant_config['lambda_risk'] * risk_penalty)
    
    # Constraints
    constraints = []
    
    # Fully invested
    constraints.append(cp.sum(w) == variant_config['leverage_limit'])
    
    # Position limits
    min_pos, max_pos = variant_config['position_limits']
    constraints.append(w >= min_pos)
    constraints.append(w <= max_pos)
    
    # Beta constraint (only for alpha maximization variants)
    if variant_config.get('beta_target') is not None:
        beta_exposure = data['betas'].T @ w
        if variant_config['beta_tolerance'] == 0.0:
            # Exact beta target
            constraints.append(beta_exposure == variant_config['beta_target'])
        else:
            # Beta range
            constraints.append(beta_exposure >= variant_config['beta_target'] - variant_config['beta_tolerance'])
            constraints.append(beta_exposure <= variant_config['beta_target'] + variant_config['beta_tolerance'])
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        if problem.status != 'optimal':
            problem.solve(solver=cp.CLARABEL, verbose=False)
    except:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    
    if problem.status == 'optimal':
        return w.value, problem.value
    else:
        print(f"Warning: Problem not solved optimally. Status: {problem.status}")
        return None, None

def simulate_ecos_path_variant(optimal_weights, variant_config, data, max_iters=40):
    """Simulate ECOS path for a variant with different constraints/objectives"""
    path = []
    objectives = []
    constraint_violations = []
    
    n_assets = len(optimal_weights)
    
    # Start from equal weights (or feasible starting point)
    current = np.ones(n_assets) / n_assets  # Equal weights
    
    # Adjust to satisfy position limits
    min_pos, max_pos = variant_config['position_limits']
    current = np.clip(current, min_pos, max_pos)
    current = current / np.sum(current) * variant_config['leverage_limit']
    
    for i in range(max_iters):
        t = i / (max_iters - 1)
        
        # Direction toward optimal
        direction = optimal_weights - current
        
        # Adaptive step size based on iteration and constraint proximity
        base_step = 0.15 * (1 - 0.6 * t)
        
        # Check constraint proximity
        min_dist_to_bound = min(
            np.min(current - min_pos),
            np.min(max_pos - current)
        )
        
        # Reduce step size if near boundaries
        boundary_factor = max(0.1, min(1.0, min_dist_to_bound / 0.1))
        step_size = base_step * boundary_factor
        
        # Take step
        candidate = current + step_size * direction
        
        # Project to constraints
        candidate = project_to_variant_constraints(candidate, variant_config, data)
        
        path.append(candidate.copy())
        
        # Calculate metrics
        obj_val = calculate_variant_objective(candidate, variant_config, data)
        objectives.append(obj_val)
        
        violations = calculate_constraint_violations(candidate, variant_config, data)
        constraint_violations.append(violations)
        
        current = candidate
        
        # Convergence check
        if np.linalg.norm(current - optimal_weights) < 1e-4:
            break
    
    return np.array(path), objectives, constraint_violations

def project_to_variant_constraints(weights, variant_config, data):
    """Project weights to satisfy variant constraints"""
    n_assets = len(weights)
    w_proj = weights.copy()
    
    # Apply position limits
    min_pos, max_pos = variant_config['position_limits']
    w_proj = np.clip(w_proj, min_pos, max_pos)
    
    # Project to leverage constraint (sum = leverage_limit)
    w_proj = w_proj / np.sum(w_proj) * variant_config['leverage_limit']
    
    # Project to beta constraint (only if beta_target is specified)
    beta_target = variant_config.get('beta_target')
    beta_tolerance = variant_config.get('beta_tolerance')
    
    if beta_target is not None and beta_tolerance is not None:
        betas = data['betas']
        current_beta = betas @ w_proj
        
        if beta_tolerance == 0.0:
            # Exact beta constraint using Lagrange multipliers
            ones = np.ones(n_assets)
            A = np.vstack([betas, ones])
            b = np.array([beta_target, variant_config['leverage_limit']])
            
            try:
                AAT_inv = np.linalg.inv(A @ A.T)
                correction = A.T @ AAT_inv @ (A @ w_proj - b)
                w_proj = w_proj - correction
            except:
                # Fallback: adjust largest beta contributor
                beta_error = current_beta - beta_target
                max_beta_idx = np.argmax(np.abs(betas))
                w_proj[max_beta_idx] -= beta_error / betas[max_beta_idx]
                w_proj = w_proj / np.sum(w_proj) * variant_config['leverage_limit']
        else:
            # Beta range constraint
            if current_beta > beta_target + beta_tolerance:
                # Reduce high-beta exposure
                excess = current_beta - (beta_target + beta_tolerance)
                high_beta_assets = betas > np.median(betas)
                if np.any(high_beta_assets):
                    reduction_factor = excess / np.sum(betas[high_beta_assets] * w_proj[high_beta_assets])
                    w_proj[high_beta_assets] *= (1 - 0.1 * reduction_factor)
            elif current_beta < beta_target - beta_tolerance:
                # Increase high-beta exposure
                deficit = (beta_target - beta_tolerance) - current_beta
                high_beta_assets = betas > np.median(betas)
                if np.any(high_beta_assets):
                    increase_factor = deficit / np.sum(betas[high_beta_assets] * w_proj[high_beta_assets])
                    w_proj[high_beta_assets] *= (1 + 0.1 * increase_factor)
            
            # Renormalize
            w_proj = w_proj / np.sum(w_proj) * variant_config['leverage_limit']
    
    # Final position limit check
    w_proj = np.clip(w_proj, min_pos, max_pos)
    
    return w_proj

def calculate_variant_objective(weights, variant_config, data):
    """Calculate objective function value for a variant"""
    objective_type = variant_config.get('objective_type', 'alpha_maximization')
    
    if objective_type == 'benchmark_tracking':
        # Tracking error objective
        benchmark_weights = variant_config['benchmark_weights']
        tracking_error = np.sum((weights - benchmark_weights)**2)
        return -tracking_error  # Negative because we want to minimize tracking error
    else:
        # Alpha maximization objective
        expected_return = variant_config['expected_returns'] @ weights
        risk = weights.T @ data['cov_matrix'] @ weights
        return expected_return - variant_config['lambda_risk'] * risk

def calculate_constraint_violations(weights, variant_config, data):
    """Calculate constraint violations"""
    violations = {}
    
    # Sum constraint
    violations['sum'] = abs(np.sum(weights) - variant_config['leverage_limit'])
    
    # Position limits
    min_pos, max_pos = variant_config['position_limits']
    violations['position_lower'] = np.sum(np.maximum(0, min_pos - weights))
    violations['position_upper'] = np.sum(np.maximum(0, weights - max_pos))
    
    # Beta constraint (only if beta_target is specified)
    beta_target = variant_config.get('beta_target')
    beta_tolerance = variant_config.get('beta_tolerance')
    
    if beta_target is not None and beta_tolerance is not None:
        current_beta = data['betas'] @ weights
        
        if beta_tolerance == 0.0:
            violations['beta'] = abs(current_beta - beta_target)
        else:
            if current_beta > beta_target + beta_tolerance:
                violations['beta'] = current_beta - (beta_target + beta_tolerance)
            elif current_beta < beta_target - beta_tolerance:
                violations['beta'] = (beta_target - beta_tolerance) - current_beta
            else:
                violations['beta'] = 0.0
    else:
        # No beta constraint for benchmark tracking variants
        violations['beta'] = 0.0
    
    return violations

def animate_portfolio_variants():
    """Create animation comparing different portfolio variants"""
    print("Generating portfolio variant animations...")
    
    variants, data = setup_portfolio_problem_variants()
    
    # Select interesting variants to compare - include benchmark tracking
    selected_variants = ['base', 'low_risk', 'high_risk', 'tight_limits', 'fed_benchmark', '60_40_tracker']
    
    # Solve all variants
    solutions = {}
    paths = {}
    
    for variant_name in selected_variants:
        variant_config = variants[variant_name]
        print(f"Solving {variant_config['name']}...")
        
        weights, obj_val = solve_portfolio_variant(variant_config, data)
        if weights is not None:
            solutions[variant_name] = (weights, obj_val)
            path, objectives, violations = simulate_ecos_path_variant(weights, variant_config, data)
            paths[variant_name] = (path, objectives, violations)
        else:
            print(f"Failed to solve {variant_name}")
    
    # Create comparative animation
    n_variants = len(selected_variants)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_variants))
    
    max_frames = max([len(paths[v][0]) for v in selected_variants if v in paths])
    
    def animate(frame):
        fig.suptitle(f'Portfolio Optimization Variants - Iteration {frame+1}/{max_frames}', 
                    fontsize=16, fontweight='bold')
        
        for i, variant_name in enumerate(selected_variants):
            if variant_name not in paths:
                continue
                
            ax = axes[i]
            ax.clear()
            
            path, objectives, violations = paths[variant_name]
            variant_config = variants[variant_name]
            
            if frame >= len(path):
                frame_idx = len(path) - 1
            else:
                frame_idx = frame
                
            current_weights = path[frame_idx]
            
            # Plot portfolio weights
            asset_positions = np.arange(len(data['asset_names']))
            weight_colors = ['green' if w > 0 else 'red' for w in current_weights]
            bars = ax.bar(asset_positions, current_weights, color=weight_colors, alpha=0.7)
            
            # Highlight constraint boundaries
            min_pos, max_pos = variant_config['position_limits']
            ax.axhline(y=max_pos, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=min_pos, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            
            # For benchmark tracking variants, show target weights
            if variant_config.get('objective_type') == 'benchmark_tracking':
                benchmark_weights = variant_config['benchmark_weights']
                ax.plot(asset_positions, benchmark_weights, 'ko-', alpha=0.5, linewidth=2, 
                       markersize=4, label='Benchmark Target')
                ax.legend(loc='upper right', fontsize=8)
            
            # Highlight assets near limits
            for j, bar in enumerate(bars):
                if abs(current_weights[j] - max_pos) < 0.05 or abs(current_weights[j] - min_pos) < 0.05:
                    bar.set_edgecolor('orange')
                    bar.set_linewidth(2)
            
            ax.set_title(f"{variant_config['name']}\nλ={variant_config['lambda_risk']}, "
                        f"Limits=[{min_pos:.2f}, {max_pos:.2f}]", fontsize=10, fontweight='bold')
            ax.set_ylabel('Weight', fontsize=9)
            ax.set_xticks(asset_positions)
            ax.set_xticklabels(data['asset_names'], rotation=45, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
            # Add metrics
            if frame_idx < len(objectives):
                beta_exposure = data['betas'] @ current_weights
                expected_alpha = variant_config['expected_returns'] @ current_weights
                portfolio_risk = np.sqrt(current_weights.T @ data['cov_matrix'] @ current_weights)
                
                # Add tracking error for benchmark variants
                if variant_config.get('objective_type') == 'benchmark_tracking':
                    benchmark_weights = variant_config['benchmark_weights']
                    tracking_error = np.sqrt(np.sum((current_weights - benchmark_weights)**2))
                    metrics_text = f"β={beta_exposure:.3f}\nα={expected_alpha:.3f}\nσ={portfolio_risk:.3f}\nTE={tracking_error:.3f}"
                else:
                    metrics_text = f"β={beta_exposure:.3f}\nα={expected_alpha:.3f}\nσ={portfolio_risk:.3f}"
                
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Create and save animation
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                 interval=400, repeat=True, blit=False)
    
    filename = 'portfolio_variants_comparison.mp4'
    print(f"Saving {filename}...")
    writer = animation.FFMpegWriter(fps=3, metadata=dict(artist='Portfolio Variants'))
    anim.save(filename, writer=writer, dpi=100)
    plt.close()
    
    print(f"Animation saved as {filename}")
    
    # Print summary analysis
    print("\n=== VARIANT ANALYSIS ===")
    for variant_name in selected_variants:
        if variant_name in solutions:
            weights, obj_val = solutions[variant_name]
            variant_config = variants[variant_name]
            
            beta_exposure = data['betas'] @ weights
            expected_alpha = variant_config['expected_returns'] @ weights
            portfolio_risk = np.sqrt(weights.T @ data['cov_matrix'] @ weights)
            max_long = np.max(weights)
            max_short = np.min(weights)
            
            print(f"\n{variant_config['name']}:")
            print(f"  Description: {variant_config['description']}")
            print(f"  Portfolio Beta: {beta_exposure:.4f}")
            print(f"  Expected Alpha: {expected_alpha:.4f}")
            print(f"  Portfolio Risk: {portfolio_risk:.4f}")
            print(f"  Max Long Position: {max_long:.3f}")
            print(f"  Max Short Position: {max_short:.3f}")
            print(f"  Objective Value: {obj_val:.4f}")
            
            # Show top 3 positions
            sorted_indices = np.argsort(np.abs(weights))[::-1]
            print(f"  Top 3 positions:")
            for i in range(3):
                idx = sorted_indices[i]
                print(f"    {data['asset_names'][idx]}: {weights[idx]:6.3f}")

def main():
    """Main function for enhanced portfolio optimization"""
    print("=== Enhanced Portfolio Optimization with Perturbations ===")
    print("Extended universe: 12 assets across equity, fixed income, commodities")
    print("Exploring different penalty functions, constraints, and objectives")
    print()
    
    data = generate_extended_market_data()
    print("Asset Universe:")
    for name in data['asset_names']:
        info = data['assets_info'][name]
        print(f"  {name} ({info['sector']}): β = {info['beta']:5.2f}, α = {info['alpha']:5.3f}")
    print()
    
    # Generate animation
    animate_portfolio_variants()
    
    print("\nPortfolio variant analysis complete!")
    print("Files created:")
    print("- portfolio_variants_comparison.mp4")
    print("\nKey insights:")
    print("- Risk aversion (λ) dramatically affects position concentration")
    print("- Position limits constrain the solver's ability to exploit alpha")
    print("- Beta targets shift the entire portfolio allocation strategy")
    print("- Alpha perturbations can create concentrated bets in high-alpha assets")

if __name__ == "__main__":
    main()