let currentChart = null;
let currentDistribution = 'binomial';

const distributions = {

    binomial: {
        name: 'Binomial Distribution',
        description: 'Models the number of successes in n independent trials with probability p.',
        applications: [
            'Quality control: Number of defective items in a batch',
            'Medical trials: Success rate of treatments',
            'Marketing: Click-through rates in advertising campaigns',
            'Sports: Number of wins in a season',
            'Finance: Number of profitable trades out of total trades'
        ],
        parameters: [
            {name: 'n', label: 'Number of trials (n)', min: 1, max: 50, default: 20, step: 1},
            {name: 'p', label: 'Probability of success (p)', min: 0.01, max: 0.99, default: 0.3, step: 0.01}
        ],
        pmf: (k, params) => {
            const n = Math.max(1, Math.floor(params.n || 20));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            if (k < 0 || k > n) return 0;
            return combination(n, k) * Math.pow(p, k) * Math.pow(1-p, n-k);
        },
        range: (params) => {
            const n = Math.max(1, Math.floor(params.n || 20));
            return Array.from({length: n + 1}, (_, i) => i);
        },
        mean: (params) => {
            const n = Math.max(1, Math.floor(params.n || 20));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            return n * p;
        },
        variance: (params) => {
            const n = Math.max(1, Math.floor(params.n || 20));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            return n * p * (1 - p);
        }
    },

    poisson: {
        name: 'Poisson Distribution',
        description: 'Models the number of events occurring in a fixed interval of time or space.',
        applications: [
            'Customer service: Number of calls received per hour',
            'Manufacturing: Number of defects per unit',
            'Biology: Number of mutations per genome',
            'Traffic: Number of accidents per day',
            'Network: Number of packets arriving per second'
        ],
        parameters: [
            {name: 'lambda', label: 'Rate parameter (λ)', min: 0.1, max: 20, default: 3, step: 0.1}
        ],
        pmf: (k, params) => {
            const lambda = Math.max(0.1, params.lambda || 3);
            return poissonPMF(k, lambda);
        },
        range: (params) => {
            const lambda = Math.max(0.1, params.lambda || 3);
            // More conservative range to prevent numerical issues
            const maxK = Math.min(30, Math.max(15, Math.ceil(lambda * 2 + 10)));
            return Array.from({length: maxK + 1}, (_, i) => i);
        },
        mean: (params) => {
            const lambda = Math.max(0.1, params.lambda || 3);
            return lambda;
        },
        variance: (params) => {
            const lambda = Math.max(0.1, params.lambda || 3);
            return lambda;
        }
    },
    
    geometric: {
        name: 'Geometric Distribution',
        description: 'Models the number of trials until the first success occurs.',
        applications: [
            'Sales: Number of calls until first sale',
            'Testing: Number of tests until first failure',
            'Gaming: Number of attempts until winning',
            'Network reliability: Time until first packet loss',
            'Job search: Number of applications until job offer'
        ],
        parameters: [
            {name: 'p', label: 'Probability of success (p)', min: 0.01, max: 0.99, default: 0.2, step: 0.01}
        ],
        pmf: (k, params) => {
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.2));
            if (k < 1) return 0;
            return Math.pow(1-p, k-1) * p;
        },
        range: (params) => {
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.2));
            const maxK = Math.min(50, Math.ceil(10/p));
            return Array.from({length: maxK}, (_, i) => i + 1);
        },
        mean: (params) => {
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.2));
            return 1 / p;
        },
        variance: (params) => {
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.2));
            return (1 - p) / (p * p);
        }
    },

    negativeBinomial: {
        name: 'Negative Binomial Distribution',
        description: 'Models the number of trials until the r-th success occurs.',
        applications: [
            'Insurance: Number of claims until r-th major claim',
            'Reliability: Number of cycles until r-th component failure',
            'Epidemiology: Number of tests until r-th positive case',
            'Marketing: Number of contacts until r-th conversion',
            'Sports: Number of games until r-th win'
        ],
        parameters: [
            {name: 'r', label: 'Number of successes (r)', min: 1, max: 20, default: 3, step: 1},
            {name: 'p', label: 'Probability of success (p)', min: 0.01, max: 0.99, default: 0.3, step: 0.01}
        ],
        pmf: (k, params) => {
            const r = Math.max(1, Math.floor(params.r || 3));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            if (k < r) return 0;
            return combination(k-1, r-1) * Math.pow(p, r) * Math.pow(1-p, k-r);
        },
        range: (params) => {
            const r = Math.max(1, Math.floor(params.r || 3));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            const maxK = Math.min(100, Math.ceil(r/p * 3));
            return Array.from({length: maxK}, (_, i) => i + r);
        },
        mean: (params) => {
            const r = Math.max(1, Math.floor(params.r || 3));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            return r / p;
        },
        variance: (params) => {
            const r = Math.max(1, Math.floor(params.r || 3));
            const p = Math.max(0.01, Math.min(0.99, params.p || 0.3));
            return (r * (1 - p)) / (p * p);
        }
    },

    hypergeometric: {
        name: 'Hypergeometric Distribution',
        description: 'Models sampling without replacement from a finite population.',
        applications: [
            'Quality control: Defective items in sample without replacement',
            'Card games: Number of specific cards in a hand',
            'Survey sampling: Responses from specific demographics',
            'Biology: Species counts in ecological samples',
            'Lottery systems: Matching numbers without replacement'
        ],
        parameters: [
            {name: 'N', label: 'Population size (N)', min: 10, max: 100, default: 50, step: 1},
            {name: 'K', label: 'Successes in population (K)', min: 1, max: 49, default: 20, step: 1},
            {name: 'n', label: 'Sample size (n)', min: 1, max: 30, default: 10, step: 1}
        ],
        pmf: (k, params) => {
            const {N, K, n} = params;
            if (k < Math.max(0, n - (N - K)) || k > Math.min(n, K)) return 0;
            return (combination(K, k) * combination(N-K, n-k)) / combination(N, n);
        },
        range: (params) => {
            const N = Math.max(10, Math.floor(params.N || 50));
            const K = Math.max(1, Math.min(N-1, Math.floor(params.K || 20)));
            const n = Math.max(1, Math.min(N, Math.floor(params.n || 10)));
            
            const minK = Math.max(0, n - (N - K));
            const maxK = Math.min(n, K);
            
            if (minK > maxK) {
                // Fallback to a simple range if constraints are impossible
                return [0];
            }
            
            return Array.from({length: maxK - minK + 1}, (_, i) => i + minK);
        },
        mean: (params) => (params.n * params.K) / params.N,
        variance: (params) => (params.n * params.K * (params.N - params.K) * (params.N - params.n)) / 
                             (params.N * params.N * (params.N - 1))
    },

    uniform: {
        name: 'Uniform Distribution',
        description: 'Assigns equal probability to each outcome in a finite range.',
        applications: [
            'Random number generation: Fair dice or coin tosses',
            'Simulation modeling: Equally likely scenarios',
            'Cryptography: Random key generation',
            'Game design: Fair selection mechanisms',
            'Statistical sampling: Equal probability selection'
        ],
        parameters: [
            {name: 'a', label: 'Lower bound (a)', min: 0, max: 20, default: 1, step: 1},
            {name: 'b', label: 'Upper bound (b)', min: 2, max: 50, default: 10, step: 1}
        ],
        pmf: (k, params) => {
            const a = Math.floor(params.a || 1);
            const b = Math.floor(params.b || 10);
            const validB = Math.max(a + 1, b);
            
            if (k < a || k > validB) return 0;
            return 1 / (validB - a + 1);
        },
        range: (params) => {
            const a = Math.floor(params.a || 1);
            const b = Math.floor(params.b || 10);
            const validB = Math.max(a + 1, b);
            
            return Array.from({length: validB - a + 1}, (_, i) => i + a);
        },
        mean: (params) => {
            const a = Math.floor(params.a || 1);
            const b = Math.floor(params.b || 10);
            const validB = Math.max(a + 1, b);
            return (a + validB) / 2;
        },
        variance: (params) => {
            const a = Math.floor(params.a || 1);
            const b = Math.floor(params.b || 10);
            const validB = Math.max(a + 1, b);
            return ((validB - a + 1) * (validB - a + 1) - 1) / 12;
        }
    }
};

function factorial(n) {
    if (n <= 1) return 1;
    if (n > 170) return Infinity; // Prevent overflow
    let result = 1;
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// More stable Poisson PMF calculation using log space
function poissonPMF(k, lambda) {
    if (k < 0) return 0;
    if (k > 500) return 0; // Prevent extreme values
    
    // For large k, use Stirling's approximation in log space
    if (k > 170) {
        // log(P(X=k)) = k*log(λ) - λ - log(k!)
        // Using Stirling's approximation: log(k!) ≈ k*log(k) - k + 0.5*log(2π*k)
        const logK = Math.log(k);
        const logLambda = Math.log(lambda);
        const stirlingLogFactorial = k * logK - k + 0.5 * Math.log(2 * Math.PI * k);
        const logProb = k * logLambda - lambda - stirlingLogFactorial;
        return Math.exp(logProb);
    }
    
    // Standard calculation for reasonable k values
    return (Math.pow(Math.E, -lambda) * Math.pow(lambda, k)) / factorial(k);
}

function combination(n, k) {
    if (k > n || k < 0) return 0;
    if (k === 0 || k === n) return 1;
    
    k = Math.min(k, n - k);
    let result = 1;
    for (let i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

function selectDistribution(dist) {
    // Remove active class from all buttons
    document.querySelectorAll('.dist-btn').forEach(btn => btn.classList.remove('active'));
    
    // Add active class to clicked button
    const clickedButton = event.target.closest('.dist-btn');
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
    
    currentDistribution = dist;
    console.log('Selected distribution:', dist);
    
    updateControls();
    updateVisualization();
}

function updateControls() {
    const dist = distributions[currentDistribution];
    if (!dist) {
        console.error('Distribution not found:', currentDistribution);
        return;
    }
    
    const applicationsHTML = `
        <div class="applications-list">
            <h4>Common Applications</h4>
            <ul>
                ${dist.applications.map(app => `<li>${app}</li>`).join('')}
            </ul>
        </div>
    `;
    
    const distInfoEl = document.getElementById('distInfo');
    if (distInfoEl) {
        distInfoEl.innerHTML = `
            <h3>${dist.name}</h3>
            <p>${dist.description}</p>
            ${applicationsHTML}
        `;
    }

    const controlsHTML = dist.parameters.map(param => `
        <div class="control-group">
            <label for="${param.name}">${param.label}</label>
            <input type="number" 
                   id="${param.name}" 
                   min="${param.min}" 
                   max="${param.max}" 
                   step="${param.step}" 
                   value="${param.default}"
                   onchange="updateVisualization()"
                   oninput="handleParameterChange(this)">
        </div>
    `).join('');
    
    const parameterControlsEl = document.getElementById('parameterControls');
    if (parameterControlsEl) {
        parameterControlsEl.innerHTML = controlsHTML;
    }
}

function getParameters() {
    const dist = distributions[currentDistribution];
    const params = {};
    
    dist.parameters.forEach(param => {
        const element = document.getElementById(param.name);
        if (element) {
            let value = parseFloat(element.value);
            // Fallback to default if invalid
            if (isNaN(value) || !isFinite(value)) {
                value = param.default;
                element.value = value; // Reset the input to default
            }
            // Clamp to valid range
            value = Math.max(param.min, Math.min(param.max, value));
            params[param.name] = value;
        } else {
            params[param.name] = param.default;
        }
    });
    
    return params;
}

function handleParameterChange(input) {
    // Add visual feedback for parameter changes
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (value < min || value > max || isNaN(value)) {
        input.style.borderColor = '#ef4444';
        input.style.backgroundColor = '#fef2f2';
    } else {
        input.style.borderColor = '#10b981';
        input.style.backgroundColor = '#f0fdf4';
        setTimeout(() => {
            input.style.borderColor = '';
            input.style.backgroundColor = '';
        }, 1000);
        updateVisualization();
    }
}

function updateVisualization() {
    try {
        const dist = distributions[currentDistribution];
        if (!dist) {
            console.error('Distribution not found:', currentDistribution);
            return;
        }
        
        const params = getParameters();
                // Debug log parameters (properly formatted)
                console.log('Parameters for', currentDistribution + ':', JSON.stringify(params, null, 2));
        
        // Extra validation for hypergeometric parameters
        if (currentDistribution === 'hypergeometric') {
            // Ensure K <= N
            if (params.K > params.N) {
                params.K = Math.min(params.K, params.N);
                const kElement = document.getElementById('K');
                if (kElement) kElement.value = params.K;
            }
            // Ensure n <= N
            if (params.n > params.N) {
                params.n = Math.min(params.n, params.N);
                const nElement = document.getElementById('n');
                if (nElement) nElement.value = params.n;
            }
            // Update max attributes dynamically
            const kElement = document.getElementById('K');
            const nElement = document.getElementById('n');
            if (kElement) kElement.max = params.N - 1;
            if (nElement) nElement.max = params.N;
        }
        
        // Extra validation for uniform parameters
        if (currentDistribution === 'uniform') {
            if (params.a >= params.b) {
                params.b = params.a + 1;
                const bElement = document.getElementById('b');
                if (bElement) bElement.value = params.b;
            }
        }

        // Calculate range and probabilities
        let range, probabilities;
        try {
            range = dist.range(params);
            if (!range || range.length === 0) {
                console.error('Empty range for distribution:', currentDistribution);
                return;
            }
            
            probabilities = range.map(k => {
                try {
                    const prob = dist.pmf(k, params);
                    return isNaN(prob) || !isFinite(prob) ? 0 : prob;
                } catch (e) {
                    console.warn(`Error calculating PMF for k=${k}:`, e);
                    return 0;
                }
            });
        } catch (e) {
            console.error('Error calculating range/probabilities:', e);
            return;
        }
        
        // Update statistics with error handling
        let mean, variance, stddev;
        try {
            mean = dist.mean(params);
            variance = dist.variance(params);
            stddev = Math.sqrt(Math.max(0, variance)); // Ensure non-negative for sqrt
        } catch (e) {
            console.warn('Error calculating statistics:', e);
            mean = 0;
            variance = 0;
            stddev = 0;
        }
        
        // Ensure we have valid numbers
        mean = isNaN(mean) || !isFinite(mean) ? 0 : mean;
        variance = isNaN(variance) || !isFinite(variance) ? 0 : variance;
        stddev = isNaN(stddev) || !isFinite(stddev) ? 0 : stddev;
        
        const meanEl = document.getElementById('meanValue');
        const varianceEl = document.getElementById('varianceValue');
        const stddevEl = document.getElementById('stddevValue');
        
        if (meanEl) meanEl.textContent = mean.toFixed(3);
        if (varianceEl) varianceEl.textContent = variance.toFixed(3);
        if (stddevEl) stddevEl.textContent = stddev.toFixed(3);

        // Update chart
        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }

        const chartCanvas = document.getElementById('distributionChart');
        if (!chartCanvas) {
            console.error('Chart canvas not found');
            return;
        }

        const ctx = chartCanvas.getContext('2d');
        if (!ctx) {
            console.error('Cannot get chart context');
            return;
        }
        
        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
        gradient.addColorStop(1, 'rgba(118, 75, 162, 0.8)');

        currentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: range.map(k => k.toString()),
                datasets: [{
                    label: 'Probability',
                    data: probabilities,
                    backgroundColor: gradient,
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: dist.name,
                        font: {
                            size: 18,
                            weight: 'bold',
                            family: 'Inter'
                        },
                        color: '#1a202c',
                        padding: 20
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 32, 44, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                            title: function(context) {
                                return `X = ${context[0].label}`;
                            },
                            label: function(context) {
                                return `P(X = ${context.label}) = ${context.parsed.y.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Value (k)',
                            font: {
                                weight: 'bold',
                                size: 14,
                                family: 'Inter'
                            },
                            color: '#4a5568'
                        },
                        grid: {
                            color: 'rgba(226, 232, 240, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#718096',
                            font: {
                                family: 'JetBrains Mono'
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Probability P(X = k)',
                            font: {
                                weight: 'bold',
                                size: 14,
                                family: 'Inter'
                            },
                            color: '#4a5568'
                        },
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(226, 232, 240, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#718096',
                            font: {
                                family: 'JetBrains Mono'
                            },
                            callback: function(value) {
                                return value.toFixed(3);
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    } catch (error) {
        console.error('Error updating visualization:', error);
        // Display user-friendly error message
        const meanEl = document.getElementById('meanValue');
        const varianceEl = document.getElementById('varianceValue');
        const stddevEl = document.getElementById('stddevValue');
        
        if (meanEl) meanEl.textContent = 'Error';
        if (varianceEl) varianceEl.textContent = 'Error';
        if (stddevEl) stddevEl.textContent = 'Error';
    }
}

// VaR Animation Code
let varAnimationId = null;
let varT = 0;
let varIsPlaying = false;
let varStrategy = 'shift';

const varInitial = { mu: 0, sigma: 0.02 };
const varTargets = {
    shift: { mu: 0.01, sigma: 0.02 },
    tighten: { mu: 0, sigma: 0.012 },
    both: { mu: 0.008, sigma: 0.014 }
};

// Math helpers for VaR
function erf(x) {
    const sign = Math.sign(x); x = Math.abs(x);
    const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
    const t = 1/(1+p*x);
    const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);
    return sign*y;
}

function pdf(x, mu, sigma) {
    const z = (x-mu)/sigma;
    return Math.exp(-0.5*z*z)/(sigma*Math.sqrt(2*Math.PI));
}

function quantile(p, mu, sigma) {
    return mu + sigma * (-1.64485362695147);
}

function cdf(x, mu, sigma) {
    return 0.5 * (1 + erf((x - mu) / (sigma * Math.sqrt(2))));
}

function lerp(start, end, t) {
    return start + (end - start) * t;
}

function easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function getCurrentVarParams() {
    const target = varTargets[varStrategy];
    const easedT = easeInOut(varT);
    return {
        mu: lerp(varInitial.mu, target.mu, easedT),
        sigma: lerp(varInitial.sigma, target.sigma, easedT)
    };
}

function drawVarChart() {
    const canvas = document.getElementById('varPlot');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const pad = 60;
    const plotW = w - 2*pad, plotH = h - 2*pad;
    
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(pad, pad);
    
    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, plotW, plotH);
    
    // Grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    for(let i = 0; i <= 10; i++) {
        const x = i/10 * plotW;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, plotH); ctx.stroke();
    }
    for(let i = 0; i <= 8; i++) {
        const y = i/8 * plotH;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
    }

    // Domain: -6% to +4%
    const xmin = -0.06, xmax = 0.04;
    const xToPx = x => (x - xmin)/(xmax - xmin) * plotW;
    
    // Current and initial parameters
    const current = getCurrentVarParams();
    const initialVar95 = -quantile(0.05, varInitial.mu, varInitial.sigma);
    const currentVar95 = -quantile(0.05, current.mu, current.sigma);
    
    // Y scaling based on max PDF
    const yMax = Math.max(pdf(varInitial.mu, varInitial.mu, varInitial.sigma), pdf(current.mu, current.mu, current.sigma));
    const yToPx = y => plotH - (y/yMax) * plotH * 0.9;

    // Draw initial distribution (blue, lighter if animating)
    ctx.strokeStyle = varIsPlaying ? 'rgba(37, 99, 235, 0.4)' : '#2563eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for(let i = 0; i <= 400; i++) {
        const x = xmin + (xmax - xmin) * i/400;
        const y = pdf(x, varInitial.mu, varInitial.sigma);
        const X = xToPx(x), Y = yToPx(y);
        if(i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.stroke();

    // Draw current distribution (green, if different from initial)
    if(varT > 0.01) {
        ctx.strokeStyle = '#059669';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for(let i = 0; i <= 400; i++) {
            const x = xmin + (xmax - xmin) * i/400;
            const y = pdf(x, current.mu, current.sigma);
            const X = xToPx(x), Y = yToPx(y);
            if(i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
        }
        ctx.stroke();
    }

    // Fixed VaR line (at initial VaR₉₅ position)
    const fixedVar95X = xToPx(-initialVar95);
    ctx.strokeStyle = '#dc2626';
    ctx.setLineDash([8, 4]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(fixedVar95X, 0);
    ctx.lineTo(fixedVar95X, plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Shade left tail (area to left of FIXED line for current distribution)
    const params = varT > 0.01 ? current : varInitial;
    const leftTailX = Math.min(-initialVar95, xmax);
    
    ctx.fillStyle = 'rgba(220, 38, 38, 0.2)';
    ctx.beginPath();
    ctx.moveTo(0, plotH);
    
    if(-initialVar95 > xmin) {
        const steps = Math.floor(200 * (leftTailX - xmin) / (xmax - xmin));
        for(let i = 0; i <= steps; i++) {
            const x = xmin + (leftTailX - xmin) * i/steps;
            const y = pdf(x, params.mu, params.sigma);
            const X = xToPx(x), Y = yToPx(y);
            if(i === 0) ctx.lineTo(X, Y); else ctx.lineTo(X, Y);
        }
        ctx.lineTo(fixedVar95X, plotH);
    } else {
        ctx.lineTo(0, plotH);
    }
    ctx.closePath();
    ctx.fill();

    // Zero line
    const zeroX = xToPx(0);
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(zeroX, 0);
    ctx.lineTo(zeroX, plotH);
    ctx.stroke();

    // X-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '12px ui-sans-serif, system-ui';
    for(let i = -6; i <= 4; i += 2) {
        const x = i/100;
        const X = xToPx(x);
        ctx.fillText(`${i}%`, X - 10, plotH + 20);
    }
    ctx.fillText('Portfolio Return →', plotW - 100, plotH + 40);

    // Y-axis label
    ctx.save();
    ctx.translate(15, plotH/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('Probability Density', -40, 0);
    ctx.restore();

    ctx.restore();

    // Calculate probability to left of fixed line for current distribution  
    const currentProbability = cdf(-initialVar95, current.mu, current.sigma);
    const newVar95 = -quantile(0.05, current.mu, current.sigma);
    
    // Update displays
    const varDisplayEl = document.getElementById('varDisplay');
    const varStatsEl = document.getElementById('varStatsDisplay');
    
    if (varDisplayEl) {
        varDisplayEl.innerHTML = `
            <div style="font-size: 14px; color: #64748b; margin-bottom: 4px;">Fixed at initial VaR₉₅:</div>
            <div style="font-size: 18px; font-weight: 600; color: #dc2626;">${(initialVar95*100).toFixed(2)}%</div>
            <div style="font-size: 12px; color: #059669; margin-top: 4px;">Risk reduced by ${((1 - currentProbability/0.05) * 100).toFixed(1)}%</div>
        `;
    }
    
    if (varStatsEl) {
        const progress = Math.round(varT * 100);
        varStatsEl.innerHTML = `
            <div><strong>Current Portfolio:</strong></div>
            <div>• Mean return: ${(current.mu*100).toFixed(1)}%</div>
            <div>• Volatility: ${(current.sigma*100).toFixed(1)}%</div>
            <div>• Prob(loss > ${(initialVar95*100).toFixed(2)}%): <strong>${(currentProbability*100).toFixed(2)}%</strong></div>
            <div>• New VaR₉₅: ${(newVar95*100).toFixed(2)}%</div>
            <div><strong>Progress:</strong> ${progress}%</div>
        `;
    }
}

function animateVar() {
    if(!varIsPlaying) return;
    
    varT += 0.008;
    if(varT >= 1) {
        varT = 1;
        varIsPlaying = false;
        const playBtn = document.getElementById('varPlayBtn');
        if (playBtn) {
            playBtn.textContent = '✓ Optimization Complete';
            playBtn.classList.remove('play');
        }
    }
    
    drawVarChart();
    
    if(varIsPlaying) {
        varAnimationId = requestAnimationFrame(animateVar);
    }
}

function startVarAnimation() {
    const playBtn = document.getElementById('varPlayBtn');
    if (!playBtn) return;
    
    if(varIsPlaying) {
        varIsPlaying = false;
        playBtn.textContent = '▶ Start Optimization';
        playBtn.classList.add('play');
        if(varAnimationId) cancelAnimationFrame(varAnimationId);
    } else {
        varIsPlaying = true;
        playBtn.textContent = '⏸ Pause';
        playBtn.classList.add('play');
        animateVar();
    }
}

function resetVar() {
    varIsPlaying = false;
    varT = 0;
    const playBtn = document.getElementById('varPlayBtn');
    if (playBtn) {
        playBtn.textContent = '▶ Start Optimization';
        playBtn.classList.add('play');
    }
    if(varAnimationId) cancelAnimationFrame(varAnimationId);
    drawVarChart();
}

function setVarStrategy(newStrategy) {
    varStrategy = newStrategy;
    const shiftBtn = document.getElementById('varShiftBtn');
    const tightenBtn = document.getElementById('varTightenBtn');
    const bothBtn = document.getElementById('varBothBtn');
    
    [shiftBtn, tightenBtn, bothBtn].forEach(btn => btn && btn.classList.remove('active'));
    
    if(newStrategy === 'shift' && shiftBtn) shiftBtn.classList.add('active');
    else if(newStrategy === 'tighten' && tightenBtn) tightenBtn.classList.add('active');
    else if(bothBtn) bothBtn.classList.add('active');
    
    if(!varIsPlaying) drawVarChart();
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Add loading state for main chart
        const chartContainer = document.querySelector('.chart-container');
        if (chartContainer) {
            chartContainer.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; height: 100%;"><div class="loading"></div></div>';
        }
        
        // Initialize after a short delay
        setTimeout(() => {
            if (chartContainer) {
                chartContainer.innerHTML = '<canvas id="distributionChart"></canvas>';
            }
            updateControls();
            updateVisualization();
            
            // Set up distribution button event listeners
            document.querySelectorAll('.dist-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const distribution = this.getAttribute('data-distribution');
                    if (distribution) {
                        selectDistribution(distribution);
                    }
                });
            });
            
            // Initialize VaR chart
            drawVarChart();
            
            // Set up VaR event listeners
            const varPlayBtn = document.getElementById('varPlayBtn');
            const varResetBtn = document.getElementById('varResetBtn');
            const varShiftBtn = document.getElementById('varShiftBtn');
            const varTightenBtn = document.getElementById('varTightenBtn');
            const varBothBtn = document.getElementById('varBothBtn');
            
            if (varPlayBtn) varPlayBtn.addEventListener('click', startVarAnimation);
            if (varResetBtn) varResetBtn.addEventListener('click', resetVar);
            if (varShiftBtn) varShiftBtn.addEventListener('click', () => setVarStrategy('shift'));
            if (varTightenBtn) varTightenBtn.addEventListener('click', () => setVarStrategy('tighten'));
            if (varBothBtn) varBothBtn.addEventListener('click', () => setVarStrategy('both'));
        }, 500);
    } catch (error) {
        console.error('Error initializing application:', error);
    }
});

// Add smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
