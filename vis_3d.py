import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def styblinski_tang_3d(x, y, z):
    """Styblinski-Tang function in 3D"""
    return 0.5 * ((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y) + (z**4 - 16*z**2 + 5*z))

def rastrigin_3d(x, y, z):
    """Rastrigin function in 3D"""
    A = 10
    return 3*A + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y)) + (z**2 - A*np.cos(2*np.pi*z))

def griewank_3d(x, y, z):
    """Griewank function in 3D"""
    return 1 + (x**2 + y**2 + z**2)/4000 - np.cos(x)*np.cos(y/np.sqrt(2))*np.cos(z/np.sqrt(3))

def plot_3d_function_surfaces(func, func_name, bounds=(-5, 5), resolution=100):
    """
    Create multiple visualizations of a 3D function:
    1. Surface plot (z=0 slice)
    2. Contour plot (z=0 slice) 
    3. Multiple z-slices
    4. 3D scatter plot of sample points
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Surface plot at z=0
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z_surface = func(X, Y, 0)  # z=0 slice
    
    surface = ax1.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(x,y,0)')
    ax1.set_title(f'{func_name}\nSurface at z=0')
    
    # 2. Contour plot at z=0
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contour(X, Y, Z_surface, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{func_name}\nContour at z=0')
    ax2.grid(True, alpha=0.3)
    
    # Add global minimum marker if visible in this slice
    if func_name == 'Styblinski-Tang':
        ax2.plot(-2.903534, -2.903534, 'r*', markersize=15, label='Global Min')
    elif func_name in ['Rastrigin', 'Griewank']:
        ax2.plot(0, 0, 'r*', markersize=15, label='Global Min')
    ax2.legend()
    
    # 3. Multiple z-slices contour
    ax3 = fig.add_subplot(2, 3, 3)
    z_slices = [-2, -1, 0, 1, 2]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, z_val in enumerate(z_slices):
        Z_slice = func(X, Y, z_val)
        ax3.contour(X, Y, Z_slice, levels=[np.percentile(Z_slice, 10)], 
                   colors=[colors[i]], alpha=0.7, linewidths=2)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title(f'{func_name}\nMultiple Z-slices')
    ax3.grid(True, alpha=0.3)
    ax3.legend([f'z={z}' for z in z_slices])
    
    # 4. 3D scatter plot showing function landscape
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Sample points in 3D space
    n_points = 2000
    x_scatter = np.random.uniform(bounds[0], bounds[1], n_points)
    y_scatter = np.random.uniform(bounds[0], bounds[1], n_points)
    z_scatter = np.random.uniform(bounds[0], bounds[1], n_points)
    f_values = func(x_scatter, y_scatter, z_scatter)
    
    # Color by function value
    scatter = ax4.scatter(x_scatter, y_scatter, z_scatter, c=f_values, 
                         cmap='viridis', alpha=0.6, s=1)
    
    # Mark global minimum
    if func_name == 'Styblinski-Tang':
        ax4.scatter([-2.903534], [-2.903534], [-2.903534], 
                   c='red', s=100, marker='*', label='Global Min')
    elif func_name in ['Rastrigin', 'Griewank']:
        ax4.scatter([0], [0], [0], c='red', s=100, marker='*', label='Global Min')
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title(f'{func_name}\n3D Function Landscape')
    ax4.legend()
    
    # 5. Cross-section through global minimum
    ax5 = fig.add_subplot(2, 3, 5)
    
    if func_name == 'Styblinski-Tang':
        # Line through global minimum
        x_line = np.linspace(bounds[0], bounds[1], 200)
        y_line = func(x_line, -2.903534, -2.903534)  # Fix y,z at global min
        ax5.plot(x_line, y_line, 'b-', linewidth=2, label='x-direction')
        ax5.axvline(-2.903534, color='red', linestyle='--', label='Global min x')
    else:
        # Line through origin
        x_line = np.linspace(bounds[0], bounds[1], 200)
        y_line = func(x_line, 0, 0)  # Fix y,z at origin
        ax5.plot(x_line, y_line, 'b-', linewidth=2, label='x-direction')
        ax5.axvline(0, color='red', linestyle='--', label='Global min x')
    
    ax5.set_xlabel('X')
    ax5.set_ylabel('f(x, y_min, z_min)')
    ax5.set_title(f'{func_name}\nCross-section through Global Min')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Function statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate some statistics
    x_test = np.linspace(bounds[0], bounds[1], 50)
    y_test = np.linspace(bounds[0], bounds[1], 50)
    z_test = np.linspace(bounds[0], bounds[1], 50)
    X_test, Y_test, Z_test = np.meshgrid(x_test, y_test, z_test)
    f_test = func(X_test, Y_test, Z_test)
    
    stats_text = f"""
    {func_name} Function Statistics:
    
    Domain: [{bounds[0]}, {bounds[1]}]³
    
    Min value in domain: {f_test.min():.3f}
    Max value in domain: {f_test.max():.3f}
    Mean value: {f_test.mean():.3f}
    Std deviation: {f_test.std():.3f}
    
    Global minimum:
    """
    
    if func_name == 'Styblinski-Tang':
        stats_text += "\n    x* = (-2.903, -2.903, -2.903)\n    f* ≈ -117.5"
    elif func_name == 'Rastrigin':
        stats_text += "\n    x* = (0, 0, 0)\n    f* = 0"
    elif func_name == 'Griewank':
        stats_text += "\n    x* = (0, 0, 0)\n    f* = 0"
    
    stats_text += f"\n\nFunction characteristics:\n"
    if func_name == 'Styblinski-Tang':
        stats_text += "• Highly multimodal\n• Many local minima\n• Fourth-order polynomial"
    elif func_name == 'Rastrigin':
        stats_text += "• Highly multimodal\n• Regular pattern of local minima\n• Cosine oscillations"
    elif func_name == 'Griewank':
        stats_text += "• Multimodal with global structure\n• Product of cosines\n• Scale-dependent difficulty"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.tight_layout()
    return fig

def visualize_all_functions():
    """Create visualizations for all three functions and save as PNG"""
    
    functions = [
        (styblinski_tang_3d, 'Styblinski-Tang', (-5, 5)),
        (rastrigin_3d, 'Rastrigin', (-5, 5)),
        (griewank_3d, 'Griewank', (-10, 10))  # Griewank needs larger domain
    ]
    
    for func, name, bounds in functions:
        print(f"Creating visualization for {name} function...")
        
        fig = plot_3d_function_surfaces(func, name, bounds)
        
        # Save as PNG
        filename = f'{name.lower().replace("-", "_")}_3d_visualization.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        # Show the plot
        plt.show()
        
        plt.close(fig)  # Close to free memory

if __name__ == "__main__":
    print("Generating 3D function visualizations...")
    print("This will create detailed visualizations showing:")
    print("- Surface plots")
    print("- Contour plots") 
    print("- 3D scatter plots")
    print("- Cross-sections")
    print("- Function statistics")
    print()
    
    visualize_all_functions()
    
    print("\nVisualization complete!")
    print("PNG files saved:")
    print("- styblinski_tang_3d_visualization.png")
    print("- rastrigin_3d_visualization.png") 
    print("- griewank_3d_visualization.png")