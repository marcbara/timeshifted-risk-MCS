import matplotlib.pyplot as plt
import numpy as np
import os

# PMJ publication settings
def apply_pmj_settings():
    """Apply PMJ publication requirements to all plots."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']  # PMJ requires Arial or Helvetica for figures
    plt.rcParams['font.size'] = 10               # PMJ requires 10pt font for figures
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5

# Utility Functions
def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row of the matrix so that it sums to 1."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums != 0)
    return np.nan_to_num(normalized)

def compute_cumulative_probabilities(matrix: np.ndarray) -> np.ndarray:
    """Compute the cumulative probabilities for each row of the matrix."""
    cumulative_matrix = np.cumsum(matrix, axis=1)
    row_sums = cumulative_matrix[:, -1:]
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    return cumulative_matrix / row_sums

def compute_p90(cumulative_probabilities: np.ndarray, bin_labels: np.ndarray) -> np.ndarray:
    """Compute the P90 value for each occurrence day using cumulative probabilities."""
    p90_values = []
    for i in range(1, cumulative_probabilities.shape[0]):  # Start from day 1
        if np.any(cumulative_probabilities[i] > 0):
            if np.any(cumulative_probabilities[i] >= 0.9):
                p90_index = np.argmax(cumulative_probabilities[i] >= 0.9)
                p90_values.append(bin_labels[p90_index])
            else:
                p90_values.append(bin_labels[-1])
        else:
            p90_values.append(0)
    return np.array(p90_values)

def detect_steady_state(matrix: np.ndarray, threshold: float = 1) -> int:
    """Detect the steady state in a matrix."""
    last_row = matrix[-1]
    for i in range(len(matrix) - 2, -1, -1):
        if np.max(np.abs(matrix[i] - last_row)) > threshold:
            return i + 1
    return 0

# Plotting Functions
def plot_p90_over_time(p90_values: np.ndarray, title: str, ylabel: str, save_path: str, grayscale: bool = False, dpi: int = 300):
    """Plot P90 values over project days."""
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=dpi)
    
    # Set grayscale styling if needed
    if grayscale:
        line_color = 'black'
        file_suffix = "_grayscale"
    else:
        line_color = 'blue'
        file_suffix = ""
        
    plt.plot(np.arange(1, len(p90_values) + 1), p90_values, color=line_color, 
             linestyle='-', linewidth=1.5, label=f'P90 {ylabel}')
    plt.title(title)
    plt.xlabel("Project Day")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save both versions if grayscale requested
    base, ext = os.path.splitext(save_path)
    output_path = f"{base}{file_suffix}{ext}"
    plt.savefig(output_path, dpi=dpi)
    plt.close()

def plot_heatmap(matrix: np.ndarray, title: str, xlabel: str, ylabel: str, cbar_label: str, 
                 xticks: np.ndarray, yticks: np.ndarray, steady_state: int, save_path: str, 
                 cmap: str, grayscale: bool = False, dpi: int = 300):
    """Plot a heatmap with customizable labels and steady state line."""
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=dpi)
    
    # Set grayscale styling if needed
    if grayscale:
        used_cmap = 'gray'
        line_color = 'black'
        text_color = 'black'
        file_suffix = "_grayscale"
    else:
        used_cmap = cmap
        line_color = 'white'
        text_color = 'white'
        file_suffix = ""
    
    plt.imshow(matrix, aspect='auto', origin='lower', cmap=used_cmap)
    cbar = plt.colorbar(label=cbar_label)
    cbar.ax.tick_params(labelsize=9)  # Slightly smaller colorbar ticks for clarity
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks[0], xticks[1], rotation=45)
    plt.yticks(yticks[0], yticks[1])
    
    if steady_state > 0:
        plt.axhline(y=steady_state, color=line_color, linestyle='--', linewidth=1.5)
        plt.text(0, steady_state, f'Steady State (Day {steady_state})', 
                 color=text_color, va='bottom', ha='left', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save both versions if grayscale requested
    base, ext = os.path.splitext(save_path)
    output_path = f"{base}{file_suffix}{ext}"
    plt.savefig(output_path, dpi=dpi)
    plt.close()

def plot_final_distribution(labels: np.ndarray, values: np.ndarray, title: str, xlabel: str, 
                           bin_size: int, save_path: str, grayscale: bool = False, dpi: int = 300):
    """Plot the final distribution as a bar chart."""
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=dpi)
    
    # Set grayscale styling if needed
    if grayscale:
        bar_color = 'black'
        edge_color = 'gray' 
        file_suffix = "_grayscale"
    else:
        bar_color = 'red'
        edge_color = 'black'
        file_suffix = ""
    
    plt.bar(labels, values, width=bin_size * 1.0, align='center', 
            alpha=0.8, color=bar_color, edgecolor=edge_color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative Frequency")
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save both versions if grayscale requested
    base, ext = os.path.splitext(save_path)
    output_path = f"{base}{file_suffix}{ext}"
    plt.savefig(output_path, dpi=dpi)
    plt.close()

# Main Function
def plot_cumulative_distributions(cost_matrix: np.ndarray, delay_matrix: np.ndarray, 
                                  cost_bin_size: int, delay_bin_size: int, 
                                  min_cost: int, min_delay: int, save_path: str = 'png/',
                                  create_grayscale: bool = True, dpi: int = 300) -> None:
    """
    Plot cumulative distributions for cost and delay, with steady-state detection.
    
    Parameters:
    -----------
    cost_matrix, delay_matrix : np.ndarray
        Matrices containing cost and delay distribution data
    cost_bin_size, delay_bin_size : int
        Bin sizes for cost and delay distributions
    min_cost, min_delay : int
        Minimum values for cost and delay
    save_path : str, default='png/'
        Directory to save output images
    create_grayscale : bool, default=True
        Whether to create grayscale versions for PMJ submission
    dpi : int, default=300
        Resolution for saved images (300 is PMJ requirement)
    """
    os.makedirs(save_path, exist_ok=True)
    heatmap_cmap = 'coolwarm'  # Original colormap for color versions

    # Preprocessing
    cost_steady_state = detect_steady_state(cost_matrix)
    delay_steady_state = detect_steady_state(delay_matrix)
    
    cost_matrix_scaled = cost_matrix / np.maximum(cost_matrix.max(axis=1, keepdims=True), 1)
    delay_matrix_scaled = delay_matrix / np.maximum(delay_matrix.max(axis=1, keepdims=True), 1)
    
    cumulative_cost_probabilities = compute_cumulative_probabilities(cost_matrix)
    cumulative_delay_probabilities = compute_cumulative_probabilities(delay_matrix)
    
    cost_labels = np.array([min_cost + i * cost_bin_size for i in range(cost_matrix.shape[1])])
    delay_labels = np.array([min_delay + i * delay_bin_size for i in range(delay_matrix.shape[1])])
    
    cost_p90 = compute_p90(cumulative_cost_probabilities, cost_labels)
    delay_p90 = compute_p90(cumulative_delay_probabilities, delay_labels)

    # Plotting - first create color versions
    # P90 plots
    plot_p90_over_time(cost_p90, "P90 Cost Values Over Project Days", "P90 Cost Value", 
                      f'{save_path}p90_cost_over_project_days.png', False, dpi)
    plot_p90_over_time(delay_p90, "P90 Delay Values Over Project Days", "P90 Delay Value", 
                      f'{save_path}p90_delay_over_project_days.png', False, dpi)

    # Calculate axes ticks
    num_cost_ticks = min(10, len(cost_labels))
    num_delay_ticks = min(10, len(delay_labels))
    num_days_ticks = min(10, cost_matrix.shape[0])

    cost_xticks = (np.linspace(0, len(cost_labels) - 1, num_cost_ticks).astype(int),
                   [f"{int(label)}" for label in np.linspace(min_cost, cost_labels[-1], num_cost_ticks)])
    delay_xticks = (np.linspace(0, len(delay_labels) - 1, num_delay_ticks).astype(int),
                    [f"{int(label)}" for label in np.linspace(min_delay, delay_labels[-1], num_delay_ticks)])
    days_yticks = (np.linspace(0, cost_matrix.shape[0] - 1, num_days_ticks).astype(int),
                   np.linspace(0, cost_matrix.shape[0] - 1, num_days_ticks).astype(int))

    # Scaled distribution heatmaps
    plot_heatmap(cost_matrix_scaled, "Scaled Cumulative Cost Distribution Over Time", "Cumulative Cost", 
                "Project Day", "Scaled Frequency", cost_xticks, days_yticks, cost_steady_state, 
                f'{save_path}scaled_cumulative_cost_distribution.png', heatmap_cmap, False, dpi)
    plot_heatmap(delay_matrix_scaled, "Scaled Cumulative Delay Distribution Over Time", "Cumulative Delay", 
                "Project Day", "Scaled Frequency", delay_xticks, days_yticks, delay_steady_state, 
                f'{save_path}scaled_cumulative_delay_distribution.png', heatmap_cmap, False, dpi)
    
    # Cumulative probability heatmaps
    plot_heatmap(cumulative_cost_probabilities, "Cumulative Cost Probability Distribution Over Time", "Cumulative Cost", 
                "Project Day", "Cumulative Probability", cost_xticks, days_yticks, 0, 
                f'{save_path}cumulative_cost_probability_distribution.png', heatmap_cmap, False, dpi)
    plot_heatmap(cumulative_delay_probabilities, "Cumulative Delay Probability Distribution Over Time", "Cumulative Delay", 
                "Project Day", "Cumulative Probability", delay_xticks, days_yticks, 0, 
                f'{save_path}cumulative_delay_probability_distribution.png', heatmap_cmap, False, dpi)

    # Final distribution histograms
    plot_final_distribution(cost_labels, cost_matrix[-1, :], "Final Cumulative Cost Distribution", "Cost", 
                          cost_bin_size, f'{save_path}final_cumulative_cost_distribution.png', False, dpi)
    plot_final_distribution(delay_labels, delay_matrix[-1, :], "Final Cumulative Delay Distribution", "Delay", 
                          delay_bin_size, f'{save_path}final_cumulative_delay_distribution.png', False, dpi)

    # Create grayscale versions if requested
    if create_grayscale:
        # P90 plots - grayscale
        plot_p90_over_time(cost_p90, "P90 Cost Values Over Project Days", "P90 Cost Value", 
                          f'{save_path}p90_cost_over_project_days.png', True, dpi)
        plot_p90_over_time(delay_p90, "P90 Delay Values Over Project Days", "P90 Delay Value", 
                          f'{save_path}p90_delay_over_project_days.png', True, dpi)
        
        # Scaled distribution heatmaps - grayscale
        plot_heatmap(cost_matrix_scaled, "Scaled Cumulative Cost Distribution Over Time", "Cumulative Cost", 
                    "Project Day", "Scaled Frequency", cost_xticks, days_yticks, cost_steady_state, 
                    f'{save_path}scaled_cumulative_cost_distribution.png', heatmap_cmap, True, dpi)
        plot_heatmap(delay_matrix_scaled, "Scaled Cumulative Delay Distribution Over Time", "Cumulative Delay", 
                    "Project Day", "Scaled Frequency", delay_xticks, days_yticks, delay_steady_state, 
                    f'{save_path}scaled_cumulative_delay_distribution.png', heatmap_cmap, True, dpi)
        
        # Cumulative probability heatmaps - grayscale
        plot_heatmap(cumulative_cost_probabilities, "Cumulative Cost Probability Distribution Over Time", "Cumulative Cost", 
                    "Project Day", "Cumulative Probability", cost_xticks, days_yticks, 0, 
                    f'{save_path}cumulative_cost_probability_distribution.png', heatmap_cmap, True, dpi)
        plot_heatmap(cumulative_delay_probabilities, "Cumulative Delay Probability Distribution Over Time", "Cumulative Delay", 
                    "Project Day", "Cumulative Probability", delay_xticks, days_yticks, 0, 
                    f'{save_path}cumulative_delay_probability_distribution.png', heatmap_cmap, True, dpi)
        
        # Final distribution histograms - grayscale
        plot_final_distribution(cost_labels, cost_matrix[-1, :], "Final Cumulative Cost Distribution", "Cost", 
                              cost_bin_size, f'{save_path}final_cumulative_cost_distribution.png', True, dpi)
        plot_final_distribution(delay_labels, delay_matrix[-1, :], "Final Cumulative Delay Distribution", "Delay", 
                              delay_bin_size, f'{save_path}final_cumulative_delay_distribution.png', True, dpi)

    print(f"Cost distribution reaches steady state at day: {cost_steady_state}")
    print(f"Delay distribution reaches steady state at day: {delay_steady_state}")
    print(f"Generated all visualizations with {dpi} DPI resolution")
    if create_grayscale:
        print(f"Grayscale versions created for PMJ submission")