import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row of the matrix so that it sums to 1.
    Replace NaN values (from division by zero) with zeros.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums != 0)
    return np.nan_to_num(normalized)

def compute_cumulative_probabilities(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative probabilities for each row of the matrix.
    Every row will start at 0 and finish at 1.
    """
    cumulative_matrix = np.cumsum(matrix, axis=1)
    # Normalize each row so that the last element becomes 1
    row_sums = cumulative_matrix[:, -1:]
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    cumulative_probabilities = cumulative_matrix / row_sums
    return cumulative_probabilities

def compute_p90(cumulative_probabilities: np.ndarray, bin_labels: np.ndarray) -> np.ndarray:
    """
    Compute the P90 value for each occurrence day using cumulative probabilities.
    Exclude day 0 from the computation.
    If the cumulative probability does not reach 0.9, return 0 for that day.
    """
    p90_values = []
    for i in range(1, cumulative_probabilities.shape[0]):  # Start from day 1, skipping day 0
        # Check if there are any risks (non-zero values) for the day
        if np.any(cumulative_probabilities[i] > 0):
            # Check if the cumulative probability reaches or exceeds 0.9
            if np.any(cumulative_probabilities[i] >= 0.9):
                # Find the index where the cumulative probability exceeds or equals 0.9
                p90_index = np.argmax(cumulative_probabilities[i] >= 0.9)
                p90_values.append(bin_labels[p90_index])
            else:
                # If it never reaches 0.9, set the P90 to the highest bin value (as it's cumulative)
                p90_values.append(bin_labels[-1])
        else:
            # No risks have occurred yet, so P90 is 0
            p90_values.append(0)
    return np.array(p90_values)

def detect_steady_state(matrix: np.ndarray, threshold: float = 1) -> int:
    """
    Detect the steady state in a matrix.
    Returns the index of the first row that is sufficiently similar to the last row.
    """
    last_row = matrix[-1]
    for i in range(len(matrix) - 2, -1, -1):
        if np.max(np.abs(matrix[i] - last_row)) > threshold:
            return i + 1
    return 0

def plot_cumulative_distributions(cost_matrix: np.ndarray, delay_matrix: np.ndarray, 
                                  cost_bin_size: int, delay_bin_size: int, 
                                  min_cost: int, min_delay: int, save_path: str = 'png/') -> None:
    """
    Plot cumulative distributions for cost and delay, with steady-state detection.
    """

    # Define the color map to be used for all heatmaps
    heatmap_cmap = 'coolwarm'  # Example: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', etc.

    # Create the subfolder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Detect steady states
    cost_steady_state = detect_steady_state(cost_matrix)
    delay_steady_state = detect_steady_state(delay_matrix)

    # Scale each row of the matrices independently
    max_cost_per_row = cost_matrix.max(axis=1, keepdims=True)
    max_cost_per_row[max_cost_per_row == 0] = 1  # Prevent division by zero
    cost_matrix_scaled = cost_matrix / max_cost_per_row

    max_delay_per_row = delay_matrix.max(axis=1, keepdims=True)
    max_delay_per_row[max_delay_per_row == 0] = 1  # Prevent division by zero
    delay_matrix_scaled = delay_matrix / max_delay_per_row

    # Compute cumulative probabilities for cost and delay matrices
    cumulative_cost_probabilities = compute_cumulative_probabilities(cost_matrix)
    cumulative_delay_probabilities = compute_cumulative_probabilities(delay_matrix)

    # Create labels for cost and delay bins
    cost_labels = np.array([min_cost + i * cost_bin_size for i in range(cost_matrix.shape[1])])
    delay_labels = np.array([min_delay + i * delay_bin_size for i in range(delay_matrix.shape[1])])

    # Compute P90 for cost and delay, starting from day 1
    cost_p90 = compute_p90(cumulative_cost_probabilities, cost_labels)
    delay_p90 = compute_p90(cumulative_delay_probabilities, delay_labels)

    # Plot and save P90 values for cost over project days
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(cost_p90) + 1), cost_p90, color='blue', linestyle='-', label='P90 Cost')  # Start from project day 1
    plt.title("P90 Cost Values Over Project Days")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Cost Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}p90_cost_over_project_days.png')
    plt.close()

    # Plot and save P90 values for delay over project days
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(delay_p90) + 1), delay_p90, color='green', linestyle='-', label='P90 Delay')  # Start from project day 1
    plt.title("P90 Delay Values Over Project Days")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Delay Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_path}p90_delay_over_project_days.png')
    plt.close()

    # Plot and save scaled cumulative cost distribution
    plt.figure(figsize=(12, 6))
    plt.imshow(cost_matrix_scaled, aspect='auto', origin='lower', cmap=heatmap_cmap)
    plt.colorbar(label='Scaled Frequency')
    plt.title("Scaled Cumulative Cost Distribution Over Time")
    plt.xlabel("Cumulative Cost")
    plt.ylabel("Project Day")

    # Set x-axis ticks
    num_cost_ticks = min(10, len(cost_labels))
    plt.xticks(np.linspace(0, len(cost_labels) - 1, num_cost_ticks).astype(int),
               [f"{int(label)}" for label in np.linspace(min_cost, cost_labels[-1], num_cost_ticks)],
               rotation=45)

    # Set y-axis ticks
    num_days_ticks = min(10, cost_matrix.shape[0])
    plt.yticks(np.linspace(0, cost_matrix.shape[0] - 1, num_days_ticks).astype(int),
               np.linspace(0, cost_matrix.shape[0] - 1, num_days_ticks).astype(int))

    # Add steady state line
    plt.axhline(y=cost_steady_state, color='w', linestyle='--')
    plt.text(0, cost_steady_state, f'Steady State (Day {cost_steady_state})', 
             color='w', va='bottom', ha='left')

    plt.tight_layout()
    plt.savefig(f'{save_path}scaled_cumulative_cost_distribution.png')
    plt.close()

    # Plot and save scaled cumulative delay distribution
    plt.figure(figsize=(12, 6))
    plt.imshow(delay_matrix_scaled, aspect='auto', origin='lower', cmap=heatmap_cmap)
    plt.colorbar(label='Scaled Frequency')
    plt.title("Scaled Cumulative Delay Distribution Over Time")
    plt.xlabel("Cumulative Delay")
    plt.ylabel("Project Day")

    # Set x-axis ticks
    num_delay_ticks = min(10, len(delay_labels))
    plt.xticks(np.linspace(0, len(delay_labels) - 1, num_delay_ticks).astype(int),
               [f"{int(label)}" for label in np.linspace(min_delay, delay_labels[-1], num_delay_ticks)],
               rotation=45)

    plt.yticks(np.linspace(0, delay_matrix.shape[0] - 1, num_days_ticks).astype(int),
               np.linspace(0, delay_matrix.shape[0] - 1, num_days_ticks).astype(int))

    # Add steady state line
    plt.axhline(y=delay_steady_state, color='w', linestyle='--')
    plt.text(0, delay_steady_state, f'Steady State (Day {delay_steady_state})', 
             color='w', va='bottom', ha='left')

    plt.tight_layout()
    plt.savefig(f'{save_path}scaled_cumulative_delay_distribution.png')
    plt.close()

    # Plot and save cumulative probability heatmap for cost
    plt.figure(figsize=(12, 6))
    plt.imshow(cumulative_cost_probabilities, aspect='auto', origin='lower', cmap=heatmap_cmap)
    plt.colorbar(label='Cumulative Probability')
    plt.title("Cumulative Cost Probability Distribution Over Time")
    plt.xlabel("Cumulative Cost")
    plt.ylabel("Project Day")

    # Set x-axis ticks
    plt.xticks(np.linspace(0, len(cost_labels) - 1, num_cost_ticks).astype(int),
               [f"{int(label)}" for label in np.linspace(min_cost, cost_labels[-1], num_cost_ticks)],
               rotation=45)

    plt.yticks(np.linspace(0, cost_matrix.shape[0] - 1, num_days_ticks).astype(int),
               np.linspace(0, cost_matrix.shape[0] - 1, num_days_ticks).astype(int))

    plt.tight_layout()
    plt.savefig(f'{save_path}cumulative_cost_probability_distribution.png')
    plt.close()

    # Plot and save cumulative probability heatmap for delay
    plt.figure(figsize=(12, 6))
    plt.imshow(cumulative_delay_probabilities, aspect='auto', origin='lower', cmap=heatmap_cmap)
    plt.colorbar(label='Cumulative Probability')
    plt.title("Cumulative Delay Probability Distribution Over Time")
    plt.xlabel("Cumulative Delay")
    plt.ylabel("Project Day")

    # Set x-axis ticks
    plt.xticks(np.linspace(0, len(delay_labels) - 1, num_delay_ticks).astype(int),
               [f"{int(label)}" for label in np.linspace(min_delay, delay_labels[-1], num_delay_ticks)],
               rotation=45)

    plt.yticks(np.linspace(0, delay_matrix.shape[0] - 1, num_days_ticks).astype(int),
               np.linspace(0, delay_matrix.shape[0] - 1, num_days_ticks).astype(int))

    plt.tight_layout()
    plt.savefig(f'{save_path}cumulative_delay_probability_distribution.png')
    plt.close()

    print(f"Cost distribution reaches steady state at day: {cost_steady_state}")
    print(f"Delay distribution reaches steady state at day: {delay_steady_state}")

    # Plot and save final cumulative cost distribution as a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(cost_labels, cost_matrix[-1, :], 
            width=cost_bin_size * 1.0, align='center', alpha=0.7, color='red')
    plt.title("Final Cumulative Cost Distribution")
    plt.xlabel("Cost")
    plt.ylabel("Cumulative Frequency")
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}final_cumulative_cost_distribution.png')
    plt.close()

    # Plot and save final cumulative delay distribution as a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(delay_labels, delay_matrix[-1, :], 
            width=delay_bin_size * 1.0, align='center', alpha=0.7, color='red')
    plt.title("Final Cumulative Delay Distribution")
    plt.xlabel("Delay")
    plt.ylabel("Cumulative Frequency")
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_path}final_cumulative_delay_distribution.png')
    plt.close()


