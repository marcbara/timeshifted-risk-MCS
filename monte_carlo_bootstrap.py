# monte_carlo_bootstrap.py
# Self-contained module for Monte Carlo simulation with bootstrap analysis

import random
import numpy as np
from scipy.stats import beta
from typing import List, Dict, Tuple
import copy
import os
import matplotlib.pyplot as plt

def pert_random(min_val: float, most_likely: float, max_val: float, lambda_val: float = 4) -> float:
    """
    Generate a random number using a PERT distribution.
    """
    range_val = max_val - min_val
    mu = (min_val + lambda_val * most_likely + max_val) / (lambda_val + 2)
    
    if mu == most_likely:
        alpha = beta_val = lambda_val / 2 + 1
    else:
        alpha = ((mu - min_val) * (2 * most_likely - min_val - max_val)) / ((most_likely - mu) * (max_val - min_val))
        beta_val = alpha * (max_val - mu) / (mu - min_val)
    
    return min_val + beta.rvs(alpha, beta_val) * range_val

def run_monte_carlo_iteration(original_risks: List[Dict]) -> List[Dict]:
    """
    Run a single iteration of the Monte Carlo simulation.
    """
    risks = copy.deepcopy(original_risks)
    risk_details = []
    cumulative_delay = 0

    sorted_risks = sorted(risks, key=lambda x: x["Occurrence Time"])

    for risk in sorted_risks:
        actual_occurrence_time = risk["Occurrence Time"] + cumulative_delay
        current_probability = risk["Initial Probability"]
        is_activated = random.random() < current_probability
        
        cost = pert_random(*[float(x) if isinstance(x, str) else x for x in risk["Cost PERT"]]) if is_activated else 0
        delay = round(pert_random(*risk["Delay PERT"])) if is_activated else 0

        risk_details.append({
            "Risk ID": risk["Risk ID"],
            "Activated": "Yes" if is_activated else "No",
            "Used Probability": current_probability,
            "Original Occurrence": risk["Occurrence Time"],
            "Actual Occurrence": actual_occurrence_time if is_activated else "-",
            "Cost": cost,
            "Delay": delay
        })

        if is_activated:
            cumulative_delay += delay

            for dep_risk_id in risk["Dependent Risks"]:
                dep_risk = next((r for r in sorted_risks if r["Risk ID"] == dep_risk_id), None)
                if dep_risk:
                    dep_risk["Initial Probability"] += float(risk["Probability Adjustment"])
                    dep_risk["Initial Probability"] = max(0, min(1, dep_risk["Initial Probability"]))

    return risk_details

def run_monte_carlo_simulation_bootstrap(risks: List[Dict], num_iterations: int, max_days: int, max_cost: int, max_delay: int, cost_bin_size: int, delay_bin_size: int, num_bootstrap_blocks=100, bootstrap_save_path='bootstrap_results/', bootstrap_dpi=300) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Enhanced version of run_monte_carlo_simulation that includes bootstrap analysis.
    
    Parameters remain the same as the original function, with these additions:
    num_bootstrap_blocks : int, default=100
        Number of blocks to track for bootstrap analysis
    bootstrap_save_path : str, default='bootstrap_results/'
        Directory to save bootstrap results
    bootstrap_dpi : int, default=300
        Resolution for output images
    
    Returns:
    --------
    Same as the original function:
        tuple: (cumulative_cost_matrix, cumulative_delay_matrix, min_cost, min_delay)
    """
    # Initialize parameters (identical to original function)
    min_cost = -max_cost
    min_delay = -max_delay
    num_cost_bins = (max_cost - min_cost) // cost_bin_size + 1
    num_delay_bins = (max_delay - min_delay) // delay_bin_size + 1

    cumulative_cost_matrix = np.zeros((max_days + 1, num_cost_bins), dtype=int)
    cumulative_delay_matrix = np.zeros((max_days + 1, num_delay_bins), dtype=int)

    # Initialize bootstrap tracking
    block_size = num_iterations // num_bootstrap_blocks
    if block_size < 50:
        print(f"WARNING: Small block size ({block_size}). Consider reducing num_bootstrap_blocks or increasing num_iterations.")
    print(f"Tracking {num_bootstrap_blocks} blocks with ~{block_size} iterations each for bootstrap analysis.")
        
    # Initialize block tracking variables
    block_cost_matrices = []
    block_delay_matrices = []
    current_block = 0
    current_block_count = 0
    current_block_cost_matrix = np.zeros((max_days + 1, num_cost_bins), dtype=int)
    current_block_delay_matrix = np.zeros((max_days + 1, num_delay_bins), dtype=int)

    # Main simulation loop (enhanced with block tracking)
    for iteration in range(num_iterations):
        # Run a single iteration (identical to original)
        activated_risks = run_monte_carlo_iteration(risks)
        cumulative_cost = 0
        cumulative_delay = 0
        
        activated_risks.sort(key=lambda x: x["Actual Occurrence"] if x["Activated"] == "Yes" else float('inf'))
        
        # Process each day (similar to original)
        for day in range(1, max_days + 1):
            # Check for risks that occur on the current day
            for risk in activated_risks:
                if risk["Activated"] == "Yes" and int(risk["Actual Occurrence"]) == day:
                    cumulative_cost += int(risk["Cost"])
                    cumulative_delay += int(risk["Delay"])

            # Ensure cumulative values don't exceed max values
            cumulative_cost = min(cumulative_cost, max_cost)
            cumulative_delay = min(cumulative_delay, max_delay)

            # Update main matrices (identical to original)
            if min_cost <= cumulative_cost <= max_cost:
                cost_bin_index = (cumulative_cost - min_cost) // cost_bin_size
                cumulative_cost_matrix[day, cost_bin_index] += 1
                # Also track for current block
                current_block_cost_matrix[day, cost_bin_index] += 1

            if min_delay <= cumulative_delay <= max_delay:
                delay_bin_index = (cumulative_delay - min_delay) // delay_bin_size
                cumulative_delay_matrix[day, delay_bin_index] += 1
                # Also track for current block
                current_block_delay_matrix[day, delay_bin_index] += 1
        
        # Handle block completion
        current_block_count += 1
        
        # Check if we've completed a block
        if current_block_count >= block_size:
            # Store current block
            block_cost_matrices.append(current_block_cost_matrix.copy())
            block_delay_matrices.append(current_block_delay_matrix.copy())
            
            # Reset for next block
            current_block += 1
            current_block_count = 0
            current_block_cost_matrix = np.zeros((max_days + 1, num_cost_bins), dtype=int)
            current_block_delay_matrix = np.zeros((max_days + 1, num_delay_bins), dtype=int)
            
            if current_block % 10 == 0:
                print(f"Completed {current_block}/{num_bootstrap_blocks} blocks")
        
        # Progress indicator (same as original)
        if (iteration + 1) % (num_iterations // 10) == 0:
            print(f"Completed {iteration + 1}/{num_iterations} iterations ({(iteration + 1) / num_iterations * 100:.1f}%)")
    
    # Store last block if it has any data
    if current_block_count > 0:
        block_cost_matrices.append(current_block_cost_matrix.copy())
        block_delay_matrices.append(current_block_delay_matrix.copy())
    
    # Perform bootstrap analysis
    print("\nPerforming bootstrap analysis...")
    
    # Import necessary visualization functions directly (to avoid circular imports)
    from visualization import compute_cumulative_probabilities, compute_p90, apply_pmj_settings
    
    # Create cost and delay bin labels
    cost_labels = np.array([min_cost + i * cost_bin_size for i in range(num_cost_bins)])
    delay_labels = np.array([min_delay + i * delay_bin_size for i in range(num_delay_bins)])
    
    # Calculate original P90 values
    original_cumulative_cost_prob = compute_cumulative_probabilities(cumulative_cost_matrix)
    original_cumulative_delay_prob = compute_cumulative_probabilities(cumulative_delay_matrix)
    
    original_cost_p90 = compute_p90(original_cumulative_cost_prob, cost_labels)
    original_delay_p90 = compute_p90(original_cumulative_delay_prob, delay_labels)
    
    # Perform bootstrap resampling
    num_blocks = len(block_cost_matrices)
    num_bootstrap_samples = 1000  # Number of bootstrap samples to generate
    
    # Initialize arrays to store bootstrap P90 curves
    cost_p90_bootstrap = np.zeros((num_bootstrap_samples, len(original_cost_p90)))
    delay_p90_bootstrap = np.zeros((num_bootstrap_samples, len(original_delay_p90)))
    
    print(f"Generating {num_bootstrap_samples} bootstrap samples from {num_blocks} blocks...")
    
    for sample in range(num_bootstrap_samples):
        # Randomly select blocks with replacement
        block_indices = np.random.randint(0, num_blocks, size=num_blocks)
        
        # Combine selected blocks
        bootstrap_cost_matrix = np.zeros_like(block_cost_matrices[0])
        bootstrap_delay_matrix = np.zeros_like(block_delay_matrices[0])
        
        for idx in block_indices:
            bootstrap_cost_matrix += block_cost_matrices[idx]
            bootstrap_delay_matrix += block_delay_matrices[idx]
        
        # Compute P90 values for this bootstrap sample
        bootstrap_cumulative_cost_prob = compute_cumulative_probabilities(bootstrap_cost_matrix)
        bootstrap_cumulative_delay_prob = compute_cumulative_probabilities(bootstrap_delay_matrix)
        
        bootstrap_cost_p90 = compute_p90(bootstrap_cumulative_cost_prob, cost_labels)
        bootstrap_delay_p90 = compute_p90(bootstrap_cumulative_delay_prob, delay_labels)
        
        # Store the P90 curves
        cost_p90_bootstrap[sample, :len(bootstrap_cost_p90)] = bootstrap_cost_p90
        delay_p90_bootstrap[sample, :len(bootstrap_delay_p90)] = bootstrap_delay_p90
        
        if (sample + 1) % (num_bootstrap_samples // 10) == 0:
            print(f"Processed {sample + 1}/{num_bootstrap_samples} bootstrap samples")
    
    # Calculate statistics across bootstrap samples
    cost_p90_median = np.median(cost_p90_bootstrap, axis=0)
    delay_p90_median = np.median(delay_p90_bootstrap, axis=0)
    
    cost_p90_lower = np.percentile(cost_p90_bootstrap, 2.5, axis=0)
    cost_p90_upper = np.percentile(cost_p90_bootstrap, 97.5, axis=0)
    
    delay_p90_lower = np.percentile(delay_p90_bootstrap, 2.5, axis=0)
    delay_p90_upper = np.percentile(delay_p90_bootstrap, 97.5, axis=0)
    
    # Ensure output directory exists
    os.makedirs(bootstrap_save_path, exist_ok=True)
    
    # Save bootstrap data
    np.savez(
        f"{bootstrap_save_path}bootstrap_results.npz",
        cost_p90_bootstrap=cost_p90_bootstrap,
        delay_p90_bootstrap=delay_p90_bootstrap,
        cost_p90_median=cost_p90_median,
        delay_p90_median=delay_p90_median,
        cost_p90_lower=cost_p90_lower,
        cost_p90_upper=cost_p90_upper,
        delay_p90_lower=delay_p90_lower,
        delay_p90_upper=delay_p90_upper,
        original_cost_p90=original_cost_p90,
        original_delay_p90=original_delay_p90
    )
    
    # Plot the results - matching your visualization style for P90 curves
    # Color version of cost P90 bootstrap
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    days = np.arange(1, len(original_cost_p90) + 1)
    
    # Color version - blue for bootstrap lines
    plt.plot(days, cost_p90_median, color='blue', linestyle='-', linewidth=1.5, 
             label='Bootstrap Median P90 Cost')
    # Add CI as a shaded area
    plt.fill_between(days, cost_p90_lower, cost_p90_upper, color='blue', alpha=0.2, 
                     label='95% Confidence Interval')
    # Add original P90 for comparison
    plt.plot(days, original_cost_p90, color='red', linestyle='-', linewidth=1.5, 
             label='Original P90 Cost')
    
    plt.title("Bootstrap Analysis of P90 Cost Values")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Cost Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save color version
    plt.savefig(f"{bootstrap_save_path}bootstrap_p90_cost.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Grayscale version of cost P90 bootstrap
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    # Grayscale version - black for main line
    plt.plot(days, cost_p90_median, color='black', linestyle='-', linewidth=1.5, 
             label='Bootstrap Median P90 Cost')
    # Add CI as a shaded area in gray
    plt.fill_between(days, cost_p90_lower, cost_p90_upper, color='gray', alpha=0.3, 
                     label='95% Confidence Interval')
    # Add original P90 for comparison (dashed in grayscale)
    plt.plot(days, original_cost_p90, color='black', linestyle='--', linewidth=1.5, 
             label='Original P90 Cost')
    
    plt.title("Bootstrap Analysis of P90 Cost Values")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Cost Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save grayscale version
    plt.savefig(f"{bootstrap_save_path}bootstrap_p90_cost_grayscale.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Delay P90 bootstrap plot
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    days = np.arange(1, len(original_delay_p90) + 1)
    
    # Color version - blue for bootstrap lines
    plt.plot(days, delay_p90_median, color='blue', linestyle='-', linewidth=1.5, 
             label='Bootstrap Median P90 Delay')
    # Add CI as a shaded area
    plt.fill_between(days, delay_p90_lower, delay_p90_upper, color='blue', alpha=0.2, 
                     label='95% Confidence Interval')
    # Add original P90 for comparison
    plt.plot(days, original_delay_p90, color='red', linestyle='-', linewidth=1.5, 
             label='Original P90 Delay')
    
    plt.title("Bootstrap Analysis of P90 Delay Values")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Delay Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save color version
    plt.savefig(f"{bootstrap_save_path}bootstrap_p90_delay.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Grayscale version of delay P90 bootstrap
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    # Grayscale version - black for main line
    plt.plot(days, delay_p90_median, color='black', linestyle='-', linewidth=1.5, 
             label='Bootstrap Median P90 Delay')
    # Add CI as a shaded area in gray
    plt.fill_between(days, delay_p90_lower, delay_p90_upper, color='gray', alpha=0.3, 
                     label='95% Confidence Interval')
    # Add original P90 for comparison (dashed in grayscale)
    plt.plot(days, original_delay_p90, color='black', linestyle='--', linewidth=1.5, 
             label='Original P90 Delay')
    
    plt.title("Bootstrap Analysis of P90 Delay Values")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Delay Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save grayscale version
    plt.savefig(f"{bootstrap_save_path}bootstrap_p90_delay_grayscale.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Also create plots showing all bootstrap curves (optional visualization)
    # Color version of all cost bootstrap curves
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    # Plot a subset of bootstrap curves for readability
    max_curves = 50
    num_samples = min(max_curves, cost_p90_bootstrap.shape[0])
    indices = np.linspace(0, cost_p90_bootstrap.shape[0] - 1, num_samples, dtype=int)
    
    # Plot individual bootstrap curves in light blue
    for i in indices:
        plt.plot(days, cost_p90_bootstrap[i], color='blue', alpha=0.1, linewidth=0.5)
    
    # Highlight median and original
    plt.plot(days, cost_p90_median, color='blue', linestyle='-', linewidth=2, 
             label='Bootstrap Median P90')
    plt.plot(days, original_cost_p90, color='red', linestyle='-', linewidth=2, 
             label='Original P90')
    
    plt.title("All Bootstrap P90 Cost Curves")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Cost Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save color version
    plt.savefig(f"{bootstrap_save_path}all_bootstrap_p90_cost.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Grayscale version of all cost bootstrap curves
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    # Plot individual bootstrap curves in light gray
    for i in indices:
        plt.plot(days, cost_p90_bootstrap[i], color='lightgray', alpha=0.2, linewidth=0.5)
    
    # Highlight median and original
    plt.plot(days, cost_p90_median, color='black', linestyle='-', linewidth=2, 
             label='Bootstrap Median P90')
    plt.plot(days, original_cost_p90, color='black', linestyle='--', linewidth=2, 
             label='Original P90')
    
    plt.title("All Bootstrap P90 Cost Curves")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Cost Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save grayscale version
    plt.savefig(f"{bootstrap_save_path}all_bootstrap_p90_cost_grayscale.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Color version of all delay bootstrap curves
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    # Plot a subset of bootstrap curves for readability
    num_samples = min(max_curves, delay_p90_bootstrap.shape[0])
    indices = np.linspace(0, delay_p90_bootstrap.shape[0] - 1, num_samples, dtype=int)
    
    # Plot individual bootstrap curves in light blue
    for i in indices:
        plt.plot(days, delay_p90_bootstrap[i], color='blue', alpha=0.1, linewidth=0.5)
    
    # Highlight median and original
    plt.plot(days, delay_p90_median, color='blue', linestyle='-', linewidth=2, 
             label='Bootstrap Median P90')
    plt.plot(days, original_delay_p90, color='red', linestyle='-', linewidth=2, 
             label='Original P90')
    
    plt.title("All Bootstrap P90 Delay Curves")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Delay Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save color version
    plt.savefig(f"{bootstrap_save_path}all_bootstrap_p90_delay.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Grayscale version of all delay bootstrap curves
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)
    
    # Plot individual bootstrap curves in light gray
    for i in indices:
        plt.plot(days, delay_p90_bootstrap[i], color='lightgray', alpha=0.2, linewidth=0.5)
    
    # Highlight median and original
    plt.plot(days, delay_p90_median, color='black', linestyle='-', linewidth=2, 
             label='Bootstrap Median P90')
    plt.plot(days, original_delay_p90, color='black', linestyle='--', linewidth=2, 
             label='Original P90')
    
    plt.title("All Bootstrap P90 Delay Curves")
    plt.xlabel("Project Day")
    plt.ylabel("P90 Delay Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save grayscale version
    plt.savefig(f"{bootstrap_save_path}all_bootstrap_p90_delay_grayscale.png", dpi=bootstrap_dpi)
    plt.close()
    
    # Print summary statistics
    print("\nBootstrap Analysis Summary:")
    print(f"P90 Cost at end (median): {cost_p90_median[-1]:.2f}")
    print(f"P90 Cost 95% CI: [{cost_p90_lower[-1]:.2f}, {cost_p90_upper[-1]:.2f}]")
    print(f"P90 Delay at end (median): {delay_p90_median[-1]:.2f}")
    print(f"P90 Delay 95% CI: [{delay_p90_lower[-1]:.2f}, {delay_p90_upper[-1]:.2f}]")
    print(f"Generated all bootstrap visualizations with {bootstrap_dpi} DPI resolution")
    print(f"Grayscale versions created for PMJ submission")
    
    # Calculate CI half-widths as percentage of median - with safe division
    # Create mask for non-zero median values
    cost_mask = cost_p90_median != 0
    delay_mask = delay_p90_median != 0

    # Initialize arrays with zeros
    cost_p90_half_width_percent = np.zeros_like(cost_p90_median)
    delay_p90_half_width_percent = np.zeros_like(delay_p90_median)

    # Calculate only where median is not zero
    cost_p90_half_width_percent[cost_mask] = ((cost_p90_upper[cost_mask] - cost_p90_lower[cost_mask]) / 2) / cost_p90_median[cost_mask] * 100
    delay_p90_half_width_percent[delay_mask] = ((delay_p90_upper[delay_mask] - delay_p90_lower[delay_mask]) / 2) / delay_p90_median[delay_mask] * 100

    # Find the maximum half-width percentage (ignoring zeros)
    max_cost_half_width_percent = np.max(cost_p90_half_width_percent[cost_mask]) if np.any(cost_mask) else 0
    max_delay_half_width_percent = np.max(delay_p90_half_width_percent[delay_mask]) if np.any(delay_mask) else 0

    # Add the precision analysis as a separate section
    print("\nPrecision Analysis:")
    print(f"Maximum CI half-width for P90 Cost: {max_cost_half_width_percent:.2f}% of median")
    print(f"Maximum CI half-width for P90 Delay: {max_delay_half_width_percent:.2f}% of median")
    print(f"Final CI half-width for P90 Cost: {cost_p90_half_width_percent[-1]:.2f}% of median")
    print(f"Final CI half-width for P90 Delay: {delay_p90_half_width_percent[-1]:.2f}% of median")



    # Add this to the end of your run_monte_carlo_simulation_bootstrap function,
    # after the bootstrap analysis but before returning results

    # Revised convergence analysis using cumulative iterations and bin size-based thresholds
    # without automated convergence detection
    print("\nPerforming convergence analysis using cumulative iterations...")

    # We need to work with cumulative matrices instead of individual block matrices
    cumulative_cost_matrices = []
    cumulative_delay_matrices = []

    # First cumulative matrix is just the first block
    cumulative_cost_matrices.append(block_cost_matrices[0].copy())
    cumulative_delay_matrices.append(block_delay_matrices[0].copy())

    # Build cumulative matrices by adding each block
    for i in range(1, len(block_cost_matrices)):
        # Add this block to the previous cumulative matrix
        cum_cost = cumulative_cost_matrices[-1] + block_cost_matrices[i]
        cum_delay = cumulative_delay_matrices[-1] + block_delay_matrices[i]
        
        cumulative_cost_matrices.append(cum_cost)
        cumulative_delay_matrices.append(cum_delay)
        
        if i % 10 == 0:
            print(f"Created {i}/{len(block_cost_matrices)-1} cumulative matrices")

    # Calculate P90 for each cumulative matrix
    cumulative_cost_p90s = []
    cumulative_delay_p90s = []
    cumulative_iterations = []

    for i, (cost_matrix, delay_matrix) in enumerate(zip(cumulative_cost_matrices, cumulative_delay_matrices)):
        # Calculate iterations for this cumulative matrix
        iterations = block_size * (i + 1)
        cumulative_iterations.append(iterations)
        
        # Calculate P90 values
        cum_cost_prob = compute_cumulative_probabilities(cost_matrix)
        cum_delay_prob = compute_cumulative_probabilities(delay_matrix)
        
        cum_cost_p90 = compute_p90(cum_cost_prob, cost_labels)
        cum_delay_p90 = compute_p90(cum_delay_prob, delay_labels)
        
        cumulative_cost_p90s.append(cum_cost_p90)
        cumulative_delay_p90s.append(cum_delay_p90)
        
        if i % 10 == 0:
            print(f"Calculated P90 for {i+1}/{len(cumulative_cost_matrices)} cumulative matrices")

    # Calculate delta_infinity between consecutive cumulative P90 curves
    delta_infinity_cost = []
    delta_infinity_delay = []
    plot_iterations = []  # For plotting (will be iterations where we calculate delta)

    for i in range(1, len(cumulative_cost_p90s)):
        # Calculate absolute differences
        delta_cost = np.max(np.abs(cumulative_cost_p90s[i] - cumulative_cost_p90s[i-1]))
        delta_delay = np.max(np.abs(cumulative_delay_p90s[i] - cumulative_delay_p90s[i-1]))
        
        # Normalize by baseline (final P90 value from largest cumulative matrix)
        baseline_cost = cumulative_cost_p90s[-1][-1] if cumulative_cost_p90s[-1][-1] != 0 else 1
        baseline_delay = cumulative_delay_p90s[-1][-1] if cumulative_delay_p90s[-1][-1] != 0 else 1
        
        # Calculate percentage
        delta_infinity_cost.append((delta_cost / baseline_cost) * 100)
        delta_infinity_delay.append((delta_delay / baseline_delay) * 100)
        
        # Store the iteration where this delta is calculated (the larger of the two)
        plot_iterations.append(cumulative_iterations[i])

    # Calculate bin size-based thresholds
    final_p90_cost = cumulative_cost_p90s[-1][-1]
    final_p90_delay = cumulative_delay_p90s[-1][-1]

    # Calculate what one bin represents as percentage of final P90
    cost_bin_percent = (cost_bin_size / final_p90_cost) * 100
    delay_bin_percent = (delay_bin_size / final_p90_delay) * 100

    # Set thresholds to 1.5 times the bin size percentage
    cost_threshold = 1.5 * cost_bin_percent
    delay_threshold = 1.5 * delay_bin_percent

    print(f"\nBin-based thresholds:")
    print(f"Cost bin size ({cost_bin_size}) is {cost_bin_percent:.2f}% of P90 cost")
    print(f"Using cost convergence threshold of {cost_threshold:.2f}%")
    print(f"Delay bin size ({delay_bin_size}) is {delay_bin_percent:.2f}% of P90 delay")
    print(f"Using delay convergence threshold of {delay_threshold:.2f}%")

    # Plot cost convergence
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)

    plt.plot(plot_iterations, delta_infinity_cost, 'b-', linewidth=1.5, label='Cost P90 Δ∞')
    plt.axhline(y=cost_threshold, color='r', linestyle='--', linewidth=1.5, 
            label=f'{cost_threshold:.2f}% Threshold (1.5 × bin size)')

    plt.title('Convergence Analysis: Maximum Change in P90 Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Δ∞ (% of baseline P90)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save color version
    plt.savefig(f"{bootstrap_save_path}cost_convergence.png", dpi=bootstrap_dpi)
    plt.close()

    # Grayscale version
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)

    plt.plot(plot_iterations, delta_infinity_cost, 'k-', linewidth=1.5, label='Cost P90 Δ∞')
    plt.axhline(y=cost_threshold, color='k', linestyle='--', linewidth=1.5, 
            label=f'{cost_threshold:.2f}% Threshold (1.5 × bin size)')

    plt.title('Convergence Analysis: Maximum Change in P90 Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Δ∞ (% of baseline P90)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save grayscale version
    plt.savefig(f"{bootstrap_save_path}cost_convergence_grayscale.png", dpi=bootstrap_dpi)
    plt.close()

    # Plot delay convergence
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)

    plt.plot(plot_iterations, delta_infinity_delay, 'b-', linewidth=1.5, label='Delay P90 Δ∞')
    plt.axhline(y=delay_threshold, color='r', linestyle='--', linewidth=1.5, 
            label=f'{delay_threshold:.2f}% Threshold (1.5 × bin size)')

    plt.title('Convergence Analysis: Maximum Change in P90 Delay')
    plt.xlabel('Iterations')
    plt.ylabel('Δ∞ (% of baseline P90)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save color version
    plt.savefig(f"{bootstrap_save_path}delay_convergence.png", dpi=bootstrap_dpi)
    plt.close()

    # Grayscale version
    apply_pmj_settings()
    plt.figure(figsize=(12, 6), dpi=bootstrap_dpi)

    plt.plot(plot_iterations, delta_infinity_delay, 'k-', linewidth=1.5, label='Delay P90 Δ∞')
    plt.axhline(y=delay_threshold, color='k', linestyle='--', linewidth=1.5, 
            label=f'{delay_threshold:.2f}% Threshold (1.5 × bin size)')

    plt.title('Convergence Analysis: Maximum Change in P90 Delay')
    plt.xlabel('Iterations')
    plt.ylabel('Δ∞ (% of baseline P90)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save grayscale version
    plt.savefig(f"{bootstrap_save_path}delay_convergence_grayscale.png", dpi=bootstrap_dpi)
    plt.close()

    print("\nConvergence Analysis:")
    print(f"Plots generated - examine visually to determine convergence point")
    
    return cumulative_cost_matrix, cumulative_delay_matrix, min_cost, min_delay

# Add analyze_single_iteration for completeness
def analyze_single_iteration(risks: List[Dict], iteration_number: int = 1) -> List[Dict]:
    """
    Analyze and display the results of a single Monte Carlo iteration.
    """
    risk_details = run_monte_carlo_iteration(risks)

    from tabulate import tabulate

    table_data = [
        [
            detail["Risk ID"],
            detail["Activated"],
            f"{detail['Used Probability']:.2f}",
            detail["Original Occurrence"],
            detail["Actual Occurrence"],
            f"{detail['Cost']:.2f}" if detail["Activated"] == "Yes" else "-",
            detail["Delay"] if detail["Activated"] == "Yes" else "-"
        ]
        for detail in risk_details
    ]

    headers = ["Risk ID", "Activated", "Used Probability", "Original Day", "Actual Day", "Cost", "Delay"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")

    print(f"\nDetailed Analysis of Iteration {iteration_number}:")
    print(table)

    return risk_details