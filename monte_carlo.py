import random
import numpy as np
from scipy.stats import beta
from typing import List, Dict, Tuple
import copy

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

def run_monte_carlo_simulation(risks: List[Dict], num_iterations: int, max_days: int, max_cost: int, max_delay: int, cost_bin_size: int, delay_bin_size: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Runs a Monte Carlo simulation to produce cumulative frequency matrices for project costs and delays.
    
    The resulting matrices represent the number of iterations where the cumulative cost or delay
    reached a particular bin value by each day of the project timeline.
    """
    min_cost = -max_cost
    min_delay = -max_delay
    num_cost_bins = (max_cost - min_cost) // cost_bin_size + 1
    num_delay_bins = (max_delay - min_delay) // delay_bin_size + 1

    cumulative_cost_matrix = np.zeros((max_days + 1, num_cost_bins), dtype=int)
    cumulative_delay_matrix = np.zeros((max_days + 1, num_delay_bins), dtype=int)

    for _ in range(num_iterations):
        activated_risks = run_monte_carlo_iteration(risks)
        cumulative_cost = 0
        cumulative_delay = 0
        
        activated_risks.sort(key=lambda x: x["Actual Occurrence"] if x["Activated"] == "Yes" else float('inf'))
        
        # Initialize occurrence days from 1 to max_days
        for day in range(1, max_days + 1):
            # Check for risks that occur on the current day
            for risk in activated_risks:
                if risk["Activated"] == "Yes" and int(risk["Actual Occurrence"]) == day:
                    cumulative_cost += int(risk["Cost"])
                    cumulative_delay += int(risk["Delay"])

            # Ensure cumulative values don't exceed max values
            cumulative_cost = min(cumulative_cost, max_cost)
            cumulative_delay = min(cumulative_delay, max_delay)

            # Determine the cost bin index for this day
            if min_cost <= cumulative_cost <= max_cost:
                cost_bin_index = (cumulative_cost - min_cost) // cost_bin_size
                cumulative_cost_matrix[day, cost_bin_index] += 1

            # Determine the delay bin index for this day
            if min_delay <= cumulative_delay <= max_delay:
                delay_bin_index = (cumulative_delay - min_delay) // delay_bin_size
                cumulative_delay_matrix[day, delay_bin_index] += 1

    # Return the cumulative matrices
    return cumulative_cost_matrix, cumulative_delay_matrix, min_cost, min_delay


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

