from monte_carlo import run_monte_carlo_simulation, analyze_single_iteration
from risk_data import read_and_validate_data
from visualization import plot_cumulative_distributions
from excel_output import save_matrices_to_excel
from config import load_config

def main():
    config = load_config()
    
    risks = read_and_validate_data(config['risk_data_file'])
    if risks is None:
        print("Error in data. Exiting.")
        return

    # Extract only the relevant parameters for run_monte_carlo_simulation
    simulation_params = {
        'num_iterations': config['num_iterations'],
        'max_days': config['max_days'],
        'max_cost': config['max_cost'],
        'max_delay': config['max_delay'],
        'cost_bin_size': config['cost_bin_size'],
        'delay_bin_size': config['delay_bin_size']
    }

    # Run the full Monte Carlo simulation
    print(f"\nTotal Iterations: {config['num_iterations']}")
    cumulative_cost_matrix, cumulative_delay_matrix, min_cost, min_delay = run_monte_carlo_simulation(risks, **simulation_params)


    # Save results and create visualizations
    # save_matrices_to_excel(cumulative_cost_matrix, cumulative_delay_matrix, config['cost_bin_size'], config['delay_bin_size'], 
    #                        min_cost, min_delay, config['output_file'])
    
    plot_cumulative_distributions(cumulative_cost_matrix, cumulative_delay_matrix, config['cost_bin_size'], 
                                  config['delay_bin_size'], min_cost, min_delay, save_path='png/', create_grayscale=True, dpi=300)

    # Analyze a single iteration
    #analyze_single_iteration(risks, iteration_number=1)

if __name__ == "__main__":
    main()