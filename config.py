import configparser

def load_config(file_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    return {
        'num_iterations': config.getint('Simulation', 'num_iterations'),
        'max_days': config.getint('Simulation', 'max_days'),
        'max_cost': config.getint('Simulation', 'max_cost'),
        'max_delay': config.getint('Simulation', 'max_delay'),
        'cost_bin_size': config.getint('Simulation', 'cost_bin_size'),
        'delay_bin_size': config.getint('Simulation', 'delay_bin_size'),
        'risk_data_file': config.get('Files', 'risk_data_file'),
        'output_file': config.get('Files', 'output_file')
    }