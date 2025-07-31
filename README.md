# Enhanced Monte Carlo Simulation for Project Risk Analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1080%2Fxxxx-blue.svg)](https://doi.org/10.1080/xxxx)

**Version 3.0** - Enhanced Monte Carlo simulation framework for comprehensive project risk analysis integrating cost and schedule impacts with time-shifted risks and dependency modeling.

## 👨‍💻 Author

**Marc Bara Iniesta**  
*Creator and Lead Developer*

## 📄 Publication

This repository contains the complete implementation for the paper:

**"Enhanced Monte Carlo Simulation for Project Risk Analysis: Integrating Cost and Schedule Impacts with Time-Shifted Risks and Dependency Modeling"**

*By Marc Bara Iniesta*  
*Published in Applied Operations and Analytics (Taylor & Francis)*

## 🎯 Overview

This advanced Monte Carlo simulation framework provides a comprehensive solution for project risk analysis, featuring:

- **Time-shifted risk modeling** with dynamic occurrence timing
- **Interdependent risk cascading** with probability adjustments
- **Dual-impact assessment** for both cost and schedule impacts
- **Bootstrap validation** with convergence analysis
- **PERT distribution modeling** for realistic uncertainty representation
- **Publication-ready visualizations** with both color and grayscale outputs

## ✨ Key Features

### 🔬 Advanced Simulation Engine
- **Monte Carlo Bootstrap**: Statistical validation with configurable bootstrap blocks
- **PERT Distributions**: Beta-PERT modeling for cost and delay uncertainties
- **Time-shifted Risks**: Dynamic risk occurrence with cumulative delay impacts
- **Dependency Modeling**: Risk cascading with probability adjustments

### 📊 Comprehensive Analytics
- **P90 Risk Curves**: Time series analysis of 90th percentile values
- **Percentile Analysis**: P50, P75, P90, P95, P99 calculations with interpolation
- **Steady State Detection**: Automatic identification of convergence points
- **Convergence Analysis**: Bootstrap-based simulation convergence validation

### 📈 Professional Visualizations
- **Heatmap Distributions**: Time-based cumulative probability matrices
- **Final Distribution Histograms**: Terminal project state analysis
- **P90 Time Series**: Evolution of risk metrics over project timeline
- **Scaled Frequency Maps**: Normalized distribution visualizations
- **Journal-ready Outputs**: 300 DPI grayscale versions for publication

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib pandas openpyxl scipy
```

### Basic Usage
```bash
# Run the complete simulation
python main.py
```

### Configuration
Edit `config.ini` to customize simulation parameters:

```ini
[Simulation]
num_iterations = 20000      # Monte Carlo iterations
max_days = 700             # Project duration (days)
max_cost = 150000          # Maximum cost impact
max_delay = 300            # Maximum delay impact (days)
cost_bin_size = 1000       # Cost histogram bin size
delay_bin_size = 1         # Delay histogram bin size

[Files]
risk_data_file = risk_data.xlsx
output_file = simulation_results.xlsx
```

## 📁 Project Structure

```
timeshifted-risk-MCS/
├── main.py                    # Main execution script
├── config.ini                 # Simulation configuration
├── config.py                  # Configuration loader
├── monte_carlo_bootstrap.py   # Enhanced simulation engine
├── monte_carlo.py            # Core simulation functions
├── risk_data.py              # Data validation and loading
├── visualization.py          # Plotting and visualization
├── excel_output.py           # Excel export functionality
├── create_risk_data.py       # Sample data generation
├── check_risk_data.py        # Data validation utility
├── risk_data.xlsx            # Input risk data
├── simulation_results.xlsx   # Simulation outputs
├── png/                      # Generated visualizations
└── bootstrap_results/        # Bootstrap analysis outputs
```

## 📋 Input Data Format

Risk data should be provided in Excel format (`risk_data.xlsx`) with the following columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| Risk ID | Unique identifier | String | R001 |
| Description | Risk description | String | Equipment failure |
| Initial Probability | Base occurrence probability | Float | 0.15 |
| Occurrence Time (Day) | Planned occurrence day | Integer | 45 |
| Cost PERT (min) | Minimum cost impact | Float | 5000 |
| Cost PERT (most likely) | Most likely cost impact | Float | 8000 |
| Cost PERT (max) | Maximum cost impact | Float | 15000 |
| Delay PERT (min) | Minimum delay impact | Integer | 2 |
| Delay PERT (most likely) | Most likely delay impact | Integer | 5 |
| Delay PERT (max) | Maximum delay impact | Integer | 10 |
| Dependent Risks | Comma-separated dependent risk IDs | String | R002,R003 |
| Probability Adjustment | Probability increase for dependent risks | Float | 0.1 |

## 📊 Generated Outputs

### Visualizations (PNG format)
- **P90 Time Series**: Cost and delay evolution over project timeline
- **Cumulative Probability Heatmaps**: Time-based distribution matrices
- **Scaled Distribution Heatmaps**: Normalized frequency analysis
- **Final Distribution Histograms**: Terminal state probability distributions

### Data Outputs
- **Excel Results**: Complete simulation matrices and summary statistics
- **Bootstrap Analysis**: Convergence validation and statistical metrics
- **Console Statistics**: Real-time percentile calculations and convergence metrics

### Publication Features
- **Dual Format Output**: Color versions for analysis, grayscale for publication
- **High Resolution**: 300 DPI output for journal requirements
- **Formatted Axes**: Thousands separators for readability
- **Statistical Annotations**: Automatic percentile calculations and reporting

## 🔧 Advanced Configuration

### Bootstrap Parameters
```python
# In monte_carlo_bootstrap.py
num_bootstrap_blocks = 100      # Number of bootstrap samples
bootstrap_save_path = 'bootstrap_results/'
bootstrap_dpi = 300            # Output resolution
```

### Visualization Settings
```python
# Journal-ready settings automatically applied
font_family = 'Arial'          # Journal requirement
font_size = 10                 # 10pt for figures
dpi = 300                      # Publication quality
```

## 📈 Statistical Methods

### Risk Modeling
- **PERT Distributions**: Beta-PERT implementation with lambda parameter control
- **Time Shifting**: Dynamic occurrence timing based on cumulative delays
- **Dependency Cascading**: Probability adjustments for interconnected risks

### Analysis Techniques
- **Monte Carlo Sampling**: Extensive iteration-based uncertainty quantification
- **Bootstrap Validation**: Statistical confidence and convergence analysis
- **Percentile Interpolation**: Precise percentile calculation between histogram bins
- **Steady State Detection**: Automatic identification of simulation convergence

## 🤝 Contributing

We welcome contributions to enhance the simulation framework. Please ensure:

1. **Code Quality**: Follow existing style conventions
2. **Documentation**: Update docstrings and README as needed
3. **Testing**: Validate changes with provided sample data
4. **Academic Standards**: Maintain scientific rigor in methodological changes

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@article{enhanced_monte_carlo_2025,
  title={Enhanced Monte Carlo Simulation for Project Risk Analysis: Integrating Cost and Schedule Impacts with Time-Shifted Risks and Dependency Modeling},
  author={Bara Iniesta, Marc},
  journal={Applied Operations and Analytics},
  publisher={Taylor \& Francis},
  year={2025},
  doi={10.1080/xxxx}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [Applied Operations and Analytics](https://doi.org/10.1080/xxxx)
- **Repository**: [GitHub](https://github.com/marcbara/timeshifted-risk-MCS)
- **Issues**: [Report Issues](https://github.com/marcbara/timeshifted-risk-MCS/issues)

## 🏆 Acknowledgments

Developed by **Marc Bara Iniesta** as part of advanced research in project risk analysis and Monte Carlo simulation methodologies.

Special thanks to:
- Taylor & Francis - Applied Operations and Analytics
- Monte Carlo simulation methodology community  
- Python scientific computing ecosystem

---

**Version 3.0** - Enhanced for publication in Applied Operations and Analytics (Taylor & Francis)