# Analysis Toolkit for Portfolio Management

This Python module, `analysis_toolkit.py`, offers a comprehensive suite of tools for performing detailed analysis and optimization of investment portfolios. It integrates with common financial datasets and provides functions for calculating various financial metrics, optimizing portfolio allocations, and plotting efficient frontiers.

## Features:
- **Data Import Functions**: Easy import of Fama-French market equity returns, EDHEC Hedge Fund Index returns, and Ken French 30 Industry Portfolios.
- **Performance Measurement**: Functions to calculate drawdowns, skewness, kurtosis, and test for normality of returns.
- **Risk Measurement**: Includes calculation of semi-deviation, historic VaR, CVaR, and Gaussian VaR (with optional Cornish-Fisher modification).
- **Portfolio Optimization**: Tools for determining the Global Minimum Volatility (GMV) portfolio, the maximum Sharpe ratio portfolio, and efficient frontier visualizations.
- **CPPI Strategy**: Implementation of Constant Proportion Portfolio Insurance (CPPI) strategy with and without drawdown constraints and a maximum cap.
- **Monte Carlo Simulation**: Conduct Monte Carlo simulations with time-varying parameters to analyze portfolio performance under different market conditions.
- **Data Visualizations**: Enhanced visualizations to plot industry market capitalizations and their weights.

## How to Use:
1. **Installation**: Clone or download this repository. Ensure you have Python installed on your machine.
2. **Dependencies**: Install required Python packages including numpy, pandas, scipy, and matplotlib.
