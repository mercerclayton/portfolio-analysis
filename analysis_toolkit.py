# Importing necessary libraries
import numpy as np
import pandas as pd
import scipy.stats
from scipy.optimize import minimize


# Fama-French Market Equity
def get_ffme_returns():
    """
    Returns the Fama-French dataset of returns by market capitalization.
    """
    ffme = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0, na_values=-99.99)
    ffme = ffme / 100
    ffme.index = pd.to_datetime(ffme.index, format='%Y%m').to_period('M')
    
    return ffme


# Hedge Fund Index
def get_hfi_returns():
    """
    Returns the EDHEC Hedge Fund Index Returns.
    """
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', header=0, index_col=0)
    hfi = hfi / 100
    hfi.index = pd.to_datetime(hfi.index, format='%d/%m/%Y').to_period('M')
    
    return hfi


# Ken French 30 Industry Portfolios
def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


# Function expects a Series type object as input
def drawdown(return_series: pd.Series):
    """
    Returns the wealth index, cumulative peaks, and percentage drawdown for a given return series.
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    cumulative_peaks = wealth_index.cummax()
    drawdown = (wealth_index - cumulative_peaks) / cumulative_peaks
    
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Cumulative Peaks': cumulative_peaks,
        'Percentage Drawdown': drawdown
    })


def skewness(returns):
    """
    Alternative to scipy.stats.skew()
    Returns the skewness of the given Series or DataFrame.
    """
    demeaned_returns = returns - returns.mean()
    expected_returns = (demeaned_returns ** 3).mean()
    sigma = returns.std(ddof=0)
    
    return expected_returns / (sigma ** 3)


def kurtosis(returns):
    """
    Alternative to scipy.stats.kurtosis() (which returns the excess kurtosis)
    Returns the skewness of the given Series or DataFrame.
    """
    demeaned_returns = returns - returns.mean()
    expected_returns = (demeaned_returns ** 4).mean()
    sigma = returns.std(ddof=0)
    
    return expected_returns / (sigma ** 4)


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal, level=level)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def semi_deviation(r):
    """
    Returns the semi-deviation (negative deviation only) of a given return series.
    Must be given a Series or DataFrame else TypeError is raised.
    """
    negative = r < 0
    
    return r[negative].std(ddof=0)


# Default level is 5%
def var_historic(r, level=5):
    """
    Returns the historic VaR at a given level.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be a Series or DataFrame.')


# Default level is 5%
def cvar_historic(r, level=5):
    """
    Returns the conditional historic VaR (Expected Shortfall) at a given level.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        beyond_var = r <= -var_historic(r, level=level)
        return -r[beyond_var].mean()
    else:
        raise TypeError('Expected r to be a Series or DataFrame.')


# Default level is 5% and modified=False        
def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR at a given level.
    If modified is True, applies the Cornish-Fisher modification.
    """
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    
    return -(r.mean() + z * r.std(ddof=0))


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns.
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    
    return compounded_growth ** (periods_per_year/n_periods) - 1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns.
    """
    
    return r.std() * np.sqrt(periods_per_year)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Returns the annualized Sharpe ratio of a given set of returns.
    """
    rf_per_period = (1 + riskfree_rate) ** (1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ret = annualize_rets(excess_ret, periods_per_year)
    vol = annualize_vol(r, periods_per_year)

    return ret/vol


def portfolio_return(weights, returns):
    """
    Calculates the return on a portfolio from the returns and weights of the  underlying assets.
    weights and returns may either be a numpy array or Nx1 matrix.
    """
    
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Returns the volatility of a portfolio from the weights and covariance matrix of the underlying assets.
    weights are a numpy array or Nx1 matrix and covmat is a NxN matrix.
    """
    
    return np.sqrt(weights.T @ covmat @ weights)


def plot_ef2(n_points, er, cov):
    """
    Plots the two asset efficient frontier.
    """
    if er.shape[0] != 2:
        raise ValueError('plot_ef2 can only plot 2 asset frontiers.')
    weights = [np.array([w, 1-w]) for w in np.linspace(0.0, 1.0, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    
    return ef.plot.line(x='Volatility', y='Returns', style='.-')


def minimize_vol(target_return, er, cov):
    """
    Returns weights which minimize the volatility for a given target return.
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    
    weight_sum = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    target_return_equal = {
        'type': 'eq',
        'fun': lambda weights, er: target_return - portfolio_return(weights, er),
        'args': (er,)
    }
    
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       bounds=bounds, constraints=(weight_sum, target_return_equal),
                       options={'disp': False})
    
    return weights.x

    
def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio.
    """
    n = cov.shape[0]
    
    return msr(0, np.repeat(1, n), cov)


def optimal_weights(n_points, er, cov):
    """
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    
    return weights


def msr(riskfree_rate, er, cov):
    """
    Returns the weights for the maximum Sharpe ratio portfolio.
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    
    weight_sum = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the Sharpe ratio for a given portfolio.
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol

    weights = minimize(neg_sharpe_ratio, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       bounds=bounds, constraints=(weight_sum),
                       options={'disp': False})

    return weights.x


def plot_ef(n_points, er, cov, style='-', show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi asset efficient frontier.
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    
    ax = ef.plot.line(x='Volatility', y='Returns', style=style)
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
        
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
    
    return ax