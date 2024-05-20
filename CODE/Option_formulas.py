from scipy.stats import norm
import numpy as np

def forward_price(Spot, Strike, rate, div, TTM):
    """
    Calculate the forward price of an asset.

    Args:
    - Spot (float): Current price of the asset.
    - rate (float): Risk-free interest rate, as a decimal.
    - div (float): Dividend yield of the asset, as a decimal.
    - TTM (float): Time to maturity of the forward contract, in years.

    Returns:
    - float: The forward price of the asset.
    """
    return Spot * np.exp((rate - div) * TTM) - Strike * np.exp(rate * TTM)

def BlackScholes(Spot, Strike, TTM, rate, div, Vol, IsCall):
    """
    Calculate the Black-Scholes option pricing model.

    Args:
    - Spot (float): Current price of the underlying asset.
    - Strike (float): Strike price of the option.
    - TTM (float): Time to maturity of the option, in years.
    - rate (float): Risk-free interest rate, as a decimal.
    - div (float): Dividend yield of the asset, as a decimal.
    - Vol (float): Volatility of the asset's returns, as a decimal.
    - IsCall (bool): True if the option is a call option, False if it is a put option.

    Returns:
    - float: The Black-Scholes price of the option.
    """
    
    N = norm.cdf  # Standard normal cumulative distribution function

    if TTM > 0:  # Positive time to maturity
        # Calculation of d1 and d2 using Black-Scholes formula components
        d1 = (np.log(Spot/Strike) + (rate - div + Vol**2 / 2) * TTM) / (Vol * np.sqrt(TTM))
        d2 = (np.log(Spot/Strike) + (rate - div - Vol**2 / 2) * TTM) / (Vol * np.sqrt(TTM))
        
        if IsCall:  # Call option pricing formula
            return Spot * np.exp(-div * TTM) * N(d1) - Strike * np.exp(-rate * TTM) * N(d2)
        else:  # Put option pricing formula
            return -Spot * np.exp(-div * TTM) * N(-d1) + Strike * np.exp(-rate * TTM) * N(-d2)
    else:  # At or past maturity
        if IsCall:  # Intrinsic value for call
            return np.maximum(Spot - Strike, 0)
        else:  # Intrinsic value for put
            return np.maximum(-Spot + Strike, 0)

def BlackScholesDelta(Spot, Strike, TTM, rate, div, Vol, IsCall):
    """
    Calculate the Black-Scholes Delta, which measures the rate of change of the option
    price with respect to changes in the underlying asset's price.

    Args:
    - Spot (float): Current price of the underlying asset.
    - Strike (float): Strike price of the option.
    - TTM (float): Time to maturity of the option, in years.
    - rate (float): Risk-free interest rate, as a decimal.
    - div (float): Dividend yield of the asset, as a decimal.
    - Vol (float): Volatility of the asset's returns, as a decimal.
    - IsCall (bool): True if the option is a call option, False if it is a put option.

    Returns:
    - float: The Delta of the option.
    """
    
    if TTM > 0:  # Positive time to maturity
        # Calculation of d1 for Delta using Black-Scholes formula components
        d1 = (np.log(Spot/Strike) + (rate - div + Vol**2 / 2) * TTM) / (Vol * np.sqrt(TTM))
        
        if IsCall:  # Delta for call option
            return norm.cdf(d1) * np.exp(-div * TTM)
        else:  # Delta for put option
            return -norm.cdf(-d1) * np.exp(-div * TTM)
    else:  # At or past maturity
        if IsCall:  # Delta is 1 if in the money, else 0
            return 1 if Spot > Strike else 0
        else:  # Delta is -1 if in the money, else 0
            return -1 if Spot < Strike else 0
        

# Assuming the previously defined BlackScholes and BlackScholesDelta functions are already in your code

def BlackScholesGamma(Spot, Strike, TTM, rate, div, Vol):
    """
    Calculate the Black-Scholes Gamma, which measures the rate of change of the option's
    Delta with respect to changes in the underlying asset's price.

    Gamma provides insight into the convexity of the option's value in relation to the price
    of the underlying asset. A higher Gamma indicates greater sensitivity of the Delta to
    movements in the underlying asset's price.

    Args:
    - Spot (float): Current price of the underlying asset.
    - Strike (float): Strike price of the option.
    - TTM (float): Time to maturity of the option, in years.
    - rate (float): Risk-free interest rate, as a decimal.
    - div (float): Dividend yield of the asset, as a decimal.
    - Vol (float): Volatility of the asset's returns, as a decimal.

    Returns:
    - float: The Gamma of the option.
    """
    if TTM > 0:  # Positive time to maturity
        d1 = (np.log(Spot/Strike) + (rate - div + Vol**2 / 2) * TTM) / (Vol * np.sqrt(TTM))
        N_prime = norm.pdf(d1)  # Value of the standard normal probability density function at d1
        gamma = N_prime * np.exp(-div * TTM) / (Spot * Vol * np.sqrt(TTM))
        return gamma
    else:  # At or past maturity, Gamma is not defined but you might choose to return 0 or handle it differently
        return 0  # Gamma approaches 0 as we reach expiration, assuming no drastic changes in underlying


