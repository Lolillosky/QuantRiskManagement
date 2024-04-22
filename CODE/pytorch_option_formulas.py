import torch
from torch.distributions import Normal

def bachelier_option_formula(forward, strike, vol, ttm, iscall):

    normal_dist = Normal(0.0, 1.0)

    if ttm > 0:

        d = (forward - strike)/(vol*torch.sqrt(ttm))

        if iscall:

            return (forward - strike) * normal_dist.cdf(d) + vol * torch.sqrt(ttm) * torch.exp(normal_dist.log_prob(d))
        
        else:

            return (strike - forward) * normal_dist.cdf(-d) + vol * torch.sqrt(ttm) * torch.exp(normal_dist.log_prob(d))

    elif (ttm == 0):

        if iscall:  # Intrinsic value for call
            return torch.maximum(forward - strike, 0.0)
        else:  # Intrinsic value for put
            return torch.maximum(strike - forward, 0.0)


def BlackScholes(Spot, Strike, TTM, rate, div, Vol, IsCall):
    """
    Calculate the Black-Scholes option pricing model using PyTorch.

    Args:
    - Spot (torch.Tensor): Current price of the underlying asset.
    - Strike (torch.Tensor): Strike price of the option.
    - TTM (torch.Tensor): Time to maturity of the option, in years.
    - rate (torch.Tensor): Risk-free interest rate, as a decimal.
    - div (torch.Tensor): Dividend yield of the asset, as a decimal.
    - Vol (torch.Tensor): Volatility of the asset's returns, as a decimal.
    - IsCall (bool): True if the option is a call option, False if it is a put option.

    Returns:
    - torch.Tensor: The Black-Scholes price of the option.
    """
    normal_dist = Normal(0.0, 1.0)
    N = normal_dist.cdf  # Standard normal cumulative distribution function

    if TTM > 0:  # Positive time to maturity
        # Calculation of d1 and d2 using Black-Scholes formula components
        d1 = (torch.log(Spot/Strike) + (rate - div + Vol**2 / 2) * TTM) / (Vol * torch.sqrt(TTM))
        d2 = (torch.log(Spot/Strike) + (rate - div - Vol**2 / 2) * TTM) / (Vol * torch.sqrt(TTM))
        
        if IsCall:  # Call option pricing formula
            return Spot * torch.exp(-div * TTM) * N(d1) - Strike * torch.exp(-rate * TTM) * N(d2)
        else:  # Put option pricing formula
            return -Spot * torch.exp(-div * TTM) * N(-d1) + Strike * torch.exp(-rate * TTM) * N(-d2)
    else:  # At or past maturity
        if IsCall:  # Intrinsic value for call
            return torch.maximum(Spot - Strike, torch.tensor(0.0))
        else:  # Intrinsic value for put
            return torch.maximum(-Spot + Strike, torch.tensor(0.0))




