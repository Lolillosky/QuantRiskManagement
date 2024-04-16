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



