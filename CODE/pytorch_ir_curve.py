import torch
import pytorch_spline
import numpy as np
from scipy.optimize import fsolve

class IR_Curve:

    def __init__(self, time_pillars, rates):

        time_pillars = torch.concatenate([torch.tensor([0]), time_pillars])
        rates = torch.concatenate([torch.tensor([0]), rates])
        time_times_rate = rates * time_pillars

        self.interpolator = pytorch_spline.NaturalCubicSpline_Torch(time_pillars,time_times_rate)

    def zero_coupon_rates(self, time_pillars):

        return self.interpolator.evaluate_spline(time_pillars) / time_pillars
    
    def discount_factors(self, time_pillars):

        return torch.exp(-self.interpolator.evaluate_spline(time_pillars))
    

class IR_Swap:

    def __init__(self, star_t, end_t, delta_t):

        self.star_t = star_t
        self.end_t = end_t
        self.delta_t = delta_t
        
        self.pay_times = torch.tensor(np.concatenate(([star_t], np.arange(end_t, star_t, -delta_t)[::-1])))
        self.dcf = self.pay_times[1:] - self.pay_times[:-1]
        self.pay_times = self.pay_times[1:]

    def calc_PV01(self, ir_curve):

        discount_factors = ir_curve.discount_factors(self.pay_times)
        return torch.sum(self.dcf * discount_factors)
    
    def calc_par_rate(self, ir_curve):

        PV01 = self.calc_PV01(ir_curve)

        return (1 - ir_curve.discount_factors(self.end_t)) / PV01
    
class CurveFitter:
    def __init__(self, time_pillars, swap_rates, delta_t):
        self.time_pillars = time_pillars
        self.swap_rates = swap_rates
        self.delta_t = delta_t
        
    def residuals(self, x):

        curve = IR_Curve(torch.tensor(self.time_pillars), torch.tensor(x))
        swap = [IR_Swap(0, t, self.delta_t) for t in self.time_pillars]

        return [s.calc_par_rate(curve).item() for s in swap] - self.swap_rates
    
    def fit(self):
        
        zc_rates = fsolve(self.residuals, self.swap_rates)

        return zc_rates