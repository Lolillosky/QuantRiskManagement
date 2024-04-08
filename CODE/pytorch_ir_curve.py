import torch
import pytorch_spline

class ir_curve:

    def __init__(self, time_pillars, rates):

        time_pillars = torch.concatenate([torch.tensor([0]), time_pillars])
        rates = torch.concatenate([torch.tensor([0]), rates])
        time_times_rate = rates * time_pillars

        self.interpolator = pytorch_spline.NaturalCubicSpline_Torch(time_pillars,time_times_rate)

    def zero_coupon_rates(self, time_pillars):

        return self.interpolator.evaluate_spline(time_pillars) / time_pillars
    
    def discount_factors(self, time_pillars):

        return torch.exp(-self.interpolator.evaluate_spline(time_pillars))
