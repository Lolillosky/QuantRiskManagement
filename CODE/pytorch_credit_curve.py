
import torch
import numpy as np
from scipy.optimize import fsolve

class Credit_Curve():

    def __init__(self, maturities, lambdas):
        self.maturities = torch.concatenate((torch.tensor([0.0]),maturities))
        self.lambdas = lambdas 

        self.surv_probs = torch.zeros_like(self.maturities)
        self.surv_probs[0] = 1.0

        self.surv_probs[1:] = torch.exp(-self.lambdas*(self.maturities[1:] - self.maturities[:-1]))

        self.surv_probs = torch.cumprod(self.surv_probs, dim =0)

    def calc_survival_prob(self, t):

        index = torch.searchsorted(self.maturities, t, right=True)

        index = torch.clamp(index, min=0, max=len(self.lambdas))
        return self.surv_probs[index-1] * torch.exp(-self.lambdas[index-1] * (t - self.maturities[index-1])) 
    

    
class CDS:

    def __init__(self, star_t, end_t, pay_delta_t, default_delta_t, recovery_rate):

        self.star_t = star_t
        self.end_t = end_t
        self.pay_delta_t = pay_delta_t
        self.default_delta_t = default_delta_t
        self.recovery_rate = recovery_rate
        
        
        self.pay_times = torch.tensor(np.concatenate(([star_t], np.arange(end_t, star_t, -pay_delta_t)[::-1])))
        self.default_times = torch.tensor(np.concatenate(([star_t], np.arange(end_t, star_t, -default_delta_t)[::-1])))
        self.dcf = self.pay_times[1:] - self.pay_times[:-1]
        self.pay_times = self.pay_times[1:]


    def calc_DV01_DL(self, ir_curve, credit_curve):

        discount_factors_pay_times = ir_curve.discount_factors(self.pay_times)
        discount_factors_default_times  = ir_curve.discount_factors(self.default_times[1:])
        surv_probs_pay_times = credit_curve.calc_survival_prob(self.pay_times)
        surv_probs_default_times = credit_curve.calc_survival_prob(self.default_times) 
        
        dflt_probs = surv_probs_default_times[:-1] - surv_probs_default_times[1:]

        dv01 = torch.sum(self.dcf * discount_factors_pay_times * surv_probs_pay_times)
        dl = (1.0 - self.recovery_rate)*torch.sum(discount_factors_default_times * dflt_probs)

        return dv01, dl 
    
    def calc_CDS_rate(self, ir_curve, credit_curve):

        dv01, dl = self.calc_DV01_DL(ir_curve, credit_curve)

        return dl / dv01
    
    def calc_receiver_CDS_NPV(self, ir_curve, credit_curve, CDS_rate):

        dv01, dl = self.calc_DV01_DL(ir_curve, credit_curve)

        return dv01 * CDS_rate - dl
  

class CreditCurveFitter:
    def __init__(self, time_pillars, CDS_rates, pay_delta_t, default_delta_t, recovery_rate, ir_curve):
        
        self.time_pillars = time_pillars
        self.CDS_rates = CDS_rates
        self.pay_delta_t = pay_delta_t
        self.default_delta_t = default_delta_t
        self.recovery_rate = recovery_rate
        self.ir_curve = ir_curve
        self.num_calibrations = 0
        self.default_intens = np.zeros_like(self.CDS_rates)

    def residuals(self, x):

        if self.num_calibrations == 0:
            credit_curve = Credit_Curve(torch.tensor(self.time_pillars[0:self.num_calibrations+1]), 
                                     torch.tensor(x))
        else:
            credit_curve = Credit_Curve(torch.tensor(self.time_pillars[0:self.num_calibrations+1]), 
                                     torch.tensor(np.concatenate((self.default_intens[0:self.num_calibrations], x))))
        
        
        cds = CDS(0, self.time_pillars[self.num_calibrations], self.pay_delta_t, self.default_delta_t, self.recovery_rate)

        model_cds = cds.calc_receiver_CDS_NPV(self.ir_curve, credit_curve, self.CDS_rates[[self.num_calibrations]]).item()
        
        return model_cds 
    
    def fit(self):


        for i in range(len(self.default_intens)):

            self.default_intens[i] = fsolve(self.residuals, 0.0)

            self.num_calibrations += 1
        

        return self.default_intens.copy()