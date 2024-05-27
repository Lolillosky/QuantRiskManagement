import numpy as np

class IRS:

  def __init__(self, maturity, num_fix_pay_year, fix_rate):

    self.fix_rate = fix_rate
    self.fix_pay_dates = np.arange(maturity, 0, -1/num_fix_pay_year)[::-1]

    # Vamos a hacer MAL el siguiente pago de flotante, pues haré sustitución
    # por principales como si el primer pago de flotante se fijase en la
    # fecha en la que calculo el NPV

    self.year_frac_fix = np.zeros(len(self.fix_pay_dates))
    self.year_frac_fix[0] = self.fix_pay_dates[0]
    self.year_frac_fix[1:] = 1/num_fix_pay_year

  def get_NPV(self, t, IRCurve):

    if t < self.fix_pay_dates[-1]:
      float_leg_npv = 1-IRCurve.DiscountFactor(t, self.fix_pay_dates[-1])
    else:
      float_leg_npv = 0

    pending_fix_payments = self.fix_pay_dates[self.fix_pay_dates > t]
    pending_year_frac = self.year_frac_fix[self.fix_pay_dates > t]

    fix_leg_npv = 0

    for i in range(len(pending_fix_payments)):

      disc = IRCurve.DiscountFactor(t, pending_fix_payments[i])

      fix_leg_npv += disc*pending_year_frac[i]*self.fix_rate

    return fix_leg_npv - float_leg_npv
  

class HWModel:

  def __init__(self, init_curve, kappa, sigma):

    self.init_curve = init_curve
    self.kappa = kappa
    self.sigma = sigma

  def DiscountFactor(self, t, T, x_t):

    B_t_T = (1-np.exp(-self.kappa*(T-t))) / self.kappa

    P_0_T = self.init_curve.DiscountFactor(0,T)
    P_0_t = self.init_curve.DiscountFactor(0,t)

    f_0_t = self.init_curve.InstantForwardRate(t)

    coef = self.sigma*self.sigma*(1-np.exp(-2*self.kappa*t))/(4*self.kappa)

    A_t_T = np.exp(B_t_T*f_0_t - coef*B_t_T*B_t_T) * P_0_T / P_0_t

    alpha_t = f_0_t + self.sigma*self.sigma*(1-np.exp(-self.kappa*t))**2/(2*self.kappa*self.kappa)

    r_t = x_t +  alpha_t

    return A_t_T * np.exp(-B_t_T*r_t)

  def SimulProcess(self, t, T, x_t):

    exp_val = x_t * np.exp(-self.kappa*(T-t))

    vol = self.sigma * np.sqrt((1-np.exp(-2*self.kappa*(T-t)))/(2*self.kappa))

    return exp_val + vol*np.random.normal(0,1)

  def get_rate(self, t, x_t):

    f_0_t = self.init_curve.InstantForwardRate(t)

    alpha_t = f_0_t + self.sigma*self.sigma*(1-np.exp(-self.kappa*t))**2/(2*self.kappa*self.kappa)

    return x_t +  alpha_t
  
class HWCurveWrapper:

    def __init__(self, HW):

        self.HW = HW
        self.x = 0

    def DiscountFactor(self, t,T):

        return self.HW.DiscountFactor(t,T,self.x)


class IRS_Portfolio:

  def __init__(self, swap_list):

    self.swap_list = swap_list

  def get_NPV(self, t, IRCurve):

    npv = 0

    for irs in self.swap_list:
      # Elem 0 es nominal y elem 1 es objeto irs
      npv += irs[0]*irs[1].get_NPV(t, IRCurve)

    return npv
  
class SurvCurve:

  def __init__(self, intensity):

    self.intensity = intensity

  def SurvProb(self, t,T):

    return np.exp(-self.intensity*(T-t))
  


class FlatIRCurve:

  def __init__(self, rate):

    self.rate = rate

  def InstantForwardRate(self, t):

    return self.rate

  def DiscountFactor(self, t,T):

    return np.exp(-self.rate*(T-t))


