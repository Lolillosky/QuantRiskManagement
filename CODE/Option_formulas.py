import numpy as np
from scipy.stats import norm

def BlackScholes(Spot, Strike, TTM, rate,div, Vol, IsCall):
  
  N = norm.cdf

  if TTM >0:
    
    d1 = (np.log(Spot/Strike) + (rate -div + Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))
    d2 = (np.log(Spot/Strike) + (rate -div - Vol*Vol/2)*TTM)/(Vol*np.sqrt(TTM))
    
    if IsCall:
      
      return Spot*np.exp(-div*TTM)*N(d1)-Strike*np.exp(-rate*TTM)*N(d2)
    
    else:
      
      return -Spot*np.exp(-div*TTM)*N(-d1)+Strike*np.exp(-rate*TTM)*N(-d2)
    
  else:
  
    if IsCall:

      return np.maximum(Spot-Strike,0)

    else:

      return np.maximum(-Spot+Strike,0)

