import numpy as np
import torch

class Portfolio_Delta_NPV_Calculator:

    def __init__(self, notionals ,contracts, option = 'numpy'):
        self.contracts = contracts
        self.notionals = notionals
        self.option = option

    def value(self,t ,risk_factors):

        if self.option == 'numpy':
            portfolio_value = np.zeros(risk_factors.shape[0])
            component_values = np.zeros((risk_factors.shape[0],len(self.contracts)))
        elif self.option == 'torch':
            portfolio_value = torch.zeros(risk_factors.shape[0])
            component_values = torch.zeros((risk_factors.shape[0],len(self.contracts)))

        for i, (n, c) in enumerate(zip(self.notionals,self.contracts)):
            
            component_values[:,i] = c(t,risk_factors)

        if self.option == 'numpy':
            portfolio_value = np.sum(self.notionals*component_values, axis = 1)
        elif self.option == 'torch':
            portfolio_value = torch.sum(self.notionals*component_values, dim = 1)

        return portfolio_value, component_values
    
    def compute_scenarios_pl(self, risk_factors_base_scenario, delta_t, risk_factors_shocked):
        
        portfolio_value_base_scenario, component_values_base_scenario = self.value(0,risk_factors_base_scenario)

        portfolio_value_shocked, component_values_shocked  = self.value(delta_t,risk_factors_shocked)

        return portfolio_value_shocked - portfolio_value_base_scenario, component_values_shocked - component_values_base_scenario

