"""{
    { LayerParameterValues{1.33, 0.9, 0.0015,   1,      5},     LayerParameterValues{1.5, 0.9, 0.28,    100,    15} },
    { LayerParameterValues{1.33, 0.9, 0,        0.2,    1},     LayerParameterValues{1.5, 0.9, 0,       0.2,    5} }, // csf broadly fixed layer
    { LayerParameterValues{1.33, 0.9, 0.0015, 1},     LayerParameterValues{1.5, 0.9, 1, 100} } // semi-infinite
};"""

param_ranges = [
    {
        "n": {
            "lower": 1.33,
            "upper": 1.50
        },
        "g": {
            "lower": 0.9,
            "upper": 0.9
        },
        "muA": {
            "lower": 0.0015,
            "upper": 0.28
        },
        "muS": {
            "lower": 1.0,
            "upper": 100
        },
        "height": {
            "lower": 5,
            "upper": 15
        }
    },
    {
        "n": {
            "lower": 1.33,
            "upper": 1.50
        },
        "g": {
            "lower": 0.9,
            "upper": 0.9
        },
        "muA": {
            "lower": 0.0,
            "upper": 0.0
        },
        "muS": {
            "lower": 0.2,
            "upper": 0.2
        },
        "height": {
            "lower": 1.0,
            "upper": 5.0
        }
    },
    {
        "n": {
            "lower": 1.33,
            "upper": 1.50
        },
        "g": {
            "lower": 0.9,
            "upper": 0.9
        },
        "muA": {
            "lower": 0.0015,
            "upper": 1
        },
        "muS": {
            "lower": 1.0,
            "upper": 100
        }
    }
]

import numpy as np

csf_free_vars = ['height', 'n']
free_variabes = csf_free_vars + ['muA', 'muS']

free_vars = [free_variabes, csf_free_vars, free_variabes[1:]] # 3 layers with csf in the middle, no height for last layer
free_vars_wout_n = []
for vars in free_vars:
    new_vars = []
    for var in vars:
        if var != 'n':
            new_vars.append(var)
    free_vars_wout_n.append(new_vars)


class Sampler():
    def __init__(self, keys=free_vars):
        self.gen = np.random.default_rng(seed=123)
        self.keys = keys

        # precomputation
        self._deltas = self._call_on_keys(lambda range: range['upper'] - range['lower'])[None, :]
        self._lowers = self._call_on_keys(lambda range: range['lower'])[None, :]
        
    def __call__(self):
        random_between = lambda range: self.gen.uniform() * (range['upper'] - range['lower']) + range['lower']
        sample = self._call_on_keys(random_between)
        
        return sample[None, :]

    def _call_on_keys(self, fun):
        return np.array([fun(param_ranges[i][var]) for i, vars in enumerate(self.keys) for var in vars])

    def trans_sample(self, sample, lower=0, upper=1):
        return (sample - self._lowers) / self._deltas
    
    def inv_trans_sample(self, sample_prime, lower=0, upper=1):
        return (sample_prime * self._deltas) + self._lowers