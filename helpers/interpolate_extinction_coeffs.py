import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

df = pd.read_table( \
    './constants/extinction_coefficients.wsv', \
    delim_whitespace=True, \
    names=('Wavelength(nm)', 'Water(OD/M/cm)', 'HbO2 (OD/M/cm)', 'HHb (OD/M/cm)',
        'Difference Cytochrome Oxidase(OD/M/cm)', 'ox-redCCO(OD/M/cm)_Moody',
        'oxCCO(OD/M/cm)_Moody', 'redCCO(OD/M/cm)_Moody', 'Fat Soybean(OD/cm)_Ekker'), \
    index_col=0 # the Wavelength(nm) column,
)

lower_thresh, upper_thresh = 700, 900
wls = np.load('./data/wavelengths.npy')
wls = wls[(wls > lower_thresh) & (wls < upper_thresh)]

def interp(name):
    fun = interp1d(df.index.values, df[name])
    return fun(wls)

def extinction_coeffs():
    names = ('Water(OD/M/cm)', 'HbO2 (OD/M/cm)', 'HHb (OD/M/cm)')
    extintion_coeffs = np.array([interp(name) for name in names])

    return wls, extintion_coeffs