import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

df_main = pd.read_table( \
    './constants/extinction_coefficients.wsv', \
    delim_whitespace=True, \
    names=('Wavelength(nm)', 'Water(OD/M/cm)', 'HbO2 (OD/M/cm)', 'HHb (OD/M/cm)',
        'Difference Cytochrome Oxidase(OD/M/cm)', 'ox-redCCO(OD/M/cm)_Moody',
        'oxCCO(OD/M/cm)_Moody', 'redCCO(OD/M/cm)_Moody', 'Fat Soybean(OD/cm)_Ekker'), \
    index_col=0 # the Wavelength(nm) column,
)

df_lipid = pd.read_table( \
    './constants/lipid_extinction_coefficient.wsv', \
    delim_whitespace=True, \
    names=('Wavelength(nm)', 'Lipid(OD/m)'), \
    index_col=0, # the Wavelength(nm) column
    skiprows=6
)

lower_thresh, upper_thresh = 740, 900
wls = np.load('./constants/wavelengths.npy')
wls = wls[(wls > lower_thresh) & (wls < upper_thresh)]

def interp(name, df, wavelengths=wls):
    fun = interp1d(df.index.values, df[name])
    return fun(wavelengths)

def extinction_coeffs(wavelengths=wls):
    names = ('Water(OD/M/cm)', 'HbO2 (OD/M/cm)', 'HHb (OD/M/cm)')
    extintion_coeffs = np.array(
        [interp(name, df_main, wavelengths=wavelengths) for name in names] +
        [interp('Lipid(OD/m)', df_lipid*1e-2, wavelengths=wavelengths)]
    )

    return wavelengths, extintion_coeffs