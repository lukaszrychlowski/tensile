import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

path = '/Users/ryszard/Downloads/13.01_seg_10-11_ekstensometr/'

def get_sample_params(path):
    df = pd.read_csv(path+file, nrows=5, delimiter=';',decimal=',', names=['time','extension','load', 'strain1'])
    df = df.drop(columns=['time', 'load', 'strain1'])
    df = df.apply(lambda x: x.str.replace(',','.'))
    sample_params = {'label': df.iloc[1][0], 'length': df.iloc[2][0], 'width': df.iloc[3][0], 'thickness': df.iloc[4][0]}
    for k, v in sample_params.items():
        try:
            sample_params[k] = float(v)
        except ValueError:
            continue
    sample_params['area'] = sample_params['width'] * sample_params['thickness']
    return sample_params
    
def importer(path):
    df = pd.read_csv(path+file, dtype=np.float32, skiprows=10, delimiter=';',decimal=',', names=['time','extension','load', 'strain1'])
    df = df.drop(columns=['time'])
    min_strain1 = df.head()['strain1'].min()
    min_ext = df.head()['extension'].min()
    df['strain1'] = df['strain1'] - min_strain1
    #df = df['strain1'].drop(df[df['strain1'] > 0])
    df['extension'] = df['extension'] - min_ext
    return df

def calc_stress(load, area):
    stress = load/area
    return stress

def calc_strain(extension, length):
    strain = extension/length
    return strain

def calc_yield(stress_offset, stress):
    return np.argwhere(np.diff(np.sign(stress_offset - stress))).flatten()

def calc_slope(df, strain_min_limit, strain_max_limit):
    df['linear_range'] = (df['strain1'] >= strain_min_limit) & (df['strain1'] <= strain_max_limit)                  # Separate linear region based on strain_min and strain_max limits
    filtered_df = df[df['linear_range'] == True]
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_df['strain1'], filtered_df['stress'])   # Calculate slope of the linear region
    df['stress_linear_offset'] = slope * (df['strain1'] - 0.002) + intercept                                        # Extend stress vals and offset them
    return df


fig, axs = plt.subplots(2)

for file in os.listdir(path):
    if file.endswith('.csv'):
        df = importer(path)
        sample_params = get_sample_params(path)
        df['stress'] = calc_stress(df['load'], sample_params['area'])
        df['strain'] = calc_strain(df['extension'], sample_params['length'])
        df = calc_slope(df, strain_min_limit=0.001, strain_max_limit=0.002)
        idx = calc_yield(df['stress_linear_offset'],df['stress'])
        print(sample_params['label'])
        print(df['stress'][idx].min())
        axs[0].scatter(df['strain1'], df["stress"], s=0.5)
        axs[0].plot(df['strain1'], df["stress_linear_offset"], linewidth=0.2)
        axs[0].scatter(df['strain1'][idx], df['stress'][idx], s=15, marker='x', c='red')
        axs[1].scatter(df['strain'], df['stress'], s=0.2)
        axs[0].set_xlim(0, 0.06)
        axs[0].set_ylim(0, 500)
        axs[1].set_xlim(0, 0.2)
        axs[1].set_ylim(0, 500)
        
plt.show()

