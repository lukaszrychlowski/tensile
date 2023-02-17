import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import mplcursors
path = '/Users/ryszard/Downloads/Specimen_RawData_7/'

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

def calc_total_elongation(df):
    return df['strain'].iloc[-1]

def calc_slope(df, strain_min_limit, strain_max_limit):
    df['linear_range'] = (df['strain1'] >= strain_min_limit) & (df['strain1'] <= strain_max_limit)                  # Separate linear region based on strain_min and strain_max limits
    filtered_df = df[df['linear_range'] == True]
    slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_df['strain1'], filtered_df['stress'])   # Calculate slope of the linear region
    df['stress_linear_offset'] = slope * (df['strain1'] - 0.002) + intercept                                        # Extend stress vals and offset them
    return df

def calc_yield(stress_offset, stress):
    return np.argwhere(np.diff(np.sign(stress_offset - stress))).flatten()

def calc_true_strain(strain):
    true_strain = np.log(1 + strain)
    return true_strain

def calc_true_stress(stress, strain):
    true_stress = stress * (1 + strain)
    return true_stress


fig, axs = plt.subplots(2)

for file in os.listdir(path):
    if file.endswith('.csv'):
        df = importer(path)
        sample_params = get_sample_params(path)
        if '13.01_seg9' in sample_params['label']:
            df['stress'] = calc_stress(df['load'], sample_params['area'])
            df['strain'] = calc_strain(df['extension'], sample_params['length'])
            df['true stress'] = calc_true_stress(df['stress'], df['strain'])
            df['true strain'] = calc_true_strain(df['strain'])
            df = calc_slope(df, strain_min_limit=0.001, strain_max_limit=0.002)
            idx = calc_yield(df['stress_linear_offset'],df['stress'])
            df['label'] = sample_params['label']
            df['path'] = path+file
            print(sample_params['label'] + ';' + str(np.round(df['stress'][idx].min(), 2)) + ';' + str(np.round(df['strain'].iloc[-1]*100, 2)))     #label;yield;total_elongation
            axs[0].scatter(df['strain1'], df["stress"], s=1, label=sample_params['label']+ ' ' + file)
            axs[0].plot(df['strain1'], df["stress_linear_offset"], linewidth=0.5)
            axs[0].scatter(df['strain1'][idx], df['stress'][idx], s=15, marker='x', c='red')
            axs[1].scatter(df['strain'], df['stress'], s=1,label=sample_params['label']+ ' ' + file)
            # axs[0].set_xlim(0, 0.06)
            # axs[0].set_ylim(0, 550)
            # axs[1].set_xlim(0, 0.2)
            # axs[1].set_ylim(0, 550)
           
mplcursors.cursor(highlight=True).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
axs[0].grid(which='both', color='gray', linewidth=0.5, alpha=0.5)
axs[1].grid(which='both', color='gray', linewidth=0.5, alpha=0.5)
axs[0].minorticks_on()
axs[1].minorticks_on()
#plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.legend(fontsize='small')
plt.savefig(path+file+'_.png', dpi=300)
plt.show()

