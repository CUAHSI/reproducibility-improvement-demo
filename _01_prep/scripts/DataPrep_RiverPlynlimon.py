#!/usr/bin/env python
# coding: utf-8

#  Code to pre-process data for riverlab case - concatenate datasets, gap fill, etc
#  save dataframe as csv that is input into the GMM-PCA-IT framework
#  Plynlimon, UK version!!!

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap

data_folder='data_input/River/'
out_data_folder = 'data_intermediate/'
res = '7H'



#%%

#df_rivervars = pd.read_csv(data_folder + 'PlynlimonData/data/PlynlimonHighFrequencyHydrochemistry.csv')

df_rivervars = pd.read_csv(data_folder + 'PlylimonEditedData_KirchnerPNAS.csv')

df_rivervars = df_rivervars[df_rivervars["Site"] == 'UHF']

orig_df_rivervars = df_rivervars.copy()

#%%
df_rivervars['date_time']=pd.to_datetime(df_rivervars['date_time']).dt.tz_localize(None)

df_rivervars = df_rivervars.reset_index()

colnames_keep = ['dayno','date_time', 'Flow cumecs', 'NO3-N mg/l', 'SO4 mg/l', 'Cl mg/l', 'Na mg/l', 'Mg mg/l', 'K mg/l', 'Ca mg/l']

for c in df_rivervars.columns:
    if c in colnames_keep:
        if c != 'date_time':
            df_rivervars[c]=pd.to_numeric(df_rivervars[c])
         
df = df_rivervars[colnames_keep]

df = df.resample(res,on='date_time').mean()
#df_rivervars['Date']=df_rivervars.index

df = df.reset_index()


#%%


df_withnans = df.copy()



#want to linearly interpolate gaps in each variable, up to 1 day (about 4 data points)
for c in df.columns:
    df[c] = df[c].interpolate(method='linear',limit=5)
    
df_fillednans = df.copy()

df['DOY']=df['date_time'].dt.dayofyear
#df['Discharge']=np.log(df['Discharge'])

df=df.set_index(df['date_time'])
df = df.drop(labels='date_time',axis=1)

colnames_keep = ['dayno','date_time', 'Flow cumecs', 'NO3-N mg/l', 'SO4 mg/l', 'Cl mg/l', 'Na mg/l', 'Mg mg/l', 'K mg/l', 'Ca mg/l']


colnames_responses = ['Ca mg/l','Mg mg/l','K mg/l','NO3-N mg/l','Cl mg/l','Na mg/l','SO4 mg/l']

for c in colnames_responses:
    df[c]=np.where(df[c]<0,0,df[c])


#convert Q to total liters of water in each timestep
#df['Q_liters'] = df['Discharge']*3.6*10**6


df['LogQ']=np.log10(df['Flow cumecs'])

df['LogQ'] = np.where(df['LogQ']<-10,np.nan,df['LogQ'])

#colnames_drivers = ['Discharge','Precip_1D','Precip_3D','Precip_7D','Precip_14D','D5TE_VWC_100cm_Avg','Temp_anomaly_14D','O2_anomaly_14D','Turbidity','GWE','Dissolved Oxygen','Temperature']

colnames_drivers = ['Flow cumecs','LogQ','dayno']

dfnew = df[colnames_responses].copy()
dfnew[colnames_drivers]=df[colnames_drivers]


#omit few points where concentrations far above averages...4 std above -set to nan

for c in colnames_responses:
    max_c = dfnew[c].mean()+4*dfnew[c].std()
    dfnew[c]=np.where(dfnew[c]>max_c,np.nan,dfnew[c])


#for c in dfnew.columns:
#    dfnew[c]=pd.to_numeric(dfnew[c])

dfnew = dfnew.dropna() #This leads to some gaps since I'm omitting any row where any variable is a nan

dfnew['Date']=dfnew.index

dfnew.to_csv(out_data_folder+'ProcessedData_RiverPlynlimon.csv')

