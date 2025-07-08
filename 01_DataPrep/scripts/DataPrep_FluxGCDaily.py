#!/usr/bin/env python
# coding: utf-8

#  Code to pre-process data for flux tower case - concatenate datasets, gap fill, etc
#  save dataframe as csv that is input into the GMM-PCA-IT framework

import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
from matplotlib.colors import ListedColormap


data_folder='DATA/FluxTowers/'
out_data_folder = 'DATA/Processed/'


df = pd.read_csv(data_folder+'Corrected_Daily_25m_2016_2023.csv')
df['Date']=pd.to_datetime(df['Date']).dt.tz_localize(None)
df['DOY'] = df['Date'].dt.dayofyear

df_raw = pd.read_csv(data_folder+'FluxData_Raw_ALL.csv',usecols=['NewDate','tau','u_star','rslt_wnd_spd','T_tmpr_rh_mean'])
df_raw['NewDate']=pd.to_datetime(df_raw['NewDate'])
df_raw['tau'] = np.where(df_raw['tau']>df_raw['tau'].quantile(.995),np.nan,df_raw['tau'])
df_raw['tau'] = np.where(df_raw['tau']<0,np.nan,df_raw['tau'])
df_raw['u_star'] = np.where(df_raw['u_star']>df_raw['u_star'].quantile(.95),np.nan,df_raw['u_star'])
df_raw['u_star'] = np.where(df_raw['u_star']<0,np.nan,df_raw['u_star'])
df_raw['rslt_wnd_spd'] = np.where(df_raw['rslt_wnd_spd']>df_raw['rslt_wnd_spd'].quantile(.995),np.nan,df_raw['rslt_wnd_spd'])
df_raw['rslt_wnd_spd'] = np.where(df_raw['rslt_wnd_spd']<0,np.nan,df_raw['rslt_wnd_spd'])



df_raw['UoverUstar'] = df_raw['rslt_wnd_spd']/df_raw['u_star']

df_raw_1D = df_raw.resample('1D',on='NewDate').mean()


df_raw_maxvals =df_raw.resample('1D',on='NewDate').max()
df_raw_minvals =df_raw.resample('1D',on='NewDate').min()

df_raw_1D['Ta_min']=df_raw_minvals['T_tmpr_rh_mean']
df_raw_1D['Ta_max']=df_raw_maxvals['T_tmpr_rh_mean']

df_raw = df_raw_1D

df_raw['Date']=df_raw.index
#df= df.dropna()


df_NDVI_MOD1 = pd.read_csv(data_folder+ 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MOD13A1-061-results.csv')
df_NDVI_MOD2 = pd.read_csv(data_folder+ 'MODIS_fluxtower_download/flux-tower-MODIS-NDVI-MYD13A1-061-results.csv')


df_NDVI_MOD1['Date']=pd.to_datetime(df_NDVI_MOD1['Date']).dt.tz_localize(None)
df_NDVI_MOD2['Date']=pd.to_datetime(df_NDVI_MOD1['Date']).dt.tz_localize(None)
df_NDVI_MOD1['NDVI']=df_NDVI_MOD1['MOD13A1_061__500m_16_days_NDVI']
df_NDVI_MOD2['NDVI']=df_NDVI_MOD2['MYD13A1_061__500m_16_days_NDVI']

df_m1 = df_NDVI_MOD1[['Date','NDVI']]
df_m2 = df_NDVI_MOD2[['Date','NDVI']]


dfMOD = pd.concat([df_m1, df_m2], axis=0).drop_duplicates('Date')


dfMOD = dfMOD.set_index('Date')

plt.plot(dfMOD['NDVI'],'.b')


dfMOD_1day = dfMOD.resample('1D').interpolate(method='linear')

plt.plot(dfMOD_1day['NDVI'],'r')

dfMOD_1day=dfMOD_1day.reset_index()


df = pd.merge(df,dfMOD_1day,on='Date',how='outer')

df = pd.merge(df,df_raw,on='Date',how='outer')


#set desired time range (if not whole dataframe)
start_date = dt.datetime(2016,4,15,0,0,0)
end_date = dt.datetime(2022,12,31,0,0,0)


df = df.loc[df['Date']>start_date]
df = df.loc[df['Date']<end_date]



df = df.set_index(df['Date'])
df = df.drop(labels='Date',axis=1)

df = df.interpolate(method='time')

#drop 2020, too much missing data during pandemic spring  
df = df[df.index.year !=2020]    

df = df.loc[df['DOY_x']>90]
df = df.loc[df['DOY_x']<310]


print(list(df.columns))


#%%

df['gdd'] = (df['Ta_max']+df['Ta_min'])/2 - 18
df['gdd']=np.where(df['gdd']<0,0,df['gdd'])

# Calculate the cumulative sum resetting at the beginning of each year
df['gdd'] = df.groupby(df.index.year)['gdd'].transform('cumsum')
df['Precip_cum'] = df.groupby(df.index.year)['Precip_Tot'].transform('cumsum')
df['ET_cum'] = df.groupby(df.index.year)['ET_corr'].transform('cumsum')

df['P_ET_cumdiff'] = df['Precip_cum']-df['ET_cum']

df['Precip_14D']=df['Precip_Tot'].rolling(14, min_periods=1).sum()
df['Precip_7D']=df['Precip_Tot'].rolling(7, min_periods=1).sum()
df['Precip_3D']=df['Precip_Tot'].rolling(3, min_periods=1).sum()
df['Precip_1D']=df['Precip_Tot'].rolling(1, min_periods=1).sum()

#NDVI_peak is tower values (peak of daytime values)
#NDVI_inc is increment from one day to next in tower values


#df['NDVI'] = df['NDVI']+df['NDVI_inc']

#vegetation fraction: account for NDVI saturating at high value 
#make low NDVI equal to nan (don't want to consider non-growing season)
df['NDVI']=np.where(df['NDVI']<0.3,np.nan,df['NDVI'])

df['Veg_frac']=(df['NDVI']-.3)/(0.8-.3)
df['Veg_frac'] = np.where(df['Veg_frac']>1,1,df['Veg_frac'])
df['Veg_frac'] = np.where(df['Veg_frac']<0.05,0.05,df['Veg_frac'])



#%%



# In[25]:


#ET partitioning into E and T - trying TSEB approach, computes transpiration from temp, radiation, veg fraction
def Transpiration(df,TempName,RadName,fgName,Seconds):
    #slope of vapor pressure curve
    T = df[TempName]
    Rn = df[RadName]
    fg=df[fgName]
    s = 4098 * (.6108 * np.exp(17.67*T/(237.3+T)))/((237.3+T)**2)
    y = .066 #psychrometric constant
    Tr = 1.3*fg*(s/(s+y))*Rn
    
    heatvap_lambda = 2.501-(2.361 * 10**-3)*T   #latent heat of vaporization

    Tr_mm = Tr/(heatvap_lambda*1000*1000/Seconds) #get in mm (to compare with ET)
    Tr_Wm2 = Tr
    
    return Tr_mm, Tr_Wm2

#maximum bound for transpiration (use max temperature from that day)
df['Tr_mm'], df['Tr_Wm2'] = Transpiration(df,'Ta_max','Rn_new','Veg_frac',3600*24)


df['Tr_mm']=np.where(df['Tr_mm']>=df['ET_corr'], df['ET_corr'],df['Tr_mm'])
df['Tr_Wm2']=np.where(df['Tr_Wm2']>=df['LE_corr'], df['LE_corr'],df['Tr_Wm2'])

plt.figure(5)
plt.plot(df['Tr_mm'],df['ET_corr'],'.')
plt.xlabel('Transpiration')
plt.ylabel('total ET')


# In[7]:

df['EF']=df['LE_corr']/(df['LE_corr']+df['H_corr'])
df['EF'] = np.where(df['EF']<0, 0, df['EF'])
df['EF'] = np.where(df['EF']>3, 3, df['EF'])
df['EF'] = np.where(np.isnan(df['ET_corr']),np.nan,df['EF'])

df['ToverET']=df['Tr_Wm2']/(df['LE_corr']+df['H_corr'])
df['ToverET']=np.where(df['ToverET']>1,1,df['ToverET'])

df['WUE']=df['GPP_DT']/df['Tr_Wm2']
df['WUE'] = np.where(df['WUE']>df['WUE'].quantile(.995), np.nan, df['WUE'])
df['WUE'] = np.where(df['WUE']<df['WUE'].quantile(.005), np.nan, df['WUE'])

df['GPPoverNDVI']=df['GPP_DT']/df['NDVI']
df['GPPoverNDVI'] = np.where(df['GPPoverNDVI']>df['GPPoverNDVI'].quantile(.995), np.nan, df['GPPoverNDVI'])
df['GPPoverNDVI'] = np.where(df['GPPoverNDVI']<df['GPPoverNDVI'].quantile(.005), np.nan, df['GPPoverNDVI'])

df['Reco_Fraction']=df['Reco_DT']/(df['Reco_DT']+np.abs(df['GPP_DT']))

#colnames_responses = ['NEE','GPP_U50_f','Reco_U50','ToverET','WUE','NDVI_update']
colnames_responses = ['NEE','Reco_Fraction','ToverET','WUE','NDVI','GPPoverNDVI','EF','GPP_DT','Reco_DT','H_corr','ET_corr','GPP_DT','Reco_DT','Tr_Wm2','tau','UoverUstar','DOY']
colnames_drivers = ['Precip_14D','Precip_7D','Precip_3D','Precip_1D','gdd','T_tmpr_rh_mean_x','D5TE_VWC_100cm_Avg','D5TE_VWC_5cm_Avg','P_ET_cumdiff','Rn_new']
nfeatures=len(colnames_responses)
ntars = len(colnames_drivers)

dfnew = df[colnames_responses].copy()
dfnew[colnames_drivers]=df[colnames_drivers]


#rolling average - 14 days (to focus on seasonal/subseasonal trends)
dfnew = dfnew.rolling(14,min_periods=1,center=True).mean()


#drop winter months
dfnew = dfnew.loc[dfnew.index.dayofyear>120]
dfnew = dfnew.loc[dfnew.index.dayofyear<300]

dfnew = dfnew.dropna() #This leads to some gaps since I'm omitting any row where any variable is a nan

dfnew.to_csv(out_data_folder+'ProcessedData_GCFluxTowerDaily.csv')

