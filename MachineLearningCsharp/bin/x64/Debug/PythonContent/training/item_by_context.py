#==============================================================================
#                   	CONTEXT PREDICTION - WEATHER&TIME INFORMATIONS
#==============================================================================
#%%
from math_utils import *
import pandas as pd
import numpy as np
import datetime as datetime
from datetime import timedelta
from pandas.io.json import json_normalize
from sklearn.externals import joblib
from urllib.request import urlopen
import json

# Connecting to the weather data sources
wd = urlopen('http://api.wunderground.com/api/7a7531c9c79ca2d3/hourly/q/FR/Bordeaux.json')# don't excess the autorized number of loading
# using this code to scrap the actual weather informations (for forecast , change history by forecast , check if need to put a date)
#%%

#%%
# do some data preparations
json_string1 =wd.read()
encoding1= wd.info().get_content_charset('utf-8')
wdf=json.loads(json_string1.decode(encoding1))
json_normalize(wdf['hourly_forecast']).columns# from pandas.io.json import json_normalize
#%%

#%%
# Feature Engineering on this weather data
DF=pd.DataFrame(json_normalize(wdf['hourly_forecast']))[['FCTTIME.hour', 'FCTTIME.min','FCTTIME.mday', 'FCTTIME.mon',
		'FCTTIME.pretty', 'FCTTIME.year','temp.metric','humidity','wdir.dir','condition']] # convert this data to a pandas data frame
# Note:the data will be recorded every 1h and until 23.00 the next day
DF.columns=['date.hour', 'date.min','date.mday', 'date.mon',
		'date.pretty', 'date.year','tempm','hum','wdire','conds'] # give a name to every column of this data set
DF['date.pretty']=pd.to_datetime(DF['date.pretty'])
DF['DoW']=[DF['date.pretty'][i].weekday()+2 for i in range(len(DF['date.pretty']))]# Sunday will be represented by the number 8
DF['DoW']=DF['DoW'].apply(replacesunday)# apply this function to change from 8 to 1 for Sunday
DF['WoY']=[DF['date.pretty'][i].strftime("%U") for i in range(len(DF['date.pretty']))]
DF['QoY']=[DF['date.pretty'][i].to_period('Q') for i in range(len(DF['date.pretty']))]
DF['QoY'] = DF['QoY'].dt.quarter# to get the number of the quarter only
DF['HMS']=[DF['date.pretty'][i].strftime("%H:%M:%S") for i in range(len(DF['date.pretty']))]
DF['Date']=[DF['date.pretty'][i].strftime("%Y-%m-%d") for i in range(len(DF['date.pretty']))]
DF=DF.rename(columns={'date.hour':'H','date.mday':'DoM','date.mon':'MoY','date.pretty':'Time','tempm':'TemperatureC',
					'hum':'Humidity','wdire':'Wind_Direction','conds':'Conditions'})
DF=DF[['Time','TemperatureC','Humidity','Wind_Direction','Conditions',
		'Date','DoW','DoM','WoY','MoY','QoY','H','HMS']]

DF['Date']=pd.to_datetime(DF['Date']) # convert to time data type the feature ' Date'
#import datetime as datetime
a_day_off=['2016-03-28','2016-05-01','2016-05-05','2016-05-08','2016-05-16','2016-07-14','2016-08-15','2016-11-01',
          '2016-11-11','2017-04-17','2017-05-01','2017-05-08','2017-05-25','2017-06-05','2017-07-14',
           '2017-08-15','2017-11-01','2017-11-11','2018-01-01','2018-04-02','2018-05-01','2018-05-08','2018-05-10',
          '20018-05-21','2018-07-14','2018-08-15','2018-11-01','2018-11-11','2018-12-25']
before_a_day_off=['2016-03-27','2016-04-30','2016-05-04','2016-05-07','2016-05-15','2016-07-13','2016-08-14','2016-10-31',
          '2016-11-10','2017-04-16','2017-04-30','2017-05-07','2017-05-24','2017-06-04','2017-07-13',
           '2017-08-14','2017-10-31','2017-11-10','2017-12-31','2018-04-01','2018-04-30','2018-05-07','2018-05-09',
          '20018-05-20','2018-07-13','2018-08-14','2018-11-01','2018-11-10','2018-12-24']
'''
def schedule(row):
    if row.strftime('%Y-%m-%d') in (a_day_off):
        val=='a_day_off'
    elif row.strftime('%Y-%m-%d') in (before_a_day_off):
        val=='before_a_day_off'
    elif datetime.date(2016,2,13)<row.date()<datetime.date(2016,2,29):
        val='holidays'
    elif datetime.date(2016,4,9)<row.date()<datetime.date(2016,4,25):
        val='holidays'
    elif datetime.date(2016,5,4)<row.date()<datetime.date(2016,5,9):
        val='holidays'
    elif datetime.date(2016,7,5)<row.date()<datetime.date(2016,9,1):
        val='holidays'
    elif datetime.date(2016,10,19)<row.date()<datetime.date(2016,11,3):
        val='holidays'
    elif datetime.date(2016,12,17)<row.date()<datetime.date(2017,1,3):
        val='holidays'
    elif datetime.date(2017,2,18)<row.date()<datetime.date(2017,3,6):
        val='holidays'
    elif datetime.date(2017,4,15)<row.date()<datetime.date(2017,5,2):
        val='holidays'
    elif datetime.date(2017,5,24)<row.date()<datetime.date(2017,5,29):
        val='holidays'
    elif datetime.date(2017,7,8)<row.date()<datetime.date(2017,9,4):
        val='holidays'
    elif datetime.date(2017,10,21)<row.date()<datetime.date(2017,11,6):
        val='holidays'
    elif datetime.date(2017,12,23)<row.date()<datetime.date(2018,1,8):
        val='holidays'
    elif datetime.date(2018,2,10)<row.date()< datetime.date(2018,2,26):
        val='holidays'
    elif datetime.date(2018,4,7)<row.date()<datetime.date(2018,4,23):
        val= 'holidays'
    elif datetime.date(2018,7,7)<row.date()<datetime.date(2018,9,3):
        val='holidays'
    elif datetime.date(2018,10,20)< row.date()< datetime.date(2018,11,5):
        val= 'holidays'
    elif datetime.date(2018,12,22)<row.date()<datetime.date(2019,1,7):
        val='holidays'
    else:
        val='no_event'
    return val

#%%
'''

#%%
# Feature engineering (Scaling features)
DF_scaled=DF[['TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY','MoY', 'QoY', 'H']]
# Import the matrix buit on the training data set
mat_minmax=pd.read_csv('C:/Users/Pham Antoine/mat_minmax.csv',sep=',')
mat_minmax=mat_minmax.set_index(['Unnamed: 0'])
mat_minmax.index.name=None
mat_minmax=mat_minmax.loc[['TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY','MoY', 'QoY', 'H'],:].values.tolist()
list_to_float=DF_scaled.dtypes.loc[DF_scaled.dtypes=='object'].index.values.tolist()# to get columns containing object format
'''
# build a function to convert columns from string to float format
def str_column_to_float(row):
	return float(row)
'''
for i in (list_to_float):
	DF_scaled=DF_scaled.copy()
	DF_scaled.loc[:,i]=DF_scaled.loc[:,i].apply(str_column_to_float)
for i  in range(len(DF_scaled.values)):
	DF_scaled=DF_scaled.copy()
	DF_scaled.iloc[i,:]=[(DF_scaled.values[i][j]-mat_minmax[j][0])/(mat_minmax[j][1]-mat_minmax[j][0]) for j in range(len(mat_minmax))]
#%%


#%%
# Feature engineering (One _Hot_Encoding)
DF_noscaled=DF[DF.columns.difference(['Date','Time','TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY','MoY', 'QoY', 'H'])] # to keep the columns which will be transformed
DF_noscaled=pd.get_dummies(DF_noscaled) # one_hot_encoding
DF_combined=pd.merge(DF_scaled,DF_noscaled,left_index = True, right_index=True,how='inner') # Merge 2 above data sets
context_features=['TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY',
		'MoY', 'QoY', 'H', 'Conditions_Clear', 'Conditions_Fog',
		'Conditions_Heavy Fog', 'Conditions_Heavy Rain',
		'Conditions_Heavy Thunderstorms and Rain', 'Conditions_Light Drizzle',
		'Conditions_Light Rain', 'Conditions_Light Rain Showers',
		'Conditions_Mist', 'Conditions_Mostly Cloudy', 'Conditions_Overcast',
		'Conditions_Partial Fog', 'Conditions_Partly Cloudy',
		'Conditions_Patches of Fog', 'Conditions_Rain',
		'Conditions_Scattered Clouds', 'Conditions_Thunderstorm',
		'Conditions_Thunderstorms and Rain', 'Conditions_Unknown',
		'HMS_08:30:00', 'HMS_09:00:00', 'HMS_09:30:00', 'HMS_10:00:00',
		'HMS_10:30:00', 'HMS_11:00:00', 'HMS_11:30:00', 'HMS_12:00:00',
		'HMS_12:30:00', 'HMS_13:00:00', 'HMS_13:30:00', 'HMS_14:00:00',
		'HMS_14:30:00', 'HMS_15:00:00', 'HMS_15:30:00', 'HMS_16:00:00',
		'HMS_16:30:00', 'HMS_17:00:00', 'HMS_17:30:00', 'HMS_18:00:00',
		'HMS_18:30:00', 'HMS_19:00:00', 'HMS_19:30:00', 'HMS_20:00:00',
		'HMS_20:30:00', 'HMS_21:00:00', 'HMS_21:30:00', 'HMS_22:00:00',
		'HMS_22:30:00', 'HMS_23:00:00', 'HMS_23:30:00', 'Wind_Direction_Calm',
		'Wind_Direction_ENE', 'Wind_Direction_ESE', 'Wind_Direction_East',
		'Wind_Direction_NE', 'Wind_Direction_NNE', 'Wind_Direction_NNW',
		'Wind_Direction_NW', 'Wind_Direction_North', 'Wind_Direction_SE',
		'Wind_Direction_SSE', 'Wind_Direction_SSW', 'Wind_Direction_SW',
		'Wind_Direction_South', 'Wind_Direction_Variable', 'Wind_Direction_WNW',
		'Wind_Direction_WSW', 'Wind_Direction_West', 'schedule_a_day_off',
		'schedule_before_aday_off', 'schedule_holidays', 'schedule_no_event'] #Concatenate  these features to all of ones  used in traing &testing task (for keeping all values that have been  seen by the training model)

DF_allfeatures=pd.DataFrame(np.zeros((0,len(context_features))),columns=context_features) # create a empty data frame containing these above features
DF_to_getcontext=pd.concat([DF_allfeatures,DF_combined], axis=0,join='outer',ignore_index=True)# concatenate 2 df

'''
def replacenull(row):# to handle with NA values generated by the above concatening task
	if np.isnan(row):
		return 0
	else:
		return row
'''
for i in DF_to_getcontext.columns:
	DF_to_getcontext[i]=DF_to_getcontext[i].apply(replacenull) # replace NAs
#Now the data is ready to be used by the model to get the real context  
#%%
# Data preparation to get a data set allowing the clustering model to predict the real context.

n=list(DF_to_getcontext.columns)
out_of_list=pd.Series(n)[~pd.Series(n).isin(pd.Series(context_features))].tolist()
DF_to_getcontext=DF_to_getcontext.loc[:,DF_to_getcontext.columns.difference(out_of_list)]
To_predict_context=DF_to_getcontext[['TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY',
		'MoY', 'QoY', 'H', 'Conditions_Clear', 'Conditions_Fog',
		'Conditions_Heavy Fog', 'Conditions_Heavy Rain',
		'Conditions_Heavy Thunderstorms and Rain', 'Conditions_Light Drizzle',
		'Conditions_Light Rain', 'Conditions_Light Rain Showers',
		'Conditions_Mist', 'Conditions_Mostly Cloudy', 'Conditions_Overcast',
		'Conditions_Partial Fog', 'Conditions_Partly Cloudy',
		'Conditions_Patches of Fog', 'Conditions_Rain',
		'Conditions_Scattered Clouds', 'Conditions_Thunderstorm',
		'Conditions_Thunderstorms and Rain', 'Conditions_Unknown',
		'Wind_Direction_Calm',
		'Wind_Direction_ENE', 'Wind_Direction_ESE', 'Wind_Direction_East',
		'Wind_Direction_NE', 'Wind_Direction_NNE', 'Wind_Direction_NNW',
		'Wind_Direction_NW', 'Wind_Direction_North', 'Wind_Direction_SE',
		'Wind_Direction_SSE', 'Wind_Direction_SSW', 'Wind_Direction_SW',
		'Wind_Direction_South', 'Wind_Direction_Variable', 'Wind_Direction_WNW',
		'Wind_Direction_WSW', 'Wind_Direction_West', 'schedule_a_day_off',
		'schedule_before_aday_off', 'schedule_holidays', 'schedule_no_event']]
#%%

#%%
# Make context prediction by the clustering model
clust_label=pd.DataFrame(clf_clust.predict(To_predict_context),columns=['Context']) # make prediction on the new data, create a DF with a column called ' columns' to store the predicted contexts
clust_label['Context']=['Context_%d' %(i+1) for i in clust_label['Context']] # add strings  'Context_%d'
Item_context_pred=clust_label.join(M_prob, on ='Context')# get probability in according the founded context
Item_context_pred.insert(0,'Time',DF['Time'])
Item_context_pred=Item_context_pred
Item_context_pred.to_csv("Item_context_pred.csv", sep=',',index=True, encoding='uf8')# sometimes important to put encoding == to avoid issue when reading th exported file
