# -*- coding: utf-8 -*-
"""
Title: "Recommender System by Top_N Ranking Based Method_ Application "

	This script can be executed to obtain both external data and the features of the model from streaming input data. 
    
    It also contains an algorithm  which aims to predict items with a high probability of being bought by a customer.
    
    The scrip written under Python can be interacted with C# to deploy the model and produce predictions.

	Authors : 
	Date: from July to November 2017
"""


#%%
import numpy as np
from builtins import print
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from datetime import timedelta
from pandas.io.json import json_normalize
from sklearn.externals import joblib
from urllib.request import urlopen
import json
#import schedule
#import time

#%%

rootPath=os.path.dirname(os.path.realpath(__file__)) + "/../"
paramsPath=rootPath + "params/"


#==============================================================================
#                   Create some math & statistical functions for this job
#==============================================================================
#%%
def square_rooted(x):
	return round(math.sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
	numerateur = sum(a*b for a,b in zip(x,y))
	denominateur = square_rooted(x)*square_rooted(y)
	return round(float(numerateur/denominateur),3)

def mahalanobis_similarity(x,y,z):
	numerateur = (x-y)# x= User's visit duration; y= mean of visit durations
	denominateur = z # z = standart deviation of visit durations
	return round(math.sqrt(pow(numerateur/denominateur,2)),3)

def num_missing(m):
		return sum(m.isnull())

def num_isnan(m):
	return sum(np.isnan(m))
#%%
#==============================================================================
#               CREATE FUNCTIONS IN ORDER TO DO FEATURE ENGINEERING
#==============================================================================
#%%
def partner(row):#Calculate the  features ' Partner' 
	if row ==1:
		no_partner=1
	else:
		no_partner=0
	return no_partner

def partner1(row):
	if row ==2:
		partner1=1
	else:
		partner1=0
	return partner1  

def partner2(row):
	if 2< row < 5:
		partner34=1
	else:
		partner34=0
	return partner34 

def partnergr(row):
	if row >4:
		partnergr=1
	else:
		partnergr=0
	return partnergr

def novelty(row): # calculate the  features 'Novelty'
	if row==1:
		first=1
	else:
		first=0
	return first

def returning(row):
	if row==2:
		returning=1
	else:
		returning=0
	return returning

def returning1(row):
	if row>2:
		returning1=1
	else:
		returning1=0  
	return returning1

def level1(row): #Calculate the  features ' Ticket level'
	if row<=X['avg_ticketU'].describe()[4]:# avg_ticket< 14 euros
		level1=1 # but that would be interessting to test with real values 
	else:
		level1=0
	return level1

def level2(row):
	if X['avg_ticketU'].describe()[4] < row <= X['avg_ticketU'].describe()[5]:
		level2=1 
	else:
		level2=0
	return level2

def level3(row):
	if X['avg_ticketU'].describe()[5] < row <= X['avg_ticketU'].describe()[6]:
		level3=1 
	else:
		level3=0
	return level3


def level4(row):
	if row >= X['avg_ticketU'].describe()[6]:
		level4=1
	else:
		level4=0
	return level4

# for defining contexts, this function can be utilized  to  modify the code number for Sunday from 8 to 1
def replacesunday(row):
	if row == 8:
		return 1
	else:
		return row
#%%

#==============================================================================
#                 IMPORT MATRIX CONTAINING FIT PARAMETERS OF THE MODEL  
#==============================================================================
#%%
X=pd.read_csv(rootPath + 'params/X.csv',sep=',')
X=X.set_index(['D.OrderHeaderID','D.PersonID'])
M_prob=pd.read_csv(rootPath + 'params/M_prob.csv',sep=',')
IT_PROFILE=pd.read_csv(rootPath + 'params/IT_PROFILE.csv',sep=',')
visit_duration_stat=pd.read_csv(rootPath + 'params/visit_duration_stat.csv',sep=',')
It_event_weighted=pd.read_csv(rootPath + 'params/It_event_weighted.csv',sep=',')

#do some more data preparations over these data sets
M_prob=M_prob.set_index(['Unnamed: 0'])
M_prob.index.name=None
IT_PROFILE=IT_PROFILE.set_index(['Unnamed: 0'])
IT_PROFILE.index.name=None
It_event_weighted=It_event_weighted.set_index(['Unnamed: 0'])
It_event_weighted.index.name=None
visit_duration_stat=visit_duration_stat.set_index(['Unnamed: 0'])
visit_duration_stat.index.name=None

seen_data=pd.read_csv(rootPath + 'params/seen_data.csv',sep=',')# data combining training and dev data set containg  all historical data
#seen_data=seen_data.set_index(['D.OrderHeaderID','D.PersonID'])
best_w=[0.8389266,0.15078162,0.01029177] #import the weights founded in traning tasks
#%%

#%%
# do some data preperation
# create a list containing all of these names
oldnames=It_event_weighted.index[It_event_weighted.index.str.contains('_event')]
# remove the string '_event' and create a new list of name
newnames=[oldnames[i][:-6] for i in range(len(oldnames))]
# rename these click events by the list 'newnames'
for i in range(len(It_event_weighted.index)):
    if It_event_weighted.index.values[i] in (oldnames):
        # put  .values
        It_event_weighted.index.values[i]= It_event_weighted.index.values[i][:-6]
    else:
        It_event_weighted.index.values[i]= It_event_weighted.index.values[i] 
#%%

#%%
#load the model that has been fit on the test(dev ) data set called 'Ap2_Eval.ipython'
#from sklearn.externals import joblib
clf_clust = joblib.load(rootPath + 'params/cluster_model.pkl') # reload the fit model to predict context ( Clustering model)
clf_NN = joblib.load(rootPath + 'params/NN_model.pkl') # reload the fit model to predict visit duration ( Neural Network model)

#%%
#==============================================================================
#                   	CONTEXT PREDICTION - WEATHER&TIME INFORMATIONS
#==============================================================================
#%%

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

def schedule(row):
    if row.strftime('%Y-%m-%d') in (a_day_off):
        val='a_day_off'
    elif row.strftime('%Y-%m-%d') in (before_a_day_off):
        val='before_a_day_off'
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


#%%
# Feature engineering (Scaling features)
DF_scaled=DF[['TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY','MoY', 'QoY', 'H']]
# Import the matrix buit on the training data set
mat_minmax=pd.read_csv(rootPath + 'params/mat_minmax.csv',sep=',')
mat_minmax=mat_minmax.set_index(['Unnamed: 0'])
mat_minmax.index.name=None
mat_minmax=mat_minmax.loc[['TemperatureC', 'Humidity', 'DoW', 'DoM', 'WoY','MoY', 'QoY', 'H'],:].values.tolist()
list_to_float=DF_scaled.dtypes.loc[DF_scaled.dtypes=='object'].index.values.tolist()# to get columns containing object format

# build a function to convert columns from string to float format
def str_column_to_float(row):
	return float(row)

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


def replacenull(row):# to handle with NA values generated by the above concatening task
	if np.isnan(row):
		return 0
	else:
		return row

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
#%%

#%%
Products=pd.read_csv("C:/Users/Pham Antoine/products_Bordeaux.csv",sep=';') #Import matrix describing items&product group
Products.columns=['ID','ProductID','ProductName','ProductGroupID','ProductGroupName','GrossPrice','NetPrice','GuidedActivityID','WorkingOrder']
Working_Order_type={0:'Menu', 1:'Beverage', 2:'Starter', 3 :'MainDishes',
                    4 :'Dessert', 5:'Coffee', 6:'Digestive', 7:'Others'}   # create a dictionary that stoke WorkingOrderType
# do some data preparation
Products['WorkingOrderType']=Products['WorkingOrder']
Products['WorkingOrderType'].replace(Working_Order_type, inplace=True)

products=pd.DataFrame(Products.groupby(['ProductGroupName','ProductGroupID','ProductName','ProductID',
                                 'WorkingOrder','WorkingOrderType'],as_index=False).sum()) # create a data set containing all ratings for this user
#%%


#%%
alcohol=['KIR','KIR ROYAL','PUNCH Maison','SANGRIA','RHUM ARRANGE','PICHET PUNCH',
				   'PICHET SANGRIA','WHISKY','RHUM','VODKA','GIN','GET','TEQUILA','MALIBU','BAILEYS','RICARD',
				   'PASTIS','PINEAU','PORTO','LILLET','MARTINI','CHAMPAGNE BOUTEILLE','Cocktail Saint Valentin alcoolisé','CHAMPAGNE COUPE','PELFORTH',
				   'PINT PELFORTH','GRIMBERGEN','HOEGGARDEN','DESPERADOS','KEKETTE EXTRA','KEKETTE RED',
				   'BIERE SANS GLUTEN','EXPRESSO','DECA','DOUBLE EXPRESSO','PAPOLLE BLANC SEC','VERRE PAPOLLE BLANC SEC',
		   'Verre de Bordeaux Rouge Agape','PAPOLLE BLANC MOEL','VERRE PAPOLLE BLANC MOEL','PAPOLLE ROSE','VERRE PAPOLLE ROSE','BIERE PRESSION','EXPRESSO SD',
		   'DECA SD','THE SD','INFUSION SD','BIERE',' PRESSION MAX MENU','BIERE PINTH','Verre Rosé la Colombette','VERRE MAX CUVEE',
		   'MAX CUVEE ','Verre de Boisson Rouge','Verre Chardonnay domaine des deux ruisseaux','VERRE PAPOLLE ROUGE','Domaine La Colombette Rosé','Boisson Rouge']
#%%
#==============================================================================
#                   CREATE A CLASS TO GET INPUT DATA AND DO FEATURE ENGINEERING TASKS
#==============================================================================
#%%
class Feature:
    '''
    Get raw inputs and transform these ones into the features which will be feed to the model
    '''

    def __init__(self,NbDiners,UserID,CreationDatetime):
        self.NbDiners = NbDiners
        self.UserID = UserID
        self.CreationDatetime = CreationDatetime
        self.ParameterList = list()
    
    def get_nbpartner(self):
        # getting the 1st input
        nb_partner=list()
        for i in (partner,partner1,partner2,partnergr):
            nb_partner.append(i(self.NbDiners))
        return nb_partner   
    
    def get_visit(self): 
        # define if it is a new user, a returning user or a familar user
        def top_visit(UserID):
            if self.UserID in (seen_data['D.PersonID'].values):
                # important to convert seen_data['D.PersonID'] to list by .values
                return (np.unique(seen_data.loc[seen_data['D.PersonID']==self.UserID,'nb_visits'])+1)
            else:
                return 1 
        nb_visit=list()
        for j in (novelty,returning,returning1):
            nb_visit.append(j(top_visit(self.UserID)))
        return nb_visit
    
    def get_ticket(self):
        def top_ticket(UserID):
            # define the buying level of the user
            if self.UserID in (seen_data['D.PersonID'].values):
                # important to convert seen_data['D.PersonID'] to list by .values
                return (np.unique(seen_data.loc[seen_data['D.PersonID']==self.UserID,'avg_ticketU']))
            else:
                return 0 # assigning 0 to an unseen user means his buying level is supposed at the level1  
        ticket_level=list()
        for k in (level1,level2,level3,level4):
            ticket_level.append(k(top_ticket(self.UserID))) # PUT THE NEW DATA HERE
        return ticket_level
    
    def get_user_profile(self):
        a = self.get_nbpartner() + self.get_visit() + self.get_ticket()
        return a
    
    def get_timedelta(self,timenow):
        #♠time_delta= (datetime.datetime.now() - self.CreationDatetime)# calculate the passed time
        self.timenow = timenow
        time_delta=(self.timenow-self.CreationDatetime).total_seconds()/60
        return time_delta
    
    def add_Event(self, Parameter):
        self.ParameterList.append(Parameter)
        return self.ParameterList

    def Event2vec(self):
        #newevent= list()
        #newevent.append(self.Parameter)
        #return newevent
        click_event=list()
        for i in range(len(np.unique(self.ParameterList))): # add np.unique
            click_event.append(np.unique(self.ParameterList)[i])
        #return click_event
        event_detection=pd.DataFrame(np.zeros((0,len(It_event_weighted.index))),columns=It_event_weighted.index) #create a data frame whose columns contain all eventual click events
        val1=1
        val2=0
        event_vector=list()
        for i in range(len(event_detection.columns)):
            if event_detection.columns[i] in click_event:
                # important to put .tolist()
                event_vector.append(val1)# add 1 to the item everytime an click on the this item is captured; so the cosinus changes everytime
            else:
                event_vector.append(val2)
        return event_vector 


    '''
    def get_Event(self):
        click_event=list()
        for i in range(len(np.unique(self.Parameter))): # add np.unique
            click_event.append(np.unique(self.Parameter)[i])
        #return click_event
        event_detection=pd.DataFrame(np.zeros((0,len(It_event_weighted.index))),columns=It_event_weighted.index) #create a data frame whose columns contain all eventual click events
        val1=1
        val2=0
        event_vector=list()
        for i in range(len(event_detection.columns)):
            if event_detection.columns[i] in click_event:
                # important to put .tolist()
                event_vector.append(val1)# add 1 to the item everytime an click on the this item is captured; so the cosinus changes everytime
            else:
                event_vector.append(val2)
        return event_vector 
	'''

#%%

#%%

'''
# simulating inputs

Parameter = ['HAMBOURGEOIS DU MOMENT', 'HAMPE', 'HOEGGARDEN', 'INFUSION', 'HAMBOURGEOIS DU MOMENT','JUS FRUIT MAISON']
NbDiners = 3
UserID = 32000
CreationDatetime = pd.to_datetime(datetime.datetime.now()- datetime.timedelta(hours=1))

#%%
#CreationDatetime
#(datetime.datetime.now()-CreationDatetime).total_seconds()

#%%
f = Feature(NbDiners,UserID,CreationDatetime,Parameter) # call the class 'Feature'
#%%

#%%

print(f.get_timedelta())
print(f.get_nbpartner())
print(f.get_visit())
print(f.get_ticket())
print(f.get_user_profile())
print(f.Event2vec())
'''
#%%


def testtime(epoch):# CreationDateTime changed to ' epoch' in C# code
	#t = datetime.date(CreationDatetime)
	#t = datetime.datetime.strftime(t, "%Y-%m-%d %H:%M:%S") # or from datetime import datetime
	t = pd.to_datetime(epoch, unit='s')# to convert epoch type to date type
	return t

#%%
#ATTENTION: this is a simulation, remove datetime.timedelta when doing real prediction
def Context2Item(Item_context_pred):
    time_to_context = max(Item_context_pred.loc[Item_context_pred['Time']<datetime.datetime.now()+datetime.timedelta(hours=2),:]['Time']) # to get the latest datetime before the creation datetime
    context2item = Item_context_pred.loc[Item_context_pred['Time']==time_to_context,M_prob.columns].values # to get the row and columns ( predited probabilities) corresponding to the last datetime 
    return context2item
#%%

#%%
#user_profile=f.get_user_profile()
def User2Item(user_profile):
    user2item=np.array([cosine_similarity(user_profile,y) for y in IT_PROFILE.values]).reshape(len(np.array([user_profile])),
                                IT_PROFILE.shape[0]) # compute the similarity beween the user and every item
    user2item=pd.DataFrame(user2item,columns=IT_PROFILE.index)
    return user2item
#%%

#%%
'''
event_detection=pd.DataFrame(np.zeros((0,len(It_event_weighted.index))),columns=It_event_weighted.index) #create a data frame whose columns contain all eventual click events
val1=1
val2=0
event_vector=list()
for i in range(len(event_detection.columns)):
    if event_detection.columns[i] in f.get_Event():
        # important to put .tolist()
        event_vector.append(val1)# add 1 to the item everytime an click on the this item is captured; so the cosinus changes everytime
    else:
        event_vector.append(val2)
'''
#%%

#%%
#event_vector=f.get_Event()
def Event2Item(event_vector):
    event2item=[cosine_similarity(event_vector,y)  for y in It_event_weighted.T.values] # predict buying probability in according to captured click events
    event2item=np.array(event2item).reshape(1,It_event_weighted.shape[1]) #create a data frame containing all of  items with the corresponded buying probability
    event2item= pd.DataFrame( event2item, columns=It_event_weighted.columns)
    return event2item
#%%
#%%
train_visit_duration = np.array(visit_duration_stat.loc[:,['mean_visit_duration','std_visit_duration']]).tolist() # from the data set 'visit_duration', take  the mean and the standart deviation
#%%
# Solution 1 ( get data from the table Event)
'''    
    #t=np.arange(2,60,5)# simulate values of visit_duration : every 5 min from 2 min to 60 min; PUT THE NEW DATA HERE
	# R code if using Data Base:
	# difftime(max(as.POSIXct(E.TimeStamp)),min(as.POSIXct(E.TimeStamp)),units='min')
	#print(t)
	#print(seen_data.columns)
'''
'''
	#2nd solution: predict visit duration by a model
	#MLPR1= MLPRegressor(hidden_layer_sizes=(200, ), activation='tanh', solver='sgd', alpha=0.001, batch_size='auto', 
	 #                  learning_rate='adaptive', learning_rate_init=0.01, power_t=0.6, max_iter=300, shuffle=True,
	  #                 random_state=1,tol=0.0001, verbose=False, warm_start=False, momentum=0.99, nesterovs_moment
	#features_to_duration=['H.NbDiners','TemperatureC','Humidity','DoW','DoM','WoY','MoY','QoY','H']# scaled data
	#visit_duration=clf_NN.predict(features_to_duration)
	#visit_duration-t for t in np.arange(2,120,1.5)
'''
#%%
'''
time_delta= (datetime.datetime.now() - f.get_TimeStamp())# calculate the passed time
time_delta=time_delta.total_seconds()/60
'''
#%%

#%%
#time_delta= 21.99999999
def Mah(time_delta):
    mah=[mahalanobis_similarity(time_delta,y,z) for y,z in train_visit_duration] # applying the function created to compute the Mahalanobis distance
    minmah= np.min([i for i in mah if i >0])
    mah=[ i if i >0 else minmah for i in mah]
    mah=np.array(mah).reshape(1,visit_duration_stat.shape[0])# shape of an array 1x125
    mah=pd.DataFrame(0.1/mah,columns= visit_duration_stat.index.values)# 1/ maha 
    return mah
#%%


#%%
#best_w=[0.8389266,0.15078162,0.01029177] #import the weights founded in traning tasks
W1,W2,W3=best_w
#%%

#%%
class Recommendation:
    '''
    Get raw inputs and transform these ones into the features which will be feed to the model
    '''
    def __init__(self,Item_context_pred,user_profile,event_vector,time_delta,W1,W2,W3):
        self.Item_context_pred = Item_context_pred
        self.user_profile = user_profile
        self.event_vector = event_vector
        self.time_delta = time_delta
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
    '''
    def Context2Item(Item_context_pred):
        time_to_context = max(Item_context_pred.loc[Item_context_pred['Time']<f.get_TimeStamp(),:]['Time']) # to get the latest datetime before the creation datetime
        Context2Item = Item_context_pred.loc[Item_context_pred['Time']==time_to_context,M_prob.columns].values # to get the row and columns ( predited probabilities) corresponding to the last datetime 
        return Context2Item
    
    def User2Item(user_profile):
        User2Item=np.array([cosine_similarity(user_profile,y) for y in IT_PROFILE.values]).reshape(len(np.array([user_profile])),
                                IT_PROFILE.shape[0]) # compute the similarity beween the user and every item
        User2Item=pd.DataFrame(User2Item,columns=IT_PROFILE.index)
        return User2Item
    
    def Event2Item(event_vector):
        Event2Item=[cosine_similarity(event_vector,y)  for y in It_event_weighted.T.values] # predict buying probability in according to captured click events
        Event2Item=np.array(Event2Item).reshape(1,It_event_weighted.shape[1]) #create a data frame containing all of  items with the corresponded buying probability
        Event2Item= pd.DataFrame( Event2Item, columns=It_event_weighted.columns)
        return Event2Item
    
    def Mah(timedelta):
        mah=[mahalanobis_similarity(time_delta,y,z) for y,z in train_visit_duration] # applying the function created to compute the Mahalanobis distance
        minmah= np.min([i for i in mah if i >0])
        mah=[ i if i >0 else minmah for i in mah]
        mah=np.array(mah).reshape(1,visit_duration_stat.shape[0])# shape of an array 1x125
        mah=pd.DataFrame(0.1/mah,columns= visit_duration_stat.index.values)# 1/ maha 
    '''
    def get_prediction(self):
        ''' get the global prediction
        '''
        Rating=100*(self.W1*Context2Item(self.Item_context_pred) + self.W2*User2Item(self.user_profile) + self.W2*Event2Item(self.event_vector) + self.W3*Mah(self.time_delta) ) # Compute the matrix of ratings
        Ranking=Rating.T # modified from dev&test
        Ranking.rename(columns={0:'score'},inplace=True)
        Ranking=Ranking.sort_values(by=['score'],ascending=False) # ranking by descending order
        return Ranking
    
    def get_topNprediction(self, top):
        '''
        get top N of Items based on the global ranking
        '''
        self.top = top
        top_N_ranking = self.get_prediction().index.values
        #top_n_ranking=k(Ranking).index.values
        print('Here is the top %d of Items to recommend to the User:' %self.top)
        for i in top_N_ranking[:self.top]:
            print(i, end = ' *** ')
    
    def get_topNseqprediction(self,top):
        '''
        get top N of Items by filtering sold items
        '''
        self.top = top
        ordered_item=list()
        for i in OrderHeader_ProductID:
            ordered_item.append(i)
            ordered_WO_type=np.ravel(products.loc[products['ProductName'].isin(ordered_item),['WorkingOrderType']].values).tolist()
        # find items that doesn't belong to Working Order type of sold items
        #top_n_ranking[pd.Series(top_n_ranking).isin (products.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])]
            print('Following the last choice of the user here is the top %d of Items to recommend :' %top)
        for i in self.get_prediction().index.values[pd.Series(self.get_prediction().index.values).isin (products.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]:
            print(i, end = ' *** ')
    '''
    def plot_topNseqprediction(self,top):
        self.top = top
        top_n_ranking = self.get_prediction()
        top_n_ranking1=top_n_ranking[pd.Series(top_n_ranking).isin (p.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]
        return sns.barplot(y=k(r.loc[r['ProductName'].isin (top_n_ranking1)])['ProductName'][:top],x=k(r.loc[r['ProductName'].isin (top_n_ranking1)]['score'][:top])),sns.plt.suptitle('Top following %d items to recommend with scores after the last choice of the user' %top);
    '''
    def get_fltrprediction(self,user_age):
        ''' 
        do filtering to exclude items not allowed to sell to child
        '''
        #self.Ranking= self.get_prediction
        self.user_age = user_age
        #self.alcohol = alcohol
        if self.user_age >18:
            return self.get_prediction()
        elif np.isnan(self.user_age):
            return self.get_prediction()
        else:
            d=self.get_prediction().loc[~self.get_prediction().index.isin(alcohol)] #filtering products being in the list we have created above
            return d
    def plot_topNprediction(self,top):
        self.top=top
        top_ranking = self.get_prediction()
        return sns.barplot(y=top_ranking.index[:self.top],x=top_ranking['score'][:self.top]),sns.plt.suptitle('Top %d items to recommend with scores' %self.top);
    
    def get_predictionGroup(self):
        '''
        Make recommendation by each product group
        '''
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description of item groups
        # 'left' for conserving all rows of the data set ' Products'
        #return(Ranking_final.to_csv('Ranking_final.csv',sep=','))
        #print(num_missing(Ranking_final['score'])) # check missing values over items to recommend
        # Note: there are 5 new items which have no been seen in training tasks
        #print('As following the %d unseen items in the training & the test data sets : ' %num_missing(Ranking_final['score']))
        #print(Ranking_final.loc[np.isnan(Ranking_final['score']),['ProductName','score']])
        Ranking_final_PGroup=list() # create empty lists for  each product group
        for i in np.unique(Ranking_final['ProductGroupID']):
            Ranking_final_PGroup.append(Ranking_final.set_index(['ProductGroupID','ProductGroupName']).loc[i]) # set the first index ='ProductGroupID' and use .loc by it
        for i in range(len(Ranking_final_PGroup)):
            print((Ranking_final_PGroup[i]).loc[:,['ProductName','score']]) # keep only 2 columns : 'ProductName','score'
        #recommend top 3 items for each product group
        for i in range(len(Ranking_final_PGroup)):
            top_group=3
            print(Ranking_final_PGroup[i]['ProductName'][:top_group])
        
    def plot_scores_his(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description of item groups
        f, ax = plt.subplots(figsize=(8,6))
        sns.distplot([value for value in Ranking_final['score'] if not math.isnan(value)])
        plt.xlabel('score')
        plt.show()
     
    def plot_scoresWO(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description of item groups
        f, ax = plt.subplots(figsize=(12,8))
        fig = sns.boxplot(y=Ranking_final['ProductGroupName'], x=Ranking_final['score']) # making a Box Plot to observe average
        return fig
     
    def get_predictionWO(self):
        '''
        Make recommendation by Working_Order type
        '''
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description of item groups
        Ranking_final_WOtype=list() # create  empty lists for  each product group
        for i in np.unique(Ranking_final['WorkingOrder']):
            Ranking_final_WOtype.append(Ranking_final.set_index(['WorkingOrder','WorkingOrderType']).loc[i])
        for i in range(len(Ranking_final_WOtype)): # keep only some columns
            print((Ranking_final_WOtype[i]).loc[:,['ProductName','score']])
        
    def get_optprediction(self):
        '''
         Product price based recommendation optimization
        '''
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        Ranking_final1.index
        # create a column called 'total_score' that sums the score and the net price of the item
        Ranking_final1['score']=Ranking_final1['score'].round(3)
        Ranking_final1['optimized_score']=(Ranking_final1[['NetPrice']].values+Ranking_final1[['score']].values)
        Ranking_opt=Ranking_final1[['ProductGroupName','WorkingOrderType','ProductName','NetPrice','score','optimized_score']].sort_values(ascending=False,by=['optimized_score'])
        Ranking_opt.index=range(len(Ranking_opt))
        #return(Ranking_opt.to_csv('Ranking_opt.csv',sep=','))
        #Recommendation by taking in account the price of item and group by WorkingOrderType
        print('Price aware recommendation by WorkingOrderType')
        for i in (np.unique(Ranking_opt[['WorkingOrderType']])):
             print(Ranking_opt.loc[Ranking_opt['WorkingOrderType']==i])
             #Recommendation by taking in account the price of item and group by ProductGroup
             print('Price aware recommendation by ProductGroupName')
        for i in (np.unique(Ranking_opt[['ProductGroupName']])):
            print(Ranking_opt.loc[Ranking_opt['ProductGroupName']==i])
        # Create a data set containg recommendation by both WorkingOrderType&ProductGroupName
            print('Price aware recommendation by WorkingOrderType&ProductGroupName')
            print(Ranking_opt.set_index(['WorkingOrderType','ProductGroupName']).sort_index())
     
    def plot_optprediction(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        #Ranking_final1.index
        # create a column called 'total_score' that sums the score and the net price of the item
        #Ranking_final1['score']=Ranking_final1['score'].round(3)
        #Ranking_final1['optimized_score']=(Ranking_final1[['NetPrice']].values+Ranking_final1[['score']].values)
        fig, ax = plt.subplots()
        print(Ranking_final1.plot.scatter(x='score',y='NetPrice',xlim=(3,17),ylim=(0,22),
                                          s=Ranking_final['score'].values*20,c=Ranking_final1['score'],
                                          cmap="coolwarm",figsize=(20,14),ax=ax));
        for i, txt in enumerate(Ranking_final1['ProductName']):
            ax.annotate(txt, (Ranking_final1['score'][i],Ranking_final1['NetPrice'][i]),
                        textcoords='data', ha='left', va='bottom',rotation=30,
                        bbox=dict(boxstyle='round,pad=0.05', fc='yellow', alpha=0.4),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'));	
        
    def plot_optpredictionWO(self):
        r=self.get_prediction().reset_index()
        r=r.rename(columns={'index':'ProductName'})
        Ranking_final=pd.merge(products,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description
        Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
        Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
        Ranking_final1.index=range(len(Ranking_final1))
        # plot all selected items by WorkingOrederType
        g = sns.FacetGrid(Ranking_final1, col="WorkingOrderType", hue="ProductName")
        print(g.map(plt.scatter, "score", "NetPrice", alpha=.9));
        print(g.add_legend());     
#%%

'''
#event_vector = f.get_Event()

#%%
#time_delta=f.get_timedelta()
#%%
#time_delta
#Mah(time_delta)
#%%

#rec=Recommendation(Item_context_pred,user_profile,event_vector,time_delta,W1,W2,W3)
#%%
#%%
#rec.get_prediction()
#%%
'''
'''
#%%
rec.get_topNprediction(20)
#%%
#==============================================================================
#                            MAKING SCORE PREDICTION
#==============================================================================
rec.plot_scores_his()
rec.plot_scoresWO()
rec.plot_optprediction()
rec.plot_optpredictionWO()
#==============================================================================
#                            Filtering alcohol items
#==============================================================================
#%%

def k(df):
    if user_age >18:
        return df
    elif np.isnan(user_age):
        return df
    else:
        d=df[~df.index.isin(alcohol)] #filtering products being in the list we have created above
        return d

#%%
#%%
user_age=50
alcohol=['KIR','KIR ROYAL','PUNCH Maison','SANGRIA','RHUM ARRANGE','PICHET PUNCH',
				   'PICHET SANGRIA','WHISKY','RHUM','VODKA','GIN','GET','TEQUILA','MALIBU','BAILEYS','RICARD',
				   'PASTIS','PINEAU','PORTO','LILLET','MARTINI','CHAMPAGNE BOUTEILLE','Cocktail Saint Valentin alcoolisé','CHAMPAGNE COUPE','PELFORTH',
				   'PINT PELFORTH','GRIMBERGEN','HOEGGARDEN','DESPERADOS','KEKETTE EXTRA','KEKETTE RED',
				   'BIERE SANS GLUTEN','EXPRESSO','DECA','DOUBLE EXPRESSO','PAPOLLE BLANC SEC','VERRE PAPOLLE BLANC SEC',
		   'Verre de Bordeaux Rouge Agape','PAPOLLE BLANC MOEL','VERRE PAPOLLE BLANC MOEL','PAPOLLE ROSE','VERRE PAPOLLE ROSE','BIERE PRESSION','EXPRESSO SD',
		   'DECA SD','THE SD','INFUSION SD','BIERE',' PRESSION MAX MENU','BIERE PINTH','Verre Rosé la Colombette','VERRE MAX CUVEE',
		   'MAX CUVEE ','Verre de Boisson Rouge','Verre Chardonnay domaine des deux ruisseaux','VERRE PAPOLLE ROUGE','Domaine La Colombette Rosé','Boisson Rouge']
#%%

#%%
rec.get_fltrprediction(17)
#%%
#%%
# get the top _N of items which will be recommended to this user & print the top_N items by leur ranking

top_n_ranking=k(Ranking).index.values
top=20
print('Here is the top %d of Items to recommend to the User:' %top)
for i in top_n_ranking[:top]:
    print(i, end = ' *** ')

#%%

#%%
#==============================================================================

#                       Visualize recommended items with their scores
#==============================================================================
#%%

top_ranking=k(Ranking)
sns.barplot(y=top_ranking.index[:top],x=top_ranking['score'][:top])
sns.plt.suptitle('Top %d items to recommend with scores' %top);

#%%

#%%
rec.plot_topNprediction(30)
#%%
#==============================================================================
#                           MAKE RECOMMENDATION IN ACCORDING TO PRODUCT GROUPS
#==============================================================================
#%%

Products=pd.read_csv("C:/Users/Pham Antoine/products_Bordeaux.csv",sep=';') #Import matrix describing items&product group
Products.columns=['ID','ProductID','ProductName','ProductGroupID','ProductGroupName','GrossPrice','NetPrice','GuidedActivityID','WorkingOrder']
Working_Order_type={0:'Menu', 1:'Beverage', 2:'Starter', 3 :'MainDishes',
                    4 :'Dessert', 5:'Coffee', 6:'Digestive', 7:'Others'}   # create a dictionary that stoke WorkingOrderType
# do some data preparation
Products['WorkingOrderType']=Products['WorkingOrder']
Products['WorkingOrderType'].replace(Working_Order_type, inplace=True)

p=pd.DataFrame(Products.groupby(['ProductGroupName','ProductGroupID','ProductName','ProductID',
                                 'WorkingOrder','WorkingOrderType'],as_index=False).sum()) # create a data set containing all ratings for this user
r=k(Ranking).reset_index()
r=r.rename(columns={'index':'ProductName'})
Ranking_final=pd.merge(p,r,on=('ProductName'),how='left').sort_values('score',ascending=False) # merge it with the one containing description of item groups
# 'left' for conserving all rows of the data set ' Products'
#return(Ranking_final.to_csv('Ranking_final.csv',sep=','))
print(num_missing(Ranking_final['score'])) # check missing values over items to recommend
# Note: there are 5 new items which have no been seen in training tasks
print('As following the %d un seen items in the training & the test data sets : ' %num_missing(Ranking_final['score']))
print(Ranking_final.loc[np.isnan(Ranking_final['score']),['ProductName','score']])

#%%

#%%
rec.get_predictionGroup()
#%%
#==============================================================================
#                            MAKE RECOMMENDATION FOR EACH PRODUCT GROUP
#==============================================================================
#%%MAKE RECOMMENDATION IN ACCORDING TO WORKING ORDER TYPE

Ranking_final_PGroup=list() # create empty lists for  each product group
for i in np.unique(Ranking_final['ProductGroupID']):
    Ranking_final_PGroup.append(Ranking_final.set_index(['ProductGroupID','ProductGroupName']).loc[i]) # set the first index ='ProductGroupID' and use .loc by it

for i in range(len(Ranking_final_PGroup)):
    print((Ranking_final_PGroup[i]).loc[:,['ProductName','score']]) # keep only 2 columns : 'ProductName','score'
#recommend top 3 items for each product group
for i in range(len(Ranking_final_PGroup)):
    top_group=3
    print(Ranking_final_PGroup[i]['ProductName'][:top_group])

#%%
#==============================================================================
#                           
#==============================================================================
#%%

Ranking_final_WOtype=list() # create  empty lists for  each product group
for i in np.unique(Ranking_final['WorkingOrder']):
    Ranking_final_WOtype.append(Ranking_final.set_index(['WorkingOrder','WorkingOrderType']).loc[i])
for i in range(len(Ranking_final_WOtype)): # keep only some columns
    print((Ranking_final_WOtype[i]).loc[:,['ProductName','score']])

#%%

#%%
rec.get_predictionWO()
#%%
#==============================================================================
#                           FILTERING ALREADY ITEM ORDERED
#==============================================================================
#%%
# Simulate items ordered by an user who has been selecting 1,2,3 or 4 items
OrderHeader_ProductID=np.random.choice(rec.get_prediction().index, size=np.random.randint(1,4),replace=True)# PUT THE NEW DATA HERE
print(OrderHeader_ProductID.tolist())
#%%
#%%

ordered_item=list()
for i in OrderHeader_ProductID:
    ordered_item.append(i)
    ordered_WO_type=np.ravel(products.loc[products['ProductName'].isin(ordered_item),['WorkingOrderType']].values).tolist()
# find items that doesn't belong to Working Order type of sold items
#top_n_ranking[pd.Series(top_n_ranking).isin (products.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])]
top=20
print('Following the last choice of the user here is the top %d of Items to recommend :' %top)
for i in rec.get_prediction().index.values[pd.Series(rec.get_prediction().index.values).isin (products.loc[~products['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]:
    print(i, end = ' *** ')

#%%

#%%
rec.get_topNseqprediction(20)
#%%

#==============================================================================
# 											 OPTIMIZATION BY PRODUCT PRICE AWARE RECOMMENDATIONS 
#==============================================================================

#%%
# Define a cutoff and plot items with their scores and  net prices     
                           
Ranking_final1=pd.DataFrame(Ranking_final.loc[(Ranking_final['score']>6)|((Ranking_final['NetPrice']>7.5)&(Ranking_final['score']>4)),:])
Ranking_final1.drop_duplicates(['ProductName'], keep='first',inplace=True)
Ranking_final1.index=range(len(Ranking_final1))
Ranking_final1.index
#return(Ranking_final1.to_csv('Ranking_final1.csv',sep=','))

#%%
#%%
# create a column called 'total_score' that sums the score and the net price of the item

Ranking_final1['score']=Ranking_final1['score'].round(3)
Ranking_final1['optimized_score']=(Ranking_final1[['NetPrice']].values+Ranking_final1[['score']].values)
Ranking_opt=Ranking_final1[['ProductGroupName','WorkingOrderType','ProductName','NetPrice','score','optimized_score']].sort_values(ascending=False,by=['optimized_score'])
Ranking_opt.index=range(len(Ranking_opt))
#return(Ranking_opt.to_csv('Ranking_opt.csv',sep=','))#
									
#Recommendation by taking in account the price of item and group by WorkingOrderType

print('Price aware recommendation by WorkingOrderType')
for i in (np.unique(Ranking_opt[['WorkingOrderType']])):
    print(Ranking_opt.loc[Ranking_opt['WorkingOrderType']==i])
#Recommendation by taking in account the price of item and group by ProductGroup
print('Price aware recommendation by ProductGroupName')
for i in (np.unique(Ranking_opt[['ProductGroupName']])):
    print(Ranking_opt.loc[Ranking_opt['ProductGroupName']==i])
# Create a data set containg recommendation by both WorkingOrderType&ProductGroupName
print('Price aware recommendation by WorkingOrderType&ProductGroupName')
print(Ranking_opt.set_index(['WorkingOrderType','ProductGroupName']).sort_index())
#return(Ranking_final.to_csv('Ranking_final.csv',sep=','))
#return(Ranking_final1.to_csv('Ranking_final1.csv',sep=','))
#return(Ranking_opt.to_csv('Ranking_opt.csv',sep=','))
#return Ranking_final.to_csv('Ranking_final.csv',sep=','),Ranking_final1.to_csv('Ranking_final1.csv',sep=',')# if using function __main__
#Ranking_opt.to_csv('Ranking_opt.csv',sep=','),Ranking_opt.loc[Ranking_opt['WorkingOrderType']==i]

#%%
#==============================================================================
#                           VISUALIZATION OF  RECOMMENDATIONS 
#==============================================================================
#%%
#Visualize items to recommend following the last choice of the user

top_n_ranking1=top_n_ranking[pd.Series(top_n_ranking).isin (p.loc[~p['WorkingOrderType'].isin (ordered_WO_type)]['ProductName'])][:top]
sns.barplot(y=k(r.loc[r['ProductName'].isin (top_n_ranking1)])['ProductName'][:top],x=k(r.loc[r['ProductName'].isin (top_n_ranking1)]['score'][:top]))
sns.plt.suptitle('Top following %d items to recommend with scores after the last choice of the user' %top);

#%%
#%%

figsize=(16,14)
def hist_viz(df,feature):
    f, ax = plt.subplots(figsize=(10,8))
    ax = sns.distplot([value for value in df[feature] if not math.isnan(value)])#import math
    plt.xlabel(feature)
    plt.show()
print(hist_viz(Ranking_final,'score')) # We need to specify a feature vector

#%%
#%%

f, ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(y=Ranking_final['ProductGroupName'], x=Ranking_final['score']) # making a Box Plot to observe average
print(fig);

#%%

fig, ax = plt.subplots();
print(Ranking_final.plot.scatter(x='score',y='NetPrice',xlim=(0,16),ylim=(0,22),
							s=Ranking_final['score'].values*30,c=Ranking_final['score'],
										cmap="coolwarm",figsize=(10,8),ax=ax));

#%%

#%%

fig, ax = plt.subplots();
print(Ranking_final1.plot.scatter(x='score',y='NetPrice',xlim=(3,17),ylim=(0,22),
							s=Ranking_final['score'].values*20,c=Ranking_final1['score'],
										cmap="coolwarm",figsize=(20,14),ax=ax));
for i, txt in enumerate(Ranking_final1['ProductName']):
    ax.annotate(txt, (Ranking_final1['score'][i],Ranking_final1['NetPrice'][i]),
                textcoords='data', ha='left', va='bottom',rotation=30,
                bbox=dict(boxstyle='round,pad=0.05', fc='yellow', alpha=0.4),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'));		

#%%
#%%

# plot all selected items by WorkingOrederType
g = sns.FacetGrid(Ranking_final1, col="WorkingOrderType", hue="ProductName")
print(g.map(plt.scatter, "score", "NetPrice", alpha=.9));
print(g.add_legend());

#%%

#==============================================================================
# Periodically Re-fit or Update??
#==============================================================================
'''


























