
import numpy as np
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
import smtplib # No need to install this library since it is a built-in package for Python ( sending email)
import os
rootPath=os.path.dirname(os.path.realpath(__file__)) + "/../"
paramsPath=rootPath + "params/"

#%%
X=pd.read_csv(paramsPath+'X.csv',sep=',')
X=X.set_index(['D.OrderHeaderID','D.PersonID'])
#%%
#==============================================================================
#                   Create some math & statistical functions for this job
#==============================================================================
#%%
def square_rooted(x):
	return round(math.sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
	numerateur = sum(a*b for a,b in zip(x,y))
	denominateur = square_rooted(x)*square_rooted(y)
	if denominateur!=0:
		return round(float(numerateur/denominateur),3)
	else:
		return float(0.0)

def mahalanobis_similarity(x,y,z):
	numerateur = (x-y)# x= User's visit duration; y= mean of visit durations
	denominateur = z # z = standart deviation of visit durations
	return round(math.sqrt(pow(numerateur/denominateur,2)),3)

def num_missing(m):
		return sum(m.isnull())

def num_isnan(m):
	return sum(np.isnan(m))

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

stat=pd.read_csv(paramsPath+'X.csv',sep=',')
stat=stat.set_index(['D.OrderHeaderID','D.PersonID']) 
def level1(row): #Calculate the  features ' Ticket level'
   global stat 
   if row<=stat['avg_ticketU'].describe()[4]:
       # avg_ticket< 14 euros
       level1=1 # but that would be interessting to test with real values 
   else:
       level1=0
   return level1

def level2(row):
    global stat
    if stat['avg_ticketU'].describe()[4] < row <= stat['avg_ticketU'].describe()[5]:
            level2=1 
    else:
            level2=0
    return level2

def level3(row):
    global stat
    if stat['avg_ticketU'].describe()[5] < row <= stat['avg_ticketU'].describe()[6]:
        level3=1 
    else:
        level3=0
    return level3


def level4(row):
    global stat
    if row >= stat['avg_ticketU'].describe()[6]:
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


def schedule(row):
    a_day_off=['2016-03-28','2016-05-01','2016-05-05','2016-05-08','2016-05-16','2016-07-14','2016-08-15','2016-11-01',
          '2016-11-11','2017-04-17','2017-05-01','2017-05-08','2017-05-25','2017-06-05','2017-07-14',
           '2017-08-15','2017-11-01','2017-11-11','2018-01-01','2018-04-02','2018-05-01','2018-05-08','2018-05-10',
          '20018-05-21','2018-07-14','2018-08-15','2018-11-01','2018-11-11','2018-12-25']
    before_a_day_off=['2016-03-27','2016-04-30','2016-05-04','2016-05-07','2016-05-15','2016-07-13','2016-08-14','2016-10-31',
          '2016-11-10','2017-04-16','2017-04-30','2017-05-07','2017-05-24','2017-06-04','2017-07-13',
           '2017-08-14','2017-10-31','2017-11-10','2017-12-31','2018-04-01','2018-04-30','2018-05-07','2018-05-09',
          '20018-05-20','2018-07-13','2018-08-14','2018-11-01','2018-11-10','2018-12-24']
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

# build a function to convert columns from string to float format
def str_column_to_float(row):
	return float(row)

def replacenull(row):# to handle with NA values generated by the above concatening task
	if np.isnan(row):
		return 0
	else:
		return row
		
		
def sendemail(from_addr, to_addr_list, cc_addr_list,
              subject, message,
              login, password,
              smtpserver='smtp.gmail.com:587'):
    header  = 'From: %s\n' % from_addr
    header += 'To: %s\n' % ','.join(to_addr_list)
    header += 'Cc: %s\n' % ','.join(cc_addr_list)
    header += 'Subject: %s\n\n' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
    return problems