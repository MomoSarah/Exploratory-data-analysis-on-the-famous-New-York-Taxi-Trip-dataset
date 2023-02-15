#!/usr/bin/env python
# coding: utf-8

# **Import Necessary Libraries**

# In[1]:


#Importing necessary libraries
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from io import BytesIO
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as dt
sns.set()
import warnings; warnings.simplefilter('ignore')


# In[2]:


#Reading Data set
url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet'
response = requests.get(url)
df = pd.read_parquet(BytesIO(response.content))


# **Explore Data**

# In[3]:


#Exploring Data
df.info()


# 'ehail_fee' contains null value. 

# In[4]:


#Dropping null column
df=df.drop('ehail_fee', axis=1)


# In[5]:


df.info()


# In[6]:


#Adding trip ID Column 
prefix = "TripID_"
df.insert(0, "trip_id", prefix + df.index.astype(str))


# In[7]:


df.head()


# In[8]:


df.tail()


# **Column Details**
# <br> trip_id- a unique identifier for each trip
# <br> VendorID- a code indicating the provider associated with the trip record
# <br> lpep_pickup_datetime - date and time when the meter was engaged
# <br> lpep_dropoff_datetime- date and time when the meter was disengaged
# <br> store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip.
# <br> RatecodeID- The final rate code in effect at the end of the trip.
# 1= Standard rate
# 2=JFK
# 3=Newark
# 4=Nassau or Westchester
# 5=Negotiated fare
# 6=Group ride 
# <br> PULocationID- Taxi Zone in which the taximeter was engaged
# <br> DOLocationID- Taxi Zone in which the taximeter was disengaged
# <br> passenger_count - the number of passengers in the vehicle (driver entered value)
# <br> trip_distance- The elapsed trip distance in miles reported by the taximeter.
# <br> fare_amount- The time-and-distance fare calculated by the meter.
# <br> extra- Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges.
# <br> mta_tax-$0.50 MTA tax that is automatically triggered based on the metered rate in use.
# 

# <br> tip_amount- 	Tip amount – This field is automatically populated for credit card tips. Cash tips are not included.
# <br> tolls_amount- Total amount of all tolls paid in trip.
# <br> improvement_surcharge- $0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015. 
# <br> total_amount- The total amount charged to passengers. Does not include cash tips.
# <br> payment_type- A numeric code signifying how the passenger paid for the trip.
# 1= Credit card
# 2= Cash
# 3= No charge
# 4= Dispute
# 5= Unknown
# 6= Voided trip
# 

# <br> trip_type- type of the trip. Contains values 1 & 2
# <br> congestion_surcharge- New York State has imposed a new Congestion surcharge for some vehicles that charge to transport people for trips that both begin and end in New York State and that begin in, end in, or pass through Manhattan south of (but not including) 96th Street.
# 
# These column details can be found at <a href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page" target="_blank">this link</a>

# In[10]:


#check for unique values of all columns.
df.nunique()


# **'trip_id' has 62495 unique values which represents each trip has unique identifier which is expected.
# 'VendorID' contains 2 unique values. From this, it can be assumed that there are two vendors.
# 
# **'lpep_pickup_datetime' and 'lpep_dropoff_datetime' are of data type datetime64[ns].
# New features from these can be created.**
# 

# In[11]:


#let us extract and create new features from datetime
df['pickup_day']=df['lpep_pickup_datetime'].dt.day_name()
df['dropoff_day']=df['lpep_dropoff_datetime'].dt.day_name()
df['pickup_day_no']=df['lpep_pickup_datetime'].dt.weekday
df['dropoff_day_no']=df['lpep_dropoff_datetime'].dt.weekday
df['pickup_hour']=df['lpep_pickup_datetime'].dt.hour
df['dropoff_hour']=df['lpep_dropoff_datetime'].dt.hour
df['pickup_month']=df['lpep_pickup_datetime'].dt.month
df['dropoff_month']=df['lpep_dropoff_datetime'].dt.month


# The following features have been created:
# ​
# * pickup_day and dropoff_day which will contain the name of the day on which the ride was taken.
# * pickup_day_no and dropoff_day_no which will contain the day number instead of characters with Monday=0 and Sunday=6.
# * pickup_hour and dropoff_hour with an hour of the day in the 24-hour format.
# * pickup_month and dropoff_month with month number with January=1 and December=12.
# ​
# <br> Now from 'lpep_pickup_datetime' and 'lpep_dropoff_datetime' a new feature **'trip_duration'** can be created for further analysis.

# In[12]:


#Create a new variable 'trip_duration'
df['trip_duration']= df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']


# In[13]:


df.trip_duration.value_counts()


# In[14]:


#Convert trip_duration into minutes
df['trip_duration']=df.trip_duration.dt.total_seconds()/60.0


# In[161]:


np.round(df.trip_duration.describe(),2)


# **The distribution of Pickup and Drop Off hours of the day**
# 
# The time part is represented by hours,minutes and seconds which is difficult for the analysis,
# <br>thus we divide the times into 4 time zones: morning (6 hrs to 10 hrs) , midday (10 hrs to 16 hrs) ,
# <br>evening (16 hrs to 22 hrs) and late night (22 hrs to 6 hrs)
# 
# 

# In[160]:


def time_of_day(x):
    if x>=datetime.time(6, 0, 1) and x <=datetime.time(10, 0, 0):
        return 'morning'
    elif x>=datetime.time(10, 0, 1) and x <=datetime.time(16, 0, 0):
        return 'midday'
    elif x>=datetime.time(16, 0, 1) and x <=datetime.time(22, 0, 0):
        return 'evening'
    elif x>=datetime.time(22, 0, 1) or x <=datetime.time(6, 0, 0):
        return 'late night'


# In[159]:


import datetime
df['pickup_time_of_day']=df['lpep_pickup_datetime'].apply(lambda x :time_of_day(datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time()) )
df['dropoff_time_of_day']=df['lpep_dropoff_datetime'].apply(lambda x :time_of_day(datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time()) )


# **Now from 'trip_distance' and 'trip_duration' a new feature 'vehicle_speed' can be created for further data analysis.**

# In[158]:


##Calculate vehicle_speed in miles/hr for further insights
df['vehicle_speed'] = (df.trip_distance/(df.trip_duration/3600))


# In[157]:


df.head()


# In[149]:


df['pickup_month'].value_counts()


# **'pickup_month' only contains values of the month January (1) and December(12).**

# In[150]:


df['dropoff_month'].value_counts()


# **'dropoff_month' only contains values of the month January (1), February(2) and December(12).** 
# <br> There might be a possibility that a trip started on the last day of January and ended in February.

# In[148]:


#Checking for wrong entries like symbols -,?,#,*,etc.
for col in df.columns:
    print('{} : {}'.format(col,df[col].unique()))


# **It can be noticed that there are no wrong entries like symbols -,?,#,*,etc. 
# <br>Also it is visible that some entries contain vavlues with 'NAN'. 
# <br>These values might have been replaced by 'NAN' for containing wrong entries.**

# In[41]:


df['passenger_count'].value_counts()


# **214 entries with 0 passenger count**. Most of the trips contains 1 passenger. These are clearly outliers. 
# 
# 

# In[135]:


#Let us remove the rows which have 0 or 7 or 9 passenger count.
df=df[df['passenger_count']!=0]
df=df[df['passenger_count']<=6]


# In[136]:


df['passenger_count'].value_counts()


# **Now, that seems like a fair distribution.**

# In[162]:


df.isnull().sum()


# **We can observe that now there are missing values in some columns.**

# **Visualizing the missing values**

# In[154]:


#Dropping null column
#df=df.drop('store_and_fwd_flag_NAN', axis=1)
#df=df.drop('RatecodeID_NAN', axis=1)
#df=df.drop('payment_type_NAN', axis=1)
#df=df.drop('trip_type_NAN', axis=1)
#df=df.drop('congestion_surcharge_NAN', axis=1)
df=df.drop('passenger_count_NAN', axis=1)


# In[155]:


sns.heatmap(df.isnull(),cbar=False,cmap='viridis')


# **Capturing NAN with a new feature**
# 
# Advantages:
# 
# It captures the importance of missing values which will help model understand the data better
# 
# Disadvantages:
# 
# Curse of Dimensionality
# 
# **Theses attributes were removed after carefully observing 'passenger_count'. After dropping the rows containing values 0, 7 or 9 these 'NAN' values were also removed.**

# In[156]:


plt.figure(figsize=(25,10))
sns.heatmap(df.corr(),cbar=True,annot=True,cmap='Blues')


# **Data Analysis**

# **Univariate Analysis**

# **Trip Duration**

# In[163]:


sns.histplot(df['trip_duration'],kde=False,bins=20)


# **The histogram is really skewed as we can see.**

# In[164]:


sns.boxplot(df['trip_duration'])


# **Let's visualize the number of trips taken in slabs of 0-10, 20-30 ... minutes respectively**

# In[165]:


df.trip_duration.groupby(pd.cut(df.trip_duration, np.arange(1,7200,600))).count().plot(kind='barh',figsize = (18,5))
plt.title('Trip Duration')
plt.xlabel('Trip Counts')
plt.ylabel('Trip Duration (seconds)')
plt.show()


# **I can observe that most of the trips took 0 - 10 mins to complete i.e. approx 600 secs. Some of the trips were 30 minutes long.**

# **VendorID**

# In[205]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,5))
ax = df['VendorID'].value_counts().plot(kind='bar',title="Vendors",ax=axes[0],color = ('blue',(1, 0.5, 0.13)))
df['VendorID'].value_counts().plot(kind='pie',title="Vendors",ax=axes[1])
ax.set_ylabel("Count")
ax.set_xlabel("Vendor Id")
fig.tight_layout()


# **Vendor 1 has more trips than Vendor 2 **

# **Trip Distance**

# In[187]:


plt.figure(figsize = (20,5))
sns.boxplot(df.trip_distance)
plt.show()


# In[206]:


print(f"There are {df.trip_distance[df.trip_distance == 0 ].count()} trip records with 0 miles distance")


# 3274 trips record with distance equal to 0. Below are some possible explanation for such records.
# <br> *Customer changed mind and cancelled the journey just after accepting it.
# <br> *Software didn't recorded dropoff location properly due to which dropoff location is the same as the pickup location.
# <br> *Issue with GPS tracker while the journey is being finished.
# <br> *Driver cancelled the trip just after accepting it due to some reason. So the trip couldn't start
# <br> *Or some other issue with the software itself which a technical guy can explain

# In[189]:


df.trip_distance.groupby(pd.cut(df.trip_distance, np.arange(0,80,10))).count().plot(kind='barh',figsize = (19,4))
plt.show()


# From the above observation it is evident that most of the rides are completed between 1-10 miles with some of the rides with distances between 10-20miles. 

# In[192]:


df_short = df[df.trip_distance <= 10].count()
df_long = df[df.trip_distance > 10].count()
print(f"Short Trips: {df_short[0]} records in total.\nLong Trips: {df_long[0]} records in total.")


# **Speed**

# In[193]:


plt.figure(figsize = (20,5))
sns.boxplot(df.vehicle_speed)
plt.show()


# In[171]:


df = df[df.vehicle_speed <= 65]
df.vehicle_speed.groupby(pd.cut(df.vehicle_speed, np.arange(0,65,10))).count().plot(kind = 'barh',figsize = (19,5))
plt.xlabel('Trip count')
plt.ylabel('Speed (mile/H)')
plt.title('Speed')
plt.show()


# **It is evident from this graph what we thought off i.e. most of the trips were done at a speed range of 10-20 miles/H.**

# **Passenger Count**

# In[194]:


df.passenger_count.value_counts()


# In[172]:


sns.distplot(df['passenger_count'],kde=False)
plt.title('Distribution of Passenger Count')
plt.show()


# **Most of the passengers travel solo**

# **Trips per Day**

# In[195]:


figure,ax=plt.subplots(nrows=2,ncols=1,figsize=(8,8))
sns.countplot(x='pickup_day',data=df,ax=ax[0])
ax[0].set_title('Number of Pickups done on each day of the week')
sns.countplot(x='dropoff_day',data=df,ax=ax[1])
ax[1].set_title('Number of dropoffs done on each day of the week')
plt.tight_layout()


# **hourwise pickup pattern across the week**

# In[82]:


n = sns.FacetGrid(df, col='pickup_day')
n.map(plt.hist, 'pickup_hour')
plt.show()


# <br> Early morning pickups seems consistently low
# <br>Taxi pickups seems to be consistent across the week at 15 Hours.

# In[196]:


figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
sns.countplot(x='pickup_time_of_day',data=df,ax=ax[0])
ax[0].set_title('The distribution of number of pickups on each part of the day')
sns.countplot(x='dropoff_time_of_day',data=df,ax=ax[1])
ax[1].set_title('The distribution of number of dropoffs on each part of the day')
plt.tight_layout()


# In[175]:


figure,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
df['pickup_hour']=df['lpep_pickup_datetime'].dt.hour
df.pickup_hour.hist(bins=24,ax=ax[0])
ax[0].set_title('Distribution of pickup hours')
df['dropoff_hour']=df['lpep_dropoff_datetime'].dt.hour
df.dropoff_hour.hist(bins=24,ax=ax[1])
ax[1].set_title('Distribution of dropoff hours')


# In[197]:


df['pickup_time_bins'] = pd.cut(x = df['pickup_hour'], bins=[1,6,9,12,15,18,21,24])
df['pickup_time_bins'].value_counts(normalize = True)


# In[198]:


sns.countplot(x = 'pickup_time_bins' , data = df)


# **Bivariate Analysis**

# In[178]:


sns.barplot(y='vehicle_speed',x='dropoff_time_of_day',data=df,estimator=np.mean)


# In[199]:


#passenger count per VendorID
group = df.groupby('VendorID').passenger_count.mean()
sns.barplot(group.index, group.values)
plt.ylabel('Passenger count')
plt.xlabel('Vendor Id')
plt.title('Passenger Count per Vendor')
plt.show()


# In[200]:


sns.catplot(x="store_and_fwd_flag", y="trip_duration",kind="strip",data=df)


# In[201]:


sns.catplot(y='trip_duration',x='passenger_count',data=df,kind="strip")


# In[107]:


#trip_duration per hour
sns.barplot(x='pickup_hour',y='trip_duration',data=df)


# In[109]:


#Trip Duration per time of day
sns.barplot(x='pickup_time_of_day',y='trip_duration',data=df)


# In[114]:


sns.barplot(x='pickup_day',y='trip_duration',data=df)


# In[202]:


#trip_distance per hour
plt.figure(figsize = (14,5))
group = df.groupby('pickup_hour').trip_distance.mean()
sns.pointplot(group.index, group.values)
plt.ylabel('Distance (mile)')
plt.title('Distance per Hour')
plt.show()


# In[182]:


##trip_distance per weekday
plt.figure(figsize = (14,5))
group = df.groupby('pickup_day').trip_distance.mean()
sns.pointplot(group.index, group.values)
plt.ylabel('Distance (mile)')
plt.title('Distance per WeekDay')
plt.show()


# In[183]:


#Average vehicle_speed per hour
plt.figure(figsize = (14,5))
group = df.groupby('pickup_hour').vehicle_speed.mean()
sns.pointplot(group.index, group.values)
plt.xlabel('Pick Up Hours')
plt.ylabel('Speed mile/h')
plt.title('Average Speed per Hour')
plt.show()


# In[ ]:




