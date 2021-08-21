#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession
import pytz
import matplotlib.pyplot as plt
from datetime import time
from datetime import datetime
from dateutil.parser import parse

# Any performance issues?
# ks.set_option("compute.ops_on_diff_frames", True)
ks.set_option('compute.default_index_type', 'distributed')


# In[2]:


def extract_year(s: str) -> int:
    return int(s.split(' ')[0].split('/')[-1])

def extract_day(s: str) -> int:
    return int(s.split(' ')[0].split('/')[1])

def extract_month(s: str) -> int:
    return int(s.split(' ')[0].split('/')[0])

def extract_date(s: str) -> str:
    return s.split(' ')[0]

def to_hour_simple(row) -> np.int32:
    return row.hour

def to_hour(row) -> np.int32:
    start = row['Start']
    end = row['End']
    if start.hour == end.hour:
        return start.hour
    else:
        return start.hour
    
def convert_ts(date_str) -> datetime:
    if date_str is not None:
        native = parse(date_str)
        local = pytz.timezone("America/New_York")
        local_dt = local.localize(native, is_dst=True)
        utc_dt = local_dt.astimezone(pytz.utc)
        return utc_dt
    else:
        return np.nan


# In[3]:


workers = 2
res = {
    'workers': 2,
}


# In[4]:


path = 's3a://oxclo-bucket-zhe/small.csv'
start_time = datetime.now()
res['start'] = start_time


# In[5]:


dkf = ks.read_csv(path)


# In[6]:


dkf = dkf[~(dkf['Trip Start Timestamp']=='Trip Start Timestamp')]
dkf['Year'] = dkf['Trip Start Timestamp'].apply(extract_year)
dkf['Month'] = dkf['Trip Start Timestamp'].apply(extract_month)
count = dkf['Trip ID'].count()


# In[7]:


res['count'] = count


# In[8]:


by_year = dkf.groupby(dkf['Year'])
year_records = by_year['Trip ID'].count()


# In[9]:


res['year_count'] = year_records.to_dict()


# In[10]:


dkf['AvgSpeed'] = dkf['Trip Miles'] / (dkf['Trip Seconds'] / 3600)
good_rec = dkf[~(((dkf['Trip Seconds'] < 60) & (dkf['AvgSpeed'] > 100)) | (dkf['Trip Miles'] > 1000) | (dkf['Fare'] > 2000))]
good_rec = good_rec[good_rec['Trip Miles'] > 0.0001]
good_rec = good_rec.spark.persist()
good_rec['Start'] = good_rec['Trip Start Timestamp'].apply(convert_ts)
good_rec['End'] = good_rec['Trip End Timestamp'].apply(convert_ts)
good_rec['Hour'] = good_rec['Start'].apply(to_hour_simple)
good_rec['Date'] = good_rec['Trip Start Timestamp'].apply(extract_date)
good_rec['Revenue'] = good_rec['Fare'] + good_rec['Tips']
g_count = good_rec.AvgSpeed.count()
good_rec = good_rec.spark.persist()


# In[11]:


res['good_rate'] = g_count / count


# In[12]:


data_2019 = good_rec[good_rec.Year == 2019]
avg_rev_2019 = data_2019.groupby(['Taxi ID', 'Date']).Revenue.sum().reset_index()
top_avg_rev_2019 = avg_rev_2019.groupby('Taxi ID').Revenue.mean()
top10_avg_rev_2019 = top_avg_rev_2019.nlargest(10)
avg_rev_2020 = good_rec[good_rec.Year == 2020].groupby(['Taxi ID', 'Date']).Revenue.sum().reset_index()
top2019_avg_rev_in_2020 = avg_rev_2020[avg_rev_2020['Taxi ID'].isin(top10_avg_rev_2019.index.to_numpy())].groupby('Taxi ID').mean()


# In[13]:


res['top10_avg_rev_2019'] = top10_avg_rev_2019.to_dict()
res['top2019_avg_rev_in_2020'] = top2019_avg_rev_in_2020.to_dict()


# In[14]:


best_in_2019 = data_2019.groupby('Taxi ID').Revenue.sum().nlargest(1)


# In[15]:


res['best_in_2019'] = best_in_2019.to_dict()


# In[16]:


mean = good_rec.groupby(['Year', 'Hour']).AvgSpeed.mean()
mean_pd = mean.unstack().to_pandas()


# In[17]:


fares = good_rec.groupby(['Year', 'Hour'])['Fare'].sum()
fares = fares.unstack().to_pandas()
res['fares'] = fares


# In[18]:


tips = good_rec.groupby(['Year', 'Hour'])['Tips'].sum()
tips = tips.unstack().to_pandas()
res['tips'] = tips


# In[19]:


good_rec['tip_pct'] = good_rec['Tips'] / good_rec['Trip Total']
tip_pct_avg = good_rec.groupby(['Year'])['tip_pct'].mean()
res['tip_pct_avg'] = tip_pct_avg.to_dict()


# In[20]:


good_rec['tip_per_dis'] = good_rec['Tips'] / good_rec['Trip Miles']
top10_tip_per_dis_2019 = good_rec[good_rec.Year==2019][['Trip ID', 'tip_per_dis']].nlargest(10, columns='tip_per_dis')
res['top10_tip_per_dis_2019'] = top10_tip_per_dis_2019


# In[22]:


group_2019 = good_rec[good_rec.Year == 2019]
group_2020 = good_rec[good_rec.Year == 2020]
tip_pct_by_month_2019 = group_2019.groupby('Month').tip_pct.mean().to_pandas()
tip_pct_by_month_2020 = group_2020.groupby('Month').tip_pct.mean().to_pandas()
df = pd.concat([tip_pct_by_month_2019.sort_index(), tip_pct_by_month_2020.sort_index()], axis=1, keys=['2019', '2020'])


# In[28]:


res['tips_by_year'] = df
end_time = datetime.now()
res['end'] = end_time


# In[32]:


print('Duration: ', end_time - start_time)
res['duration'] = end_time - start_time


# In[33]:


res