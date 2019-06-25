import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import datetime as dt



df_original = pd.read_csv('data.csv')
print(df_original.head())

from dateutil.parser import parse
df_original.date = df_original.date.apply(lambda x : parse(x))
df_original.week = df_original.date.apply(lambda x : x.strftime("week_%A"))
df_original.hour = df_original.date.apply(lambda x : x.strftime("%H"))
df_original.minute = df_original.date.apply(lambda x: x.strftime("%M"))
df_original.hour = df_original.hour.astype('int')
df_original.minute = df_original.minute.astype('int')



range0 = []
range30 = []
for i in range(0,30,1) :
    range30.append(i)
    range0.append(0)

range1 = []
range60 = []
for i in range(30,60,1) :
    range60.append(i)
    range1.append(1)

df_original.minute = df_original.minute.replace(range30, range0)
df_original.minute = df_original.minute.replace(range60, range1)

df_original.hour_half = df_original.loc[df_original.minute == 1].hour.apply(lambda a : a + 0.5)
df_original.hour_half.loc[df_original.minute == 0] = df_original.loc[df_original.minute == 0 ].hour.apply(lambda x : x)




import random
df_original['count'] = df_original['count'].apply(lambda x  : random.randint(1, 4))
df_original.head()




import seaborn as sns
import matplotlib.pyplot as plt


figure = plt.figure(figsize = [15, 10])
sns.countplot(df_original.hour)
plt.show()



# make a dummy variables
df_original_with_dummies = pd.concat([df_original, pd.get_dummies(df_original.week)], axis = 1)
df_original_with_dummies = pd.concat([df_original_with_dummies, pd.get_dummies(df_original.hour_half)], axis= 1)
