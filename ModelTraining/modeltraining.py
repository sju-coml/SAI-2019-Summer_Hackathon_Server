import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import datetime as dt


df_real_original = pd.read_csv('data.csv')

#-----------------10%샘플 데이터 기준시
#df_original = df_real_original.sample(frac=0.1)
#-----------------10%샘플 데이터 기준시


#-----------------전체 데이터 기준시
df_original = df_real_original
#-----------------전체 데이터 기준시

df_original = df_original.sort_index()
df_original.head()



from dateutil.parser import parse
df_original.date = df_original.date.apply(lambda x : parse(x))
df_original.head()


df_original.week = df_original.date.apply(lambda x : x.strftime("week_%A"))
df_original.head()



df_original.hour = df_original.date.apply(lambda x : x.strftime("%H"))
df_original.minute = df_original.date.apply(lambda x: x.strftime("%M"))
df_original.head()



df_original.hour = df_original.hour.astype('int')
df_original.minute=  df_original.minute.astype('int')

range0 = []
range1 = []

range30 = []
for i in range(0,30,1) :
    range30.append(i)
    range0.append(0)

range60 = []
for i in range(30,60,1) :
    range60.append(i)
    range1.append(1)

df_original.minute = df_original.minute.replace(range30, range0)
df_original.minute = df_original.minute.replace(range60, range1)
df_original.head()



df_original.hour_half = df_original.loc[df_original.minute == 1].hour.apply(lambda a : a + 0.5)
df_original.hour_half.loc[df_original.minute == 0] = df_original.loc[df_original.minute == 0 ].hour.apply(lambda x : x)
df_original.head(10)




import random
df_original['count'] = df_original['count'].apply(lambda x  : random.randint(1, 4))
df_original.head()




df_original['date'] = pd.to_datetime(df_original['date']).values.astype('datetime64[D]')
df_original['month'] = df_original['date'].astype('str').str.slice(5, 7)
df_original['day'] = df_original['date'].astype('str').str.slice(8, 10)




df_original['groupbycolumn'] = df_original.date.apply(lambda x : str(x.strftime("%Y%m%d")))
df_original['groupbycolumn'] =  df_original.groupbycolumn.apply(lambda x : str(x)) + df_original.hour_half.apply(lambda x : str(x))
df_original.head()



df_groupedcolumn = df_original.groupby('groupbycolumn')['count'].sum()
df_original = df_original.drop_duplicates('groupbycolumn')
df_original = pd.merge(df_original, df_groupedcolumn, left_on = 'groupbycolumn', right_index = True)

df_original = df_original.drop(['count_x', 'groupbycolumn'] , axis = 1)
df_original.head()

df_original_with_dummies = pd.concat([df_original, pd.get_dummies(df_original.week)], axis = 1)
df_original_with_dummies = pd.concat([df_original_with_dummies, pd.get_dummies(df_original.hour_half)], axis= 1)
df_original_with_dummies.to_csv('1st_preprocessed_full_data.csv', index = False)