import os
import pandas as pd
import numpy as np
import scipy

# change cwd to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# assign files to objects
controversy = os.path.dirname(os.path.abspath(__file__))+'/79k controversy.csv'
absolutism = os.path.dirname(os.path.abspath(__file__))+'/79k absolutism.csv'

try:
    open(str(controversy))
    open(str(absolutism))
except:
    print('Problem opening one of the datasets.')
    exit()

df = pd.read_csv(str(controversy))
df = df[df['Label'].notnull()]
df = df[df['ColumnID'].notnull()]
df = df[df['Text'].notnull()]
df = df[df['Segment'].notnull()]
df = df[df['HighControversial'].notnull()]
df = df[df['MediumControversial'].notnull()]
df = df[df['LowControversial'].notnull()]

a = np.where(df.iloc[0:df.shape[0], 1].values == 1, True, False)
highControversy = df.iloc[0:df.shape[0], 5].values
print('Point-biserial correlation value for high-controversy and real/fake news: '+str(scipy.stats.pointbiserialr(a, highControversy).statistic))
mediumControversy = df.iloc[0:df.shape[0], 6].values
print('Point-biserial correlation value for medium-controversy and real/fake news: '+str(scipy.stats.pointbiserialr(a, mediumControversy).statistic))
lowControversy = df.iloc[0:df.shape[0], 7].values
print('Point-biserial correlation value for low-controversy and real/fake news: '+str(scipy.stats.pointbiserialr(a, lowControversy).statistic))
print('')

df = pd.read_csv(str(absolutism))
df = df[df['Label'].notnull()]
df = df[df['ColumnID'].notnull()]
df = df[df['Text'].notnull()]
df = df[df['Segment'].notnull()]
df = df[df['Absolutist'].notnull()]

a = np.where(df.iloc[0:df.shape[0], 1].values == 1, True, False)
absolutismValue = df.iloc[0:df.shape[0], 5].values
print('Point-biserial correlation value for absolutism and real/fake news: '+str(scipy.stats.pointbiserialr(a, absolutismValue).statistic))