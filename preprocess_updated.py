import os
import pandas as pd
import unicodedata

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

# change cwd to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# assign files to objects
file79kFAKE = os.path.dirname(os.path.abspath(__file__))+'/DataSet_Misinfo_FAKE.csv'
file79kTRUE = os.path.dirname(os.path.abspath(__file__))+'/DataSet_Misinfo_TRUE.csv'

try:
    open(str(file79kFAKE))
    open(str(file79kTRUE))
    
except:
    print('Problem opening one of the datasets.')
    exit()

# Read file containing fake instances from 
# 'Misinformation & Fake News text dataset 79k'
# Create/add a column called 'Label'
# Assign 0 under 'Label' column for all instances
# 0 indicates that the instance is fake
# Assigning 0 to fake instances matches with 'WELFake Dataset' assignment
data1 = pd.read_csv(str(file79kFAKE), header=None)
data1 = data1.iloc[1:]
data1 = data1.iloc[:,1:]
data1['Label'] = 0
data1.columns.values[0] = "Text"
data1 = data1[data1['Text'].notnull()]


# Read file containing true instances from
# 'Misinformation & Fake News text dataset 79k'
# Create/add a column called 'Label'
# Assign 1 under 'Label' column for all instances
# 1 indicates that the instance is true
# Assigning 1 to true instances matches with 'WELFake Dataset' assignment
data2 = pd.read_csv(str(file79kTRUE), header=None)
data2 = data2.iloc[1:]
data2 = data2.iloc[:,1:]
data2['Label'] = 1
data2.columns.values[0] = "Text"
data2 = data2[data2['Text'].notnull()]

# combine the fake and true instances together in one large dataframe
combinedData = pd.concat([data1,data2]).sample(frac=1,random_state=1,ignore_index=True)
print("Removing control characters from 'Misinformation & Fake News text dataset 79k' dataset. May take some time.")
i=0
while(i<len(combinedData.iloc[0:combinedData.shape[0],0].values)):
    combinedData.iloc[0:combinedData.shape[0],0].values[i]=remove_control_characters(combinedData.iloc[0:combinedData.shape[0],0].values[i])
    i=i+1
print(combinedData)
print(combinedData.dtypes)
print(combinedData.columns)

# Export data into csvs
print("Exporting '79k_Combined.csv'")
combinedData.to_csv( '79k_Combined.csv', header=True, index=False, encoding='utf-8')