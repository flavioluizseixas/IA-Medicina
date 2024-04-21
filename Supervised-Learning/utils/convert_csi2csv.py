import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
lenses = fetch_ucirepo(id=58) 
  
# data (as pandas dataframes) 
X = lenses.data.features 
y = lenses.data.targets 
  
# metadata 
print(lenses.metadata) 
  
# variable information 
print(lenses.variables) 

All_data = pd.concat([X, y], axis=1)
All_data.to_csv('data/csv/lenses.csv')
All_data.to_excel('data/xlsx/lenses.xlsx')
