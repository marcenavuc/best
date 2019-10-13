import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('extended_table.csv')
df = df.dropna()
df.head()

X_train, X_valid, y_train, y_valid = train_test_split(df[df.columns[3:-2]].values, df['tariff_changes_count'].values, test_size=0.15)
model = LogisticRegression(n_jobs = -1).fit(X_train, y_train)
print(model.score(X_valid, y_valid))

df2 = pd.read_excel('Data_Motiv_2.xlsx')
df2 = df2.drop(columns = ['TARIFF_ID'])
df2 = df2.dropna()
X_test = df2[df2.columns[2:-1]]
y_pred = model.predict_proba(X_test)
print(y_pred)

df_solution = pd.DataFrame(data = y_pred)
df_solution.columns = ['0', '1', '2']
df_solution.to_csv('solution.csv')
