import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
import argparse



def read_table(name):
    if name[name.find('.') + 1:] == 'xlsx':
        table = pd.read_excel(name)
    else:
        table = pd.read_csv(name)
    table = table.dropna()
    table = table.reset_index(drop=True)
    return table


def extract_features(table, left_index, right_index):
    subs_ids = np.unique(table['SUBS_ID'])
    series_list = []
    for id in tqdm(subs_ids):
        df = table[table['SUBS_ID'] == id]
        for column in df.columns[left_index:right_index]:
            df[column + '_mean'] = df[column].mean()
            df[column + '_std'] = df[column].std()
            df[column + '_range'] = df[column].max() - df[column].min()
            df[column + '_median'] = df[column].median()
            df = df.drop(columns=[column])
        df = df.loc[df.index.tolist()[0]]
        series_list.append(df)
    balanced_df_with_features = pd.concat(series_list, axis=1)
    balanced_df_with_features = balanced_df_with_features.T.set_index('SUBS_ID')
    return balanced_df_with_features

df = read_table('Dannye_po_Data_Motiv.xlsx')
print('Loaded train table')
same_subs = []
for i in range(0, df.shape[0] - 2):
    if df['SUBS_ID'][i] == df['SUBS_ID'][i + 1] == df['SUBS_ID'][i + 2]:
        same_subs.append(i)
same_subs = np.array(same_subs)
prepared = df.loc[same_subs]
check_ids = []
for i in range(df['SUBS_ID'].values.shape[0]):
    if df['SUBS_ID'][i] in prepared['SUBS_ID'].values:
        check_ids.append(i)
check_ids = np.array(check_ids)
three_df = df.iloc[check_ids]
print('three_df')
subs_ids = np.unique(three_df['SUBS_ID'])
tariff_changed = []
for idx in subs_ids:
    if np.unique(three_df[three_df['SUBS_ID'] == idx]['TARIFF_ID']).shape[0] - 1 > 0:
        for i in range(3):
            tariff_changed.append(1)
    else:
        for i in range(3):
            tariff_changed.append(0)
tariff_changed = pd.Series(tariff_changed)
minimal_rows = np.min(tariff_changed.value_counts())

three_df['tariff_changed'] = tariff_changed
tariff_not_changed = three_df[three_df['tariff_changed'] == 0].head(minimal_rows)
tariff_changed = three_df[three_df['tariff_changed'] == 1].head(minimal_rows)
balanced_df = pd.concat([tariff_not_changed, tariff_changed])

balanced_df_with_features = extract_features(balanced_df, 3, -2).dropna()
balanced_df_with_features = balanced_df_with_features.reset_index(drop=True)
print('Prepared train table')
X = balanced_df_with_features[balanced_df_with_features.columns[4:]].values
y = balanced_df_with_features['tariff_changed'].values
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)
train_result = model.predict(X_test)
print("accuracy: %s" % accuracy_score(y_test, train_result))
print("recall: %s" % recall_score(y_test, train_result, average='macro'))
print("precision: %s" % precision_score(y_test, train_result, average='macro'))

test_table = pd.read_excel('Data_Motiv_2.xlsx')
test_table = test_table.drop(columns=['TARIFF_ID'])
test_table = test_table.dropna()

print(len(test_table.columns[2:-1]))
test_table = extract_features(test_table, 2, -1).dropna()
print('Prepared test table')
y_result = model.predict_proba(test_table[test_table.columns[1:]])

df_solution = pd.DataFrame(data=y_result)
df_solution.columns = ['0', '1']
df_solution.to_csv('solution.csv')
