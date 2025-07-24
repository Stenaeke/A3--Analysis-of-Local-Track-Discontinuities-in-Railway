#%%
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import pylab as py
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import svm


#%%
# Load the data from all three files.
df_trail1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
df_trail2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
df_trail3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')

df_list = [df_trail1, df_trail2, df_trail3]

#%%
# Remove the columns start_time, axle, cluster, tsne_1, and tsne_2 from the dataset.
for df in df_list:
    print(df.info())

for df in df_list:
    df.drop(['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2'], axis=1, inplace=True, errors='ignore')

for df in df_list:
    print(df.info())

#%%
# Combine the three datasets into a single unified dataset.
df_combined_trails = pd.concat([df_trail1, df_trail2, df_trail3], ignore_index=True)


#%%
# Replace all normal events with 0 and all other events with 1.

for value in df_combined_trails.index:
  if df_combined_trails.loc[value, "event"] == "normal":
    df_combined_trails.loc[value, "event"] = 0
  else:
     df_combined_trails.loc[value, "event"] = 1


#%%
# Normalize the dataset.

df_event_column = df_combined_trails["event"]
df_combined_trails.drop(["event"], axis=1, inplace=True)

scaler = Normalizer()
scaler.set_output(transform='pandas')
normalized_data = scaler.fit_transform(df_combined_trails)

normalized_data['event'] = df_event_column.values

#%%
# Split the data into training and testing sets in an 80/20 ratio.
trails_train, trails_test = train_test_split(normalized_data,test_size=0.20)

print(f"Train:\n{trails_train.shape}")
print(f"Test:\n{trails_test.shape}")

#%%
#Cross-Validation:
model = svm.SVC(kernel='linear')
X_train = trails_train.drop('event', axis=1)
y_train = trails_train['event'].values.astype(int)
scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())
print("Standard deviation of cross-validation scores:", scores.std())


#%%
#Comparison task:
model.fit(X_train, y_train)

X_test = trails_test.drop('event', axis=1)
y_test = trails_test['event'].values.astype(int)
y_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {test_accuracy:.3f}")


#%%
plt.figure(figsize=(8, 5))
plt.bar(['Cross-Validation', 'Test Set'], [scores.mean(), test_accuracy])
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.show()

#%%
with pd.ExcelWriter('combined_trails_normalized.xlsx') as writer:
    normalized_data.to_excel(writer)

