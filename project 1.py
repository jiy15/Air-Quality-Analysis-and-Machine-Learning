import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:\Users\HP\Downloads\air_quality.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())


print(df.isnull().sum())
df = df.drop_duplicates()
# Fill missing numerical values
df.fillna(df.median(numeric_only=True), inplace=True)
# Remove rows where pollutant_id is missing
df = df.dropna(subset=['pollutant_id'])


#simple linear regression
X = df[['pollutant_max']]
y = df['pollutant_avg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.title("Simple Linear Regression")
plt.show()



#multiple linear regression
X = df[['pollutant_min', 'pollutant_max','latitude', 'longitude']]
y = df['pollutant_avg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))




#polynomial regression

X = df[['pollutant_max']].copy()
X['pollutant_max_square'] = X['pollutant_max'] ** 2
y = df['pollutant_avg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))



#knn
X = df[['pollutant_min', 'pollutant_max','pollutant_avg', 'latitude', 'longitude']]
y = df['pollutant_id'].astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mean = X_train.mean()
std = X_train.std().replace(0, 1)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("KNN Confusion Matrix")
plt.show()



#decision tree
X = df[['pollutant_min', 'pollutant_max','pollutant_avg', 'latitude', 'longitude']]
y = df['pollutant_id'].astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.show()

plt.figure(figsize=(15, 8))
plot_tree(model, filled=True,feature_names=X.columns,class_names=[str(i) for i in sorted(y.unique())])
plt.show()




















