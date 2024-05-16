

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression  # linear_model'den LinearRegression'u çağırdık
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # SimpleImputer eksik değerleri doldurmak için kullanılan bir sınıftır. Bu sınıftan SimpleImputer() nesnesi alınır ve eksik veriyi frekans aralığıan ya da ortalamasına göre doldurulur.


df = pd.read_csv('Data/satislar.csv')
print(df.head().to_string())

# region hangi ayda ne kadar satış yapılmış bul ve gelecek aylarda ne kadar satılabilir normalizasyon yaparak tahmin et

df.columns

df['Aylar'].value_counts()

df['Satislar'].value_counts()


X = df[['Aylar']].values
print(X)

y = df['Satislar'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

scaler = preprocessing.StandardScaler()  # Burada bütün train ve test setlerini normalizasyon ile istediğim aralığa indirgedim. Çünkü verilerin boyutu çok dengesizdi aralarında ilişki yoktu.
X_train = scaler.fit_transform(X_train)  # Bu soruda normalizasyon yapılmasına gerek yok sadece pratik amaçlı yaptım. İleride çok kullanılacak.
X_test = scaler.fit_transform(X_test)

y_train = scaler.fit_transform(y_train.reshape(-1,1))  # y_train ve y_test dizisi 1D olduğu için .reshape(-1,1) kullanarak 2D dizeye dönüştürdük. Çünkü bizden 2D dize bekliyor..!
y_test = scaler.fit_transform(y_test.reshape(-1,1))

print(X_train)
print(X_test)
print(y_train)
print(y_test)


linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train,y_train)


y_prediction = linear_regression.predict(X_test)  # predict = tahmin demektir.
print(y_prediction)


print(f'r2 skoru: {r2_score(y_test, y_prediction)}')



plt.figure(figsize=(10,7))
plt.scatter(X_train,y_train,alpha=0.5, color='b')
plt.plot(X_test, linear_regression.predict(X_test), color='green')
plt.ylabel('Satışlar')
plt.xlabel('Aylar')
# endregion



# region hangi ayda ne kadar satış yapılmış bul ve gelecek aylarda ne kadar satılabilir tahmin et
df.columns

X = df[['Aylar']].values
print(X)

y = df['Satislar'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)

dogrusal_reg = linear_model.LinearRegression()
dogrusal_reg.fit(X_train, y_train)

y_predic = dogrusal_reg.predict(X_test)
print(y_predic)

plt.figure(figsize=(10,7))
plt.scatter(X_train,y_train, alpha=0.5, color='b')
plt.plot(X_test, dogrusal_reg.predict(X_test), color='green')
plt.ylabel('Satışlar')
plt.xlabel('Aylar')
# endregion