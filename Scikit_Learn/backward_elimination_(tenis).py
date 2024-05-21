

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm



# region Veri setindeki nem oranını lineer regresyonla tahmin yap ve P-Values'e göre Backward Elimination yap
df = pd.read_csv('Data/tenis.csv')
print(df.head().to_string())

outlook = df['outlook'].values
print(outlook)

le_outlook = preprocessing.LabelEncoder().fit(['sunny','overcast','rainy'])
outlook = le_outlook.transform(outlook)
print(outlook)

windy = df['windy'].values
print(windy)

le_windy = preprocessing.LabelEncoder().fit(['False','True'])
windy = le_windy.transform(windy)
print(windy)

play = df['play'].values
print(play)

le_play = preprocessing.LabelEncoder().fit(['no','yes'])
play = le_play.transform(play)
print(play)


o = pd.DataFrame(data=outlook, columns=['Outlook'])
print(o)

w = pd.DataFrame(data=windy, columns=['Windy'])
print(w)

p = pd.DataFrame(data=play, columns=['Play'])
print(p)

s1 = pd.concat([w,p], axis=1)
print(s1)

s2 = pd.concat([o,s1],axis=1)
print(s2)

df_1 = pd.concat([df,s2], axis=1)
print(df_1)

df_1 = df_1.drop(['outlook','windy','play'], axis=1)
print(df_1)

new_df = df_1[['Outlook','temperature','humidity','Windy','Play']]
print(new_df)

X = new_df[['Outlook','temperature','Windy','Play']].values
print(X)

y = new_df['humidity'].values
print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


regression = linear_model.LinearRegression()
regression.fit(X_train, y_train)


y_predict = regression.predict(X_test)
print(y_predict)

# Backward Elimination işlemi için bağımlı değişken sütunu ekleme
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)  # X.shape[0] ile X'in satır sayısını alıyoruz ve 1 sütundan oluşan birler matrisi oluşturuyoruz
print(X)

X_liste = new_df[['Outlook','temperature','Windy','Play']].values
print(X_liste)
X_liste = np.array(X_liste, dtype=float)

# OLS modelini oluşturma
model = sm.OLS(y,X).fit()
print(model.summary())

# P değerlerine göre sütunları eleme işlemi
p_values = model.pvalues  # pvalues değerine bakar
threshold = 0.05  # yani Signification Level (SL)
while True:
    p_values = model.pvalues
    max_p_value = p_values.max()
    if max_p_value > threshold:
        max_p_value_index = p_values.argmax()
        X = np.delete(X, max_p_value_index, axis=1)
        model = sm.OLS(y, X).fit()
    else:
        break

print(model.summary())
# endregion