

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm  # statsmodel.api istatistiksel analizler, veri keşfi ve modelleme için kullanılan bir Python kütüphanesi. Regresyon modelleri, doğrusal modeller, karışık etkiler modeli, zaman serisi analizi, hipotez testleri gibi çeşitli istatistiksel modelleri oluşturmak ve analiz etmek için kullanılır.



# region veriyi ön işleyip
df = pd.read_csv('Data/veriler.csv')
print(df)

df.columns

ulke = df['ulke'].values
print(ulke)

cinsiyet = df['cinsiyet'].values
print(cinsiyet)
df['ulke'].value_counts()

le_ulke = preprocessing.LabelEncoder().fit(['tr','us','fr'])
ulke = le_ulke.transform(ulke)
print(ulke)


le_cinsiyet = preprocessing.LabelEncoder().fit(['e','k'])
cinsiyet = le_cinsiyet.transform(cinsiyet)
print(cinsiyet)

sonuc = pd.DataFrame(data=ulke, columns=['ulke'])
print(sonuc)

sonuc2 = pd.DataFrame(data=cinsiyet, columns=['cinsiyet'])
print(sonuc2)

s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

new_df = pd.concat([df,s], axis=1)
print(new_df)

new_df.columns = ['ulke_s','boy','kilo','yas','cinsiyet_s','ulke','cinsiyet']
print(new_df)

new_df = new_df.drop(['ulke_s', 'cinsiyet_s'], axis=1)
print(new_df)

new_df = new_df[['ulke','boy','kilo','yas','cinsiyet']]
print(new_df)


# cinsiyet için tahmin bulma
X = new_df[['ulke','boy','kilo','yas']].values
print(X)

y = new_df['cinsiyet'].values
print(y)


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)


regression = linear_model.LinearRegression()
regression.fit(X_train,y_train)

y_prediction = regression.predict(X_test)
print(y_prediction)


# boy için tahmin bulma
boy_df = new_df[['boy']]
print(boy_df)

boy_harici_df = new_df[['ulke','kilo','yas','cinsiyet']]
print(boy_harici_df)

X = boy_harici_df[['ulke','kilo','yas','cinsiyet']].values
print(X)

y = boy_df['boy'].values
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

regresyon = linear_model.LinearRegression()
regresyon.fit(X_train,y_train)

y_tahmin = regresyon.predict(X_test)
print(y_tahmin)


# geri eleme (Backward Elimination) # Burada amacımız hangi değişken sistemi (boy çıktısını (yani y)) daha fazla bozuyorsa yani hangi değişkenin P-Valuesi(P) Significance Level'den(SL) daha yüksekse onu eleyerek devam edicem. Bu yüzden değişkenlerin hangisinin daha fazla etkilediğini görebilmek için bir dizi üzerinden gideceğim.
X = np.append(arr=np.ones((22,1)).astype(int), values=boy_harici_df, axis=1)  # Her değişken için aslında her satır için 1 ekledik çünkü intercept veya bias terimi sabit bir katsayi eklemek gerekir. Bu şekilde bağımsız değişkenlerin (X) katsayıları ile birlikte sabit bir terim kullanarak hedef değişkenini (y) tahmin edilmesini sağlar. Genelde sabit değişken (bias yada intercept) için 1 kullanılır.
print(X)

X_listesi = boy_harici_df[['ulke', 'kilo', 'yas', 'cinsiyet']].values  # Bütün kolonları aldım çünkü ilerde eleme işlemi yaparken bazı kolonları çıkartacağım. Bir nevi ön hazırlık yaptım..!
print(X_listesi)
X_listesi = np.array(X_listesi,dtype=float)
model = sm.OLS(y,X_listesi).fit()  # sm.OLS() sistemin istatiksel değerlerini çıkartmayı sağlar.  # Burada y (yani boy) bağımlı değişkeni ile X_listesi yani bağımsız değişkenleri eğitiyoruz.
print(model.summary())


# Backward elimination kısmı için sm.OLS() ile en yüksek P-Values'i gördük ve şimdi modelden çıkarıyoruz
X_listesi = boy_harici_df[['kilo', 'yas', 'cinsiyet']].values
print(X_listesi)
X_listesi = np.array(X_listesi,dtype=float)
model = sm.OLS(y,X_listesi).fit()
print(model.summary())


X_listesi = boy_harici_df[['kilo', 'cinsiyet']].values
print(X_listesi)
X_listesi = np.array(X_listesi,dtype=float)
model = sm.OLS(y,X_listesi).fit()
print(model.summary())
# endregion