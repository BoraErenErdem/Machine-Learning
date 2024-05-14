

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential  # model oluşturur
from tensorflow.keras.layers import Dense  # katman oluşturur



df = pd.read_csv('Data/merc.csv')
print(df.head().to_string())


# region veri setindeki en pahalı arabaların %1'ini silip derin öğrenme ve regrasyon modellemesi yaparak arasındaki ilişkiyi bul
sns.pairplot(df)  # .pairplot() veri setini incelemek için görselleştirdim

df.describe()  # veri setini inceledim

sns.displot(df['price'])  # .displot() dağılım grafiği gösterir. Sütunların dağılım grafiğine tek tek bakman daha mantıklı çünkü bütün sütunlara bakmaya çalışınca görselleştirme optimal olmuyor.

sns.countplot(df)  # .countplot() kaç tane olduğunu gösterir

numeric_df = df.select_dtypes(include='number')  # df.corr() yapılamadı çünkü veri setinde string ifadeler de var o yüzden .select_dtypes(include='number') ifadesiyle sadece sayısal olan değerleri alıp öyle korelasyon ilişkisine baktık.
print(numeric_df.corr())

numeric_df.corr()['price'].sort_values()  # sadece price sütununun diğer sütunlarla ilişkisine baktım. ilişkisi yüksek olanlar price sütununu en çok etkileyenler, düşük olanlarsa en az etkileyenler

sns.scatterplot(x='mileage', y='price', data=df)  # .scatterplot() nokta nokta dağılımını gösterir. Burada mileage ile price'ın dağılımını gösterdi

# En yüksek fiyatlı ilk 20 arabayı bul
df.sort_values('price', ascending=False).head(20)

# Yaklaşım olarak veri setinde verinin %99'u alınırsa yani %1'i çıkartılırsa verinin yansıttığı genel tabloyu bozmadan işleme devam edebiliriz.
len(df)  # veri setinin sayısına baktım
len(df) * 0.01  # veri setinin %1'ini aldım ve bana 131 değerini verdi. Bu veri setinden 131 tane en pahalı arabayı çıkarırsam genel veriyi bozmadan işleme devam edebilirim.

yuzdedoksandokuz_df = df.sort_values('price', ascending=False).iloc[131:]  # veri setini fiyatına göre sıraladık ve iloc[131:] diyerek ilk 131 veriden sonrasını alıp yeni dataframe'e atadık.
print(yuzdedoksandokuz_df)  # Böylece ilk 131 en pahalı arabaları sildik.

sns.displot(yuzdedoksandokuz_df['price'])

# veri setini yıllara göre grupla ve ortalama hangi yılda arabalar kaç fiyata satılmış bul
df.groupby('year')[['price']].mean()  # sonuca bakılınca 1970'te satılan arabanın fiyatı geri kalan veri setiyle uyumsuz. Veri setinden çıkarıp çıkarmamak tercihe bağlı bu örnekte çıkar.

df = yuzdedoksandokuz_df

df[df['year'] != 1970].groupby('year')[['price']].mean()  # veri setinden 1970 yılındaki arabaların fiyat ortalamasını attık. Tamamen tercihe bağlıdır.
df.describe()
df = df[df['year'] != 1970]
df.groupby('year')[['price']].mean()


# veri setinde regrasyon yapabilmek için sayısal olmayan sütunu düşür (transmission)
df.head()

df = df.drop(['transmission'], axis=1)
print(df.head())


df.columns

X = df[['year','mileage', 'tax', 'mpg', 'engineSize']].values
print(X)

y = df['price'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()

model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# NOT #
# batch_size= kullanımını etkileyen faktörler:
# * Veri Seti Boyutu eğer küçükse küçük batch_size= kullanılabilir. Veri Seti Boyutu büyükse daha büyük batch_Size kulanılabilir.
# * Hız ve Performans olarak büyük batch_size= eğitim süresini hızlandırabilir. Küçük batch_size=  daha iyi genelleşme sağlayabilir. Yani hız ve performans arasında denge kurulmalı.

# epochs= kullanımı daha çok deneme yanılma yöntemiyle belirlenir. Ancak çok fazla epochs aşırı öğrenme (overfitting) riskini arttırabilir. Optimum değer için küçükten başlamak daha iyidir.

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=250, epochs=300)
# validation_data=(X_test, y_test): Bu parametre, eğitim sırasında modelin performansını değerlendirmek için ayrılmış doğrulama veri setini belirtir. Bu, modelin eğitim sırasında aşırı uyum olup olmadığını kontrol etmek için kullanılır.

# batch_size=250: Bu, her bir eğitim adımında kullanılacak örnek sayısını belirtir.


kayip_veri = pd.DataFrame(model.history.history)
print(kayip_veri.head())  # Burada hem normal kayıp (loss) hem de doğrulama kaybı (validation loss) görülür

kayip_veri.plot()  # loss ve validation loss grafiğini gösterir

tahmin_dizisi = model.predict(X_test)
print(tahmin_dizisi)

mean_absolute_error(y_test, tahmin_dizisi)  # output: 3239 yani şöyle yorumla gerçek veri ile arasında 3239 pound bir fark var. Bu veri setine göre bu sonuç göz ardı edilemez sapma var demek..!

plt.scatter(y_test, tahmin_dizisi)
plt.plot(y_test, y_test, color='green')  # veriyi görselleştirip yorumladık


df.iloc[2]  # bu kısımda rastgele bir arabayı aldık
yeni_araba_Series = df.drop(['price'], axis=1).iloc[2]  # aldığımız rastgele arabayı yeni bir değişkene atadık ve price sütununu düşürdük. Buradaki amacımız bakalım eğittiğimiz model doğru tahmin edebilecek mi onu göreceğiz.
print(yeni_araba_Series)

yeni_araba_Series = scaler.transform(yeni_araba_Series.values.reshape(-1,5)) #.values diyince 1D array olduğu için kabul etmedi o yüzden reshape(-1,5) yapıp -1 sabit değeri verip 5 sütun yap dedik.
model.predict(yeni_araba_Series)
# endregion




# region veri setinde bozuk verileri varsa onar yoksa da veri ön işlemesi yapıp derin öğrenme ile fiyat tahmin analizini yap
sns.pairplot(df)

df.columns

df.isnull().sum()

df['price'].value_counts()

numeric_df = df.select_dtypes(include='number')
print(numeric_df)

numeric_df.corr()

numeric_df.corr()['price'].sort_values()
numeric_df.corr()['engineSize'].sort_values()

sns.scatterplot(x='engineSize', y='price', data=df)

df.describe()

df.columns
X = df[['year','transmission', 'mileage', 'tax', 'mpg', 'engineSize']].values
print(X)

y = df['price'].values
print(y)


df.head()
df['transmission'].value_counts()

print(X[:,1])

le_transmission = preprocessing.LabelEncoder().fit(['Semi-Auto', 'Automatic', 'Manual', 'Other'])
X[:,1] = le_transmission.transform(X[:,1])
print(X)


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_test = minmax_scaler.transform(X_test)

derinogr_modeli = Sequential()
derinogr_modeli.add(Dense(11, activation='relu'))
derinogr_modeli.add(Dense(11, activation='relu'))
derinogr_modeli.add(Dense(11, activation='relu'))
derinogr_modeli.add(Dense(11, activation='relu'))
derinogr_modeli.add(Dense(10, activation='relu'))
derinogr_modeli.add(Dense(1))

derinogr_modeli.compile(optimizer='adam', loss='mse')


derinogr_modeli.fit(X_train, y_train, validation_data=(X_test, y_test,), batch_size=250, epochs=309)


loss_data_df = pd.DataFrame(derinogr_modeli.history.history)
print(loss_data_df.head())

loss_data_df.plot()

y_predict = derinogr_modeli.predict(X_test)
print(y_predict)

mean_absolute_error(y_test,y_predict)

plt.scatter(y_test, y_predict)
plt.plot(y_test, y_test, color='green')


# string ifade yer aldığı için transmission sütununun tamamına 0 yerleştirdim yani int değere dönüştürdüm ve işleme öyle devam ettim
for i in range(len(df)):
    if df['transmission'][i] == 'Automatic':
        df['transmission'][i] = 0

print(df)

df.iloc[20]
new_car_Series = df.drop(['price'], axis=1).iloc[20]
print(new_car_Series)

new_car_Series = minmax_scaler.transform(new_car_Series.values.reshape(-1,6))
derinogr_modeli.predict(new_car_Series)
# endregion