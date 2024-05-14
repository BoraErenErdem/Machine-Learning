

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout  # Dropout aşırı uydurmayı azaltmaya yardımcı olan bir düzenleme tekniği olan "dropout" uygular. Dropout, eğitim sırasında belirli bir olasılıkla rastgele bir kısmı nöronların çıkışını sıfıra ayarlar. Bu, ağın genelleştirme yeteneğini artırır ve aşırı uydurmayı azaltır.

from tensorflow.keras.callbacks import EarlyStopping  # EarlyStopping geri çağrı, model eğitimi sırasında belirli bir metriğin (genellikle kayıp) iyileşme durduğunda eğitimi durdurur. Bu, aşırı uydurmayı önlemeye ve eğitim süresini optimize etmeye yardımcı olur. Erken durdurma, eğitim sürecini otomatik olarak sonlandırarak gereksiz hesaplama maliyetini ve kaynak kullanımını azaltabilir. EarlyStopping geri çağrısını kullanarak, eğitim sırasında modeli belirli bir performans metriği üzerinde izleyebilir ve gerekli durumlarda eğitimi durdurabilirsin. Model daha iyi genelleştirmeye ve aşırı uyumu azaltmaya yardımcı olabilir.

from sklearn.metrics import classification_report  # sınıflandırma modelinin performansını anlamak için kullanılır ve bu performansı sınıf bazında detaylı bir şekilde raporlar. doğruluk, hassasiyet, geri çağırma ve F1 skoru gibi metrikleri hesaplar ve bu metrikleri sınıf bazında raporlar.argüman olarak y_test (y_gerçek) ve y_prediction (y_tahmini) alır.

from sklearn.metrics import confusion_matrix  # sınıflandırma modelinin performansını değerlendirmek için kullanılan bir metriktir. Bu işlev, gerçek ve tahmin edilen sınıflar arasındaki kafa karışıklığını belirlemek için kullanılır. argüman olarak y_test (y_gerçek) ve y_prediction (y_tahmini) alır. Karışıklık matrisi, modelin gerçek ve tahmin edilen sınıflar arasındaki kafa karışıklığını görselleştirir.


# Kısaca classification_report modelin sınıflandırma performansını metriklerle raporlarken, confusion_matrix modelin her sınıf için doğru ve yanlış sınıflandırmalarını bir matris şeklinde görselleştirir.





df = pd.read_csv('Data/maliciousornot.csv')
print(df.head().to_string())

df.info()
df.describe()

df.corr()['Type'].sort_values()  # Type'ına göre arasındaki korelasyona (ilşkisine) baktım. Yani hangi sütun Type'ı en çoktan en aza doğru etkiliyor ona baktım.

sns.countplot(x='Type', data=df)

df.corr()['Type'].sort_values().plot(kind='bar')


df.columns
X = df[['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS',
        'TCP_CONVERSATION_EXCHANGE', 'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS',
        'APP_BYTES', 'SOURCE_APP_PACKETS', 'REMOTE_APP_PACKETS',
        'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES', 'APP_PACKETS',
        'DNS_QUERY_TIMES', 'SOURCE_A', 'SOURCE_B', 'SOURCE_C', 'SOURCE_D',
        'SOURCE_F', 'SOURCE_E', 'SOURCE_G', 'SOURCE_H', 'SOURCE_I', 'SOURCE_J',
        'SOURCE_K', 'SOURCE_M', 'SOURCE_L', 'SOURCE_N', 'SOURCE_O', 'SOURCE_P',
        'SOURCE_R', 'SOURCE_S']].values
print(X)

y = df['Type'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=15)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)


# 1.ÖRNEK: Bu kısımda overfitting'i gördük ve overfitting durumunda validation_loss ve loss arasında neler oluyor onu gözlemledik. Bu modeli kullanmıcaz sadece görmek ve mantığını anlamak amaçlı yaptık.
model = Sequential()

model.add(Dense(30, activation='relu'))  # Burada 30 dememizin sebebi aslında normalde veri setinde kaç tane column(sütun) varsa giriş layer'ına yani giriş ağında (giriş Dense) o kadar yazılması önerilir. Veri setimde 30 tane column olduğu için ona eşit şekilde olsun diye 30 nöron verdim.. Aynı kuramın devamı ilk baştakine sütun sayısı kadar verildiğinde geriye kalan layerslar yani katmanlara sütun sayısı kadar verdiğin nöron ile 1 arasında olması mantıklı olur deniyor.
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # sigmoid fonksiyonu 0 ve 1 verdiği için genelde ikili sınıflandırma problemlerinde mantıklı. Ancak çok sınıflı sınıflandırma problemleri için, son katmanda "softmax" aktivasyon fonksiyonu daha yaygın olarak kullanılır..!

model.compile(optimizer='adam', loss='binary_crossentropy')  # ikili sınıflandırma yapacağımız için loss kısmına binary_crossentropy yazdık.


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=700)

model_kayip_df = pd.DataFrame(model.history.history)
print(model_kayip_df)

model_kayip_df.plot()



# 2.ÖRNEK: Bu kısımda early_stopping değişkenine EarlyStopping() nesnesini atıyoruz. Burada amaç biz epochs'u çok versek bile aslında EarlyStopping() nesnesi val_loss'u takip ederek duruma göre overfitting olacağı epochs'u durduruyor ve overfitting olmsmsdını sağlıyor.
model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=25)  # monitor= neyi takip edip izlemesini istiyorsak onu belirityoruz. Burada val_loss'u takip etmesini istedik.
# patience= epochs'lara bakar ve kaç epochs sonrası modelde iyileştirme olmazsa durduracağını gösterir. (15 ile 25 arası ideal)
# mode= Burada val_loss'u minimumda tutmaya çalıştığımız için "min" dedik. Ancak başka parametre takip etseydik içine "max" ya da "auto" yazabilirdik. Duruma göre değişebilir.

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=700, callbacks=[early_stopping])  # callbacks= oluşturduğum early_stopping değişkenini buraya yazmam gerekiyor. Bizden liste istediği için liste içinde vermemiz gerekir.

model_kayip_df = pd.DataFrame(model.history.history)
print(model_kayip_df)

model_kayip_df.plot()


# 3.ÖRNEK: Bu kısımda Dropout() ile yüzde kaçında nöronları turn-off (açıp kapatacağımız) yapacağımızı belirledik. İlk input katmanından sonra son katmana kadar ekledik. Son katmana Dropout() vermemize gerek yok. Dikkat edilmesi gereken Dropout() kısmına 0.5'ten fazla değer veririsen seni uyarır. Çünkü 0.5'ten yukarısını vermek genellikle öğrenim açısından riskli olabilir.
model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=25)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=700, callbacks=[early_stopping])

model_kayip_df = pd.DataFrame(model.history.history)
print(model_kayip_df)

model_kayip_df.plot()

tahminler = np.round(model.predict(X_test)).astype(int)  # np.round() modelin tahminlerini yuvarlar. 0.5'ten büyükse 1, küçükse 0 olarak yuvarlar.
print(tahminler)  # astype() yuvarlanmış tahminleri hangi türe dönüştürmek istersek ona dönüştürür. integer'a dönüştürmesini istediğimiz için int yazdık.


# sınıflandırma değerleri ve sınıflandırma tahminleri
print(classification_report(y_test, tahminler))  # precision: ne kadar doğru tahmin etmiş  #

print(confusion_matrix(y_test, tahminler))      # [[85 6] bu ifade de 6 tane yanlış bulmuş geri kalanları doğru bulmuş demek. Bu ver setine göre oldukça iyi.
                                                #  [12 62]]