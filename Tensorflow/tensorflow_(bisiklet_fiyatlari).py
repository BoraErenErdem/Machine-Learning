
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # Ölçeklendirme işlemini yapmak için gerekli olan sınıf, genellikle [0, 1] veya [-1, 1] ölçeklendirilmesini sağlar.
import tensorflow as tf
from tensorflow.keras.models import Sequential  # Burada modeli oluşturduk
from tensorflow.keras.layers import Dense  # Burada model içersine katmanları koyduk
from sklearn.metrics import mean_absolute_error  #  gerçek değerlerle tahmin edilen değerler arasındaki ortalama mutlak farkı ölçer. Değeri sıfıra ne kadar yakınsa, modelin tahminleri gerçek değerlere o kadar yakındır. Daha düşük bir MAE, daha iyi bir model performansını gösterir.
import seaborn as sns
from sklearn.metrics import mean_squared_error  # gerçek değerlerle tahmin edilen değerler arasındaki ortalama karesel farkı ölçer. Değeri sıfıra ne kadar yakınsa, modelin tahminleri gerçek değerlere o kadar yakındır.
from tensorflow.keras.models import load_model  # bu yaptığım modeli kaydedip sonra tekrar kullanmama olanak sağlar




# veriyi dataframe'e çevirip görselleştirme
df = pd.read_csv('Data/bisiklet_fiyatlari.csv')
print(df.head().to_string())

df.dtypes

sns.pairplot(df)  # tüm verilerin scatter plot gibi grafiklerini çıkardı


# veriyi test_train olarak bölme
df.columns

X = df[['BisikletOzellik1', 'BisikletOzellik2']].values
print(X)

y = df['Fiyat'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=15)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# scaling (ölçeklendirme): Buradaki amaç nöronlara vereceğimiz verinin boyutunu küçültmek ve böylece daha hızlı işlem yapabilelim. Aslında Normalizasyon yaptık..!
scaler = MinMaxScaler()  # MinMaxScaler() ile yaptık
scaler.fit(X_train)
print(X)  # Bu kısım yani fit ve transform işlemlerini ayrı ayrı yapılması farklı veri setleri üzerinde ölçekleme yapabilir ve uygun ölçeklendirme parametrelerini (minimum ve maksimum değerler) yeniden kullanmamızı sağlar.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)



# Modeli oluşturma ve modele katmanlar ekleme
model = Sequential()  # Modeli oluşturduk

model.add(Dense(4, activation='relu'))  # Bu kısımda katmanları (Hidden Layers) ekledik. Bu katmanları istediğim kadar ekleyebilirim.
# Dense(4,) kısmı kaç tane nöron olacağını yazdık
# , activation='relu')) kısmı ise ReLU (derin öğrenmede sıklıla kullanılan fonksiyon) yazdık.
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(1))  # Bu kısım output yani çıktı kısmıdır. Genelde sadece 1 tane nöron vermek yeterli olur.

model.compile(optimizer='rmsprop', loss='mse')  # Bu kısımda modeli compile() ile birleştiriyoruz ve içine optimizasyon algoritmasını yazıyoruz.
# optimizer= optimizasyon fonksiyonlarını buraya yazarız. Normalde en iyi sonuç veren Adam fonksiyonu olsa da burada görebilelim diye rmsprop kullandık.
# loss= Burada loss parametresi, modelin eğitim sırasında kullanacağı kayıp (loss) fonksiyonunu belirtir. Kayıp fonksiyonu, gerçek etiketlerle modelin tahmin ettiği değerler arasındaki farkı ölçer. Bu farkın küçük olması, modelin daha iyi bir performans sergilediği anlamına gelir burada kullandığımız mse(Mean Square Error) scikt learnde gördüğümüz özelliği kullanırız.


# Modeli Eğitme (Train)

# epochs= "Epok" (epoch), bir makine öğrenimi modelinin tüm eğitim veri setini bir kez geçtiği zaman dilimini ifade eder. Eğitim sırasında, model bir dizi adımdan (iteration) oluşur ve her adımda (iteration) bir mini-batch (küçük bir veri alt kümesi) kullanılarak parametreler güncellenir. Bir epok, modelin tüm eğitim veri setini bir kez gördüğü noktadır.

# Örneğin, eğitim veri seti 1000 örneğe sahipse ve mini-batch boyutu 100 ise, bir epokta model toplamda 10 adımdan (iteration) oluşur (1000 / 100 = 10). Her adımda, model bir mini-batch kullanarak eğitim veri setinin bir parçasını işler.

# Modelin eğitimi genellikle birden çok epok boyunca gerçekleştirilir. Birden çok epok kullanılması, modelin daha fazla veriyle eğitilmesini sağlar ve genellikle daha iyi bir performans elde edilmesine yardımcı olur. Ancak, çok fazla epok kullanmak aşırı uydurma (overfitting) riskini artırabilir, bu nedenle epok sayısı dikkatlice seçilmelidir..!

model.fit(X_train, y_train, epochs=250)

loss = model.history.history['loss']  # loss fonksiyonunun değerlerini verir # Bilerek ['loss'] diye belirttik çünkü bize dictionary döner ve biz dictionaryden bunu dizeye çevirip böyle ifade edebiliriz.
print(loss)


# loss fonksiyonunun değerlerini görselleştirme (model.history.history görselleştirme)
sns.lineplot(x=range(len(loss)), y=loss)  # x= x ekseninde loss içerisinde kaç tane veri var onu göstermesi için range(len(loss)) yaptık.
# y= y ekseninde loss'un kendisini görmek istedik.
# sns.lineplot() çizgi şeklinde göstermesi için kullandık.

# loss değerlerini bulma
train_loss = model.evaluate(X_train,y_train)  # train'deki loss değerini verir
test_loss = model.evaluate(X_test,y_test)  # test'deki loss değerini verir
# model.evaluate yöntemi, eğitilmiş bir makine öğrenimi modelinin performansını değerlendirmek için kullanılır. Bu yöntem, modelin belirli bir test veri kümesi üzerindeki performansını ölçer ve genellikle doğruluk, kayıp veya başka bir performans metriği sağlar.
print(train_loss)
print(test_loss)


# Modeli Değerlendirme
test_tahminleri = model.predict(X_test)
print(test_tahminleri)


tahmin_df = pd.DataFrame(y_test,columns=['Gerçek Y'])
print(tahmin_df)

test_tahminleri.shape
test_tahminleri = pd.Series(test_tahminleri.reshape(330,))  # test tahminlerini Series'e dönüştürdük çünkü tahmin_df'nin içerisine yerleştirmek istiyoruz. # reshape(330,) ile index olamdan konumlandırmak istediğimiz için böyle yaptık.
print(test_tahminleri)


# concat metodu ile tahmin_df ve test_tahminleri'ni birleştirelim
tahmin_df = pd.concat([tahmin_df,test_tahminleri],axis=1)
print(tahmin_df)

# tahmindf sütununda yazan 0'ın adını değiştirme işlemi
tahmin_df.columns = ['Gerçek Y', 'Tahmin Y']
print(tahmin_df)

# Görselleştirme
sns.scatterplot(x='Gerçek Y', y='Tahmin Y', data=tahmin_df)

mean_absolute_error(tahmin_df['Gerçek Y'], tahmin_df['Tahmin Y'])  # Output: 7.304537427869318 hata değeri. Bunu şöyle yorumla: Ortalama fiyatı 872tl olan bir üründen 7tl sapma payın olursa sorun olmaz. Bu ver iseti için geçerlidir başka veri setinde 7'lik bir sapma çok iyi veya çok kötü oalrak değerlendirilebilir.

df.describe()


# Yeni bir veri geldiğinde tahminimiz ne kadar başarılı değerlendirelim
yeni_bisiklet_ozellikleri = [[1751,1750]]

yeni_bisiklet_ozellikleri = scaler.transform(yeni_bisiklet_ozellikleri)

model.predict(yeni_bisiklet_ozellikleri)


# Modeli Kayıt Etme
model.save('bisiklet_modeli.h5')

# Modeli Çağırma
bisiklet_modeli = load_model('bisiklet_modeli.h5')

# Yukarıdaki Modeli Kayıt Etme ve Modeli Çağırma işlemleri artık eski kaldığı için yeni sürümlerde farklı yollardan çağırılıyor. Eski model artık kullanışsız.

# NOT: Tensorflow'da model kurarken önce ihtiyaçların belirlenmesi ve istenilen sonucun kurgulanması çok önemli.