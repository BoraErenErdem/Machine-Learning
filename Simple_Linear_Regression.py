

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from math import floor
from sklearn.metrics import mean_squared_error


df = pd.read_csv('Data/FuelConsumption.csv')
print(df.head().to_string())


# NOT: MSE modelin tahminlerinin doğruluğunu ölçerken, R-kare skoru bağımsız değişkenlerin bağımlı değişkendeki varyansı ne kadar açıkladığını ölçer.


# region ENGINESIZE ile CO2EMISSIONS arasındaki ilişkiyi bul
# Simple Linear Regression yapacağımız için sadece bir tane bağımsız değişken kullanacağız. Bu yüzden tahmin edeceğimiz attribute ile bağımsız değişken içeren bir dataframe hazırlayacağız. Simple Linear Regression'da diğer alanlara ihtiyacımız yok..!

cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]  # CO2EMISSIONS sadece ENGINESIZE'a bağlı. Yani bağımsız değişken ENGINESIZE, bağımlı değişken CO2EMISSIONS
print(cdf.head().to_string())


# Ana veri setimizi train ve test olmak üzere 2 ayrı veri setine böleriz.

# Ana veri setimizin %80'lik kısmını train için, geriye kalan %20'lik kısmını test için böldük. Bunun sebebi ML algoritmamızı ne kadar çok veriyle train edersek o kadar başarılı sonuç alırız.

# Veri setini bölerken hem train hem test setinin homojen olmasına özen göster. Homojen olursa daha sağlıklı ve doğru çalışır. Ayrıca bu bölme işlemi tüm ML algoritmalarında kullanılır.


msk = np.random.rand(len(df)) <= 0.8
print(msk)

train_df = cdf[msk]  # True ve Flase olan değerleri gösterdi
test_df = cdf[~msk]  # True olan değerleri False'a çevirip gösterdi

print(f'Train Set Satırı: {train_df.shape}')
print(f'Test Set Satırı: {test_df.shape}')


regression = linear_model.LinearRegression()  # regression linear_model'i LinearRegression() nesnesine dönüştürdük.

train_x = np.asarray(train_df[['ENGINESIZE']])  # Burada nesneyi numpy dizisine dönüştürmek için np.asarray kullandık çünkü linear_model.LinearRegression() sınıfını numpy dizisi ile kullanılmalı.
train_y = np.asarray(train_df[['CO2EMISSIONS']])  # Burada da aynı işlemi yaptık. Artık train_x ve train_y numpy dizilerini temsil eder ve daha rahat işlem yapmamızı sağlar.

regression.fit(train_x,train_y)  # fit() fonksiyonu regrasyondaki teta0 ve teta1 i verir yani matematiksel işlemi yapar. Yani tetaya dönüştürür.

print(f'Coefficient %.2f' % regression.coef_[0][0])  # Bunun nedeni, kütüphanenin genel işleyiş tarzıdır. Sklearn kütüphanesi, katsayıları (coef_) daima 2D bir dizi olarak döndürür. Ayrıca coef katsayısı liste içinde liste döndürdüğü için önce listeyi seçtik sonra indeksi seçtik. (type = np.array)

print(f'Intercept %.2f' % regression.intercept_[0])  # intercept kesişim gösterir tek liste olduğu için listeyi seçtik teta0 burada intercept, teta1 burada coef.
# %.2f ifadesi noktadan sonra şu kadar sayı yazdır anlamına gelir


engine_size = float(input('Lütfen Motor Hacmini Giriniz: '))
y = regression.intercept_[0] + regression.coef_[0][0] * engine_size  # lineer regresyon formülünü yaptık. y = regression.intercept_[0] (teta0) + regression.coef_[0][0] * engine_size (teta1 * x)

print(f'Carbon Emisyonu: {floor(y)}')


# Motor Hacmi ve CO2 Emisyonu arasındaki ilişkiyi görselleştirdik
plt.figure(figsize=(10,7))
plt.scatter(train_df['ENGINESIZE'], train_df['CO2EMISSIONS'], alpha=0.50, color='r')
plt.plot(train_x, regression.coef_[0][0] * train_x + regression.intercept_[0], color='b')
plt.suptitle('Engine Size ve CO2 Arasındaki İlişki', color='b')
plt.ylabel('Emisyon', color='r')
plt.xlabel('Engine Size', color='r')
plt.show()


# regresyon işlemi sonucunda elde edilen katsayılarımız ne kadar doğru test edelim.
test_x = np.asarray(test_df[['ENGINESIZE']])
test_y = np.asarray(test_df[['CO2EMISSIONS']])
test_tahmini = regression.predict(test_x)
print(test_tahmini)
print(f'r2 score: %.2f' % r2_score(test_tahmini, test_y))  # r2_score() doğruluk payını bulur. Score 0'a yaklaşdıkça testimiz başarısız 1'e yaklaştıkça testimiz başarı demektir. Bunun haricinde regresyon ile mean_squared_error'da kullanılabilir.
# endregion




# region Bu sefer de CYLINDERS ve CO2EMISSIONS arasındaki ilişkiyi lineer regresyonla bulalım

cdf = df[['CYLINDERS', 'CO2EMISSIONS']]  # bağımsız CYLINDERS bağımlı CO2EMISSIONS
print(cdf.head().to_string())


msk = np.random.rand(len(df)) <= 0.7
print(msk)

train_df = cdf[msk]
test_df = cdf[~msk]
print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')

regresyon = linear_model.LinearRegression()

train_x = np.asarray(train_df[['CYLINDERS']])
train_y = np.asarray(train_df[['CO2EMISSIONS']])

regresyon.fit(train_x, train_y)

print('Coef Katsayısı %.2f' % regresyon.coef_[0][0])

print('Intercept katsayısı %.2f' % regresyon.intercept_[0])


cylinder_num = int(input('Lütfen Silindir Sayısını Giriniz: '))
y = regresyon.coef_[0][0] * cylinder_num + regresyon.intercept_[0]
print(f'Carbon Emisyonu = {y}')


# görselleştirelim

plt.figure(figsize=(10,7))
plt.scatter(train_df['CYLINDERS'], train_df['CO2EMISSIONS'], alpha=0.50, color='green')
plt.plot(train_x, regresyon.coef_[0][0] * train_x + regresyon.intercept_[0], color='b')
plt.suptitle('Cylinder Num ve CO2 Arasındaki İlişki', color='r')
plt.ylabel('Emisyon', color='r')
plt.xlabel('Cylinder Num', color='r')
plt.legend(train_df.index, prop={'size':8})
plt.show()


# regresyon sonucunda elde edilen katsayıların doğruluğunu test edelim

test_x = np.asarray(test_df[['CYLINDERS']])
test_y = np.asarray(test_df[['CO2EMISSIONS']])
dogruluk_tahmini = regresyon.predict(test_x)
print(dogruluk_tahmini)
print(f'r2_score: %.2f' % r2_score(dogruluk_tahmini,test_y))
# endregion





# region FUELCONSUMPTION_COMB ile CO2 arasındaki ilişkiyi yap.  ????? S O R ????

fdf = df[['FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(fdf.head().to_string())

msk = np.random.rand(len(df)) <= 0.9
print(msk)

train_df = fdf[msk]
test_df = fdf[~msk]

print(train_df.shape)
print(test_df.shape)

regression = linear_model.LinearRegression()

train_x = np.asarray(train_df[['FUELCONSUMPTION_COMB']])
train_y = np.asarray(train_df[['CO2EMISSIONS']])

regression.fit(train_x, train_y)

reg_coef = float('%.2f' % regression.coef_[0][0])  # floata dönüştürdüm çünkü içinde string ifade olduğu için string olarak algılanmasın diye float türüne dönüştürdüm
reg_intercept = float('%.2f' % regression.intercept_[0])

print(reg_coef)
print(reg_intercept)

ortalama_yakit_tuketimi = float(input("Ortalama YakıtTüketimini Giriniz: "))
y = reg_coef * ortalama_yakit_tuketimi + reg_intercept
print(floor(y))


plt.figure(figsize=(10,7))
plt.scatter(train_df['FUELCONSUMPTION_COMB'], train_df['CO2EMISSIONS'], alpha=0.50, color='b')
plt.plot(train_x, reg_coef * train_x + reg_intercept, color='green')
plt.suptitle('Ortalama Yakıt Tüketimi ve CO2 Arasındaki İlişki', color='r')
plt.ylabel('Emisyon', color='r')
plt.xlabel('Ortalama Yakıt Tüketimi', color='r')
plt.legend(['Veri Noktaları', 'Tahmin Doğrusu'], prop={'size':10})
plt.show()


# doğruluk payını test edelim
test_x = np.asarray(test_df[['FUELCONSUMPTION_COMB']])
test_y = np.asarray(test_df[['CO2EMISSIONS']])
dogruluk_payi = regression.predict(test_x)
print(dogruluk_payi)
print(f'r2 Skoru = %.2f' % r2_score(dogruluk_payi,test_y))  # r2_score fonksiyonunda, ilk argüman gerçek değerler ve ikinci argüman tahmin edilen değerler olmalıdır. Bu nedenle, doğruluk payını hesaplarken bu sırayı dikkate alarak değişiklik yapıldı. neden böyle yapıldı (test_y, dogruluk_payi)
# endregion



# region FUELCONSUMPTION_COMB_MPG sütunun sil
df.drop('FUELCONSUMPTION_COMB_MPG', inplace=True, axis=1)
print(df.head().to_string())
# endregion