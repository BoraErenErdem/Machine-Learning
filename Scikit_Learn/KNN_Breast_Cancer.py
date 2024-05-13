

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor  # k en yakın komşuluk regresyonu



# from sklearn.datasets import load_breast_cancer komutu ile 2007 yılında yapılmış meme kanseri riskinin olup olmaması ilişkisine bakılmıştır.



# Veri yükleniyor
data = load_breast_cancer()
X = data.data
y = data.target

print(X)
print('=================')
print(y)

# Veri setinin ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Doğruluk skorlarını saklamak için boş listeler oluşturuluyor
train_accuracy = []
test_accuracy = []

# Kullanılacak komşuluk sayısına kadar döngü
komsuluk_sayisi = int(input("Komşuluk sayısını giriniz: "))

for k in range(1, komsuluk_sayisi):
    sinif = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    # Eğitim ve test doğruluk skorlarını hesaplayıp listelere ekleniyor
    train_accuracy.append(sinif.score(X_train, y_train))  # .score() ile accuracy_score() aynı işlevi görür aralarında fark yoktur..! Genel olarak accuracy_score() sınıflandırma modellerinde kullan.
    test_accuracy.append(sinif.score(X_test, y_test))

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')


# Grafik çizdirme
plt.figure(figsize=(10,7))
plt.plot(range(1, komsuluk_sayisi), train_accuracy, label='Eğitim Doğruluk')
plt.plot(range(1, komsuluk_sayisi), test_accuracy, label='Test Doğruluk')
plt.ylabel('Doğruluk', color='r')
plt.xlabel('Komşuluk Sayısı', color='r')
plt.legend()
plt.show()

print(f'En iyi train sırası: {train_accuracy.index(max(train_accuracy)) + 1}\n'
        f'En iyi train değeri: {max(train_accuracy)}')

print(f'En iyi test sırası: {test_accuracy.index(max(test_accuracy)) + 1}\n'
        f'En iyi test değeri: {max(test_accuracy)}')



# KNN En Yakın Komşuluk Regresyonu
regression = KNeighborsRegressor(n_neighbors=20)

regression.fit(X_train, y_train)

regressionn_score = regression.score(X_test, y_test)
print(f'Regresyon skoru: {regressionn_score}')  # Tahmin edilen değer, yeni bir nokta için belirlenen komşu noktaların ortalamasıdır.