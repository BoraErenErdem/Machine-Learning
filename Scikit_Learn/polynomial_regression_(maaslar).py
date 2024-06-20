

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures  # Bu ifade herhangi bir sayıyı polinomal şekilde ifade etmeye yarar ve polinom derecesi vermeye yarar
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model



df = pd.read_csv('Data/maaslar.csv')
print(df.head().to_string())

# Verileri sıralama
df = df.sort_values('Egitim Seviyesi')
print(df.head().to_string())

X = df[['Egitim Seviyesi']].values
print(X)

y = df['maas'].values
print(y)

# Sıralanmış verilerle train_test_split kullanma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Linear regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)


# Polynomial regression
polynomial_features = PolynomialFeatures(degree=4)  # Dereceyi 4 olarak belirledik. Derecenin 4 olma sebebi modelin ve grafiğin daha düzgün sonuç vermesi çıktının ve modelin daha anlamlı olması
X_poly_train = polynomial_features.fit_transform(X_train)
X_poly_test = polynomial_features.transform(X_test)

polynomial_regression = LinearRegression()
polynomial_regression.fit(X_poly_train, y_train)

# Daha düzgün bir polinom eğrisi çizmek için geniş bir aralıkta X değerleri oluşturma
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)  # np.linspace() fonksiyonu belirli bir aralıkta eşit aralıklı sayılar oluşturmak için kullanılır. Buradaki kullanma şekli başlangıç ve bitiş noktaları arasında belirli sayıda eşit aralıklı nokta üretmektir. X.min(), X.max(), 500 ifadeleri ise X'in minimum ve maksimum olacak şekilde arasını 500 eş parçaya böl dedik. Bunun amacı grafiği ve çıktıyı daha net ve pürüszüs gösterilmesini sağlamak. 500 yerine başka sayı verebilirim ancak bu maliyet, doğru olmayan çıktı ya da işlem yükü olarak bana geri dönebilir.
# reshape(-1,1) kısmında ise -1 kısmı otomatik olarak boyutunu belirlemesi için yazdım. Virgülden sonra 1 yazmamın sebebi iki boyutlu bir sütun matrisi olması ve her satırın bir veri noktası olması için yazdım. Eğer reshape(-1,) yapsaydım o zaman tek boyutlu kalmaya devam edecekti. Ancak benim makine öğreniminde işlem yapabilmem için 2 boyutlu sütun matrisine ihtiyacım var (kural bu)
X_range_poly = polynomial_features.transform(X_range)


# NOT: Birçok makine öğrenimi algoritması, girdilerin iki boyutlu matrisler halinde olmasını gerektirir..! (500,1 gibi)


# train verisi için görselleştirme
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.plot(X_range, polynomial_regression.predict(X_range_poly), color='red', label='Polynomial Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.title('Polinom Regresyon ile Maaş Tahmini (Eğitim Verisi)')
plt.legend()
plt.show()

# test verisi için görselleştirme
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_range, polynomial_regression.predict(X_range_poly), color='orange', label='Polynomial Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.title('Polinom Regresyon ile Maaş Tahmini (Test Verisi)')
plt.legend()
plt.show()




# region Polinomal regresyon ile maaşların eğitim seviyesine göre oranını bul ve grafikte göster
df = pd.read_csv('Data/maaslar.csv')
print(df.to_string())

df.columns

X = df[['Egitim Seviyesi']].values
print(X)

y = df['maas'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


regresyon = linear_model.LinearRegression()
regresyon.fit(X_train,y_train)

polinom_regresyon = preprocessing.PolynomialFeatures(degree=3)
X_train_polinom = polinom_regresyon.fit_transform(X_train)
X_test_polinom = polinom_regresyon.transform(X_test)


polinom_regresyon_islemi = linear_model.LinearRegression()
polinom_regresyon_islemi.fit(X_train_polinom,y_train)

X_graph_pice = np.linspace(X.min(),X.max(),500).reshape(-1,1)
X_graph_pice_polinom = polinom_regresyon.transform(X_graph_pice)


# train verisi için görselleştirme
plt.scatter(X_train, y_train, color='green', label='Train Data')
plt.plot(X_graph_pice, polynomial_regression.predict(X_range_poly), color='red', label='Polynomial Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.title('Polinom Regresyon ile Maaş Tahmini (Eğitim Verisi)')
plt.legend()
plt.show()

# test verisi için görselleştirme
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_graph_pice, polynomial_regression.predict(X_range_poly), color='purple', label='Polynomial Regression')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.title('Polinom Regresyon ile Maaş Tahmini (Test Verisi)')
plt.legend()
plt.show()


# predict kısmı (tahmin)
print(regresyon.predict([[11]]))
print(regresyon.predict([[6.6]]))
# endregion