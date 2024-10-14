

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Genellikle sınıflandırma için kullanılır (taş veya mayın gibi). Genellikle sigmoid fonksiyonu kullanılır bu da tahmin edilen değeri 0 ile 1 arasına sıkıştırır. Böylece başarılı şekilde sınıflandırma yapılmış olur
from sklearn.metrics import accuracy_score


df = pd.read_csv('Data/Copy of sonar data.csv', header=None)  # header=None kullanımı ilk satırın sütun ismi olarak değil veri olarak alınmasını istediğim için yaptım
print(df.head().to_string())

df.shape

df.describe()  # verilerin istatistiksel ölçümlerini belirttim

df[60].value_counts()  # 60'ıncı sütundaki yani son sütundaki verilerden kaç farklı şekilde var ona baktım

# NOT: Yukarıdaki işlemi yaptığımızda (kaç tane mine kaç tane rock verisi var baktığımızda) sayıların birbirine yakın olduğunu görüyoruz. Bu verilerle predict edeceğimiz model iyi sonuç verecektir. Ancak veriler arasıdna çok fazla fark olsaydı (500 mine vs 5000 rock gibi) o zaman o verilerle predict edeceğimiz model doğru başarılı olmayacak ve doğru sonuçları veremeyecekti..!


df.groupby(60).mean()  # groupby(60) ile mine ve rock olan 60'ıncı sütundaki veriyi grupladık ve .mean() ile her grubun ortalama değerlerini hesapladık.


# verileri ve etiketleri ayırma (separating data and labels)
# 1.yol
X = df.drop(columns=60, axis=1)
print(X)

y = df[60]
print(y)


# 2.yol
X = df[[0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59]]
print(X)

y = df[60]
print(y)


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)  # stratify=y kullanmamızın sebebi sınıf dengesini korumak için yaptık. (örnek olarak veriyi böldüğümüzde hem train verisinde %70M %30R hem de test verisinde %70M %30R olarak eşit etiketlere göre eşit dağıtılmasını sağladık.) Böylelikle modelin performansı ve tutarlılığı arttı.

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


model = LogisticRegression()
model.fit(X_train,y_train)


# Model Evaluation (Modeli Değerlendirme)
# Bu kısım çok kritik ve önemlidir çünkü modelin genelleme yeteneğini ölçmek, overfitting veya underfitting kontrolü ve model performansını farklı metriklerle karşılaştırıp optimize etmek için evaluation yapmak gerekir.

# accuracy (doğruluk) değerlendirmesi
# 1.yol
y_predict = model.predict(X_test)  # model.predict(X_test): Modelin test verisi üzerindeki tahminlerini yapar ve sonuçları y_predict değişkenine atar.
print(y_predict)

print(f'train accuracy: {accuracy_score(y_train, model.predict(X_train))}')  # Eğitim Seti Doğruluğu: y_train üzerinde tahmin yaparak accuracy_score fonksiyonu ile hesaplanır.
print(f'test accuracy: {accuracy_score(y_test, y_predict)}')  # Test Seti Doğruluğu: y_test ve y_predict kullanılarak hesaplanır.

# 2.yol
y_predict = model.predict(X_test)
print(y_predict)

X_predict = model.predict(X_train)
print(X_predict)

print(f'train accuracy: {accuracy_score(y_train,X_predict)}')
print(f'test accuracy: {accuracy_score(y_test,y_predict)}')


input_data = (0.0249,0.0119,0.0277,0.0760,0.1218,0.1538,0.1192,0.1229,0.2119,0.2531,0.2855,0.2961,0.3341,0.4287,0.5205,0.6087,0.7236,0.7577,0.7726,0.8098,0.8995,0.9247,0.9365,0.9853,0.9776,1.0000,0.9896,0.9076,0.7306,0.5758,0.4469,0.3719,0.2079,0.0955,0.0488,0.1406,0.2554,0.2054,0.1614,0.2232,0.1773,0.2293,0.2521,0.1464,0.0673,0.0965,0.1492,0.1128,0.0463,0.0193,0.0140,0.0027,0.0068,0.0150,0.0012,0.0133,0.0048,0.0244,0.0077,0.0074)

input_data_reshape = np.asarray(input_data).reshape(1,-1)

input_data_prediction = model.predict(input_data_reshape)
print(input_data_prediction)

if (input_data_prediction[0] == 'R'):
        print('The object is a Rock. / Bu obje bir Kaya.')
else:
        print('WARNING!! The object is a Mine!! / TEHLİKE!! Bu obje bir Mayın!!')