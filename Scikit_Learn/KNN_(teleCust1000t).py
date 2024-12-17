

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns



df = pd.read_csv('Data/teleCust1000t.csv')
print(df.head(10).to_string())

df.info()

# region KNN algoritması yap (custcat için bul)

# Hangi paketi kullanan kaç kullanıcı var
df['custcat'].value_counts()



# K-Nearest Neighborhood algoritması veri setimizdeki her bir satırı ya da noktanın bir diğerine olan mesafesini hesaplamak için geometrik hesaplamalar kullanır. Bunun için veri setindeki her bir value'yi alacağız. Yani bir features matrix oluşturuyoruz. (özellikler matrisi)

print(df.columns)

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values  # tüm satırların tek tek valueslerini aldık

print(X)


# custcat için de bir matrix oluştur

y = df['custcat'].values  # custcat sütununun tüm satırının valueslerini aldık

print(y)


# X matrixindeki değerleri incelediğimizde income verisinde 944, age verisinde 34, gender verisinde 0 ya da 1 verileri bulunmaktadır. Bu tüm değerler scaler olarak farklı büyüklüklere sahiptir. Bu yüzden veri setimizdeki valueların normalize edelim. (sayıları rastgele örnek olsun diye verdim)

X = preprocessing.StandardScaler().fit_transform(X)  # preprocessing modülünün StandardScaler() snıfından instance alıp fit_transform() fonksiyonunu kullanarak Normalizasyon işlemi yaptık.
print(X)


# Veri setini train ve test olarak split et
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)
print(f'Train Sets: {X_train.shape}, {y_train.shape}')
print(f'Test Sets: {X_test.shape}, {y_test.shape}')


# Train veri setimizdeki her bir nokta için yani features için en yakın 4 komşusuna bakacak şekilde KNN algoritmamızı train edelim. Burada 4 komşuya bakmasının her hangi bir mantığı yok kafamıza göre verdik. Yakınlık derecesi ve grupların sayısı burada önemli faktördür.

neighboor = KNeighborsClassifier(n_neighbors=4).fit(X_train,y_train)  # n_neighbors= bakılacak komşu sayısı  # Bu sınıfın fit() fonksiyonu mesafe hesaplaması yapar.
y_result = neighboor.predict(X_test)
print(y_result)

# NOT: predict() eğitilmiş bir modelin özelliklerle ilişkilendirilen bir çıktıyı tahmin etmek için kullanılır..!


# Doğruluk değerlenidirmesi
# Birden fazla sınıf oluşturulacak yada etiket oluşturulacak modellerde her bir subclass ya da label için doğrulama yapmak gerekmektedir.

print(f'Train Set Accuracy: {metrics.accuracy_score(y_train, neighboor.predict(X_train))}')  # output: 0.543
print(f'Test Set Accuracy: {metrics.accuracy_score(y_test, neighboor.predict(X_test))}')  # output: 0.325

# Yukarıdaki çıktı incelendiğinde 4 komşu için algoritmanın başarısız olduğunu görülür. Accuracy score 1'e yaklaştıkça başarılıdır.

# NOT: accuracy_score() fonkisyonu arkasında çalışan matematikte Jaccard Similarity Index kullanılmaktadır. Yani 'y_train' ile 'neighboor.predict(X_train)' arasındaki benzerliği ve çeşitliliği bulmak için kullanılmaktadır..!


# Veri setimizde 4 tip telekominikasyon paketi var. Bu yüzden bizden 4 tane sınıf oluşturmamız ve kullanıcıların hangi sınıflara ait olduğunu saptamamız gerekir. Bu bağlamda yeni gelen bir müşterinin features'larına göre ona en uygun paketi önerebilelim.
# Yukarıda 4 komşu için başarısız olduk. Peki kaç komşuya bakarsak en iyi sonucu elde edeceğiz.

k_neighboor = int(input('K sayısı giriniz: '))
array_lenght = k_neighboor - 1  # index eleman sayısından 1 eksik olduğu için 1 çıkardık.
jsi_acc = np.zeros(array_lenght)
std_acc = np.zeros(array_lenght)

for k in range(1,k_neighboor):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_prediction = neigh.predict(X_test)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, y_prediction)
    std_acc[k - 1] = np.std(y_prediction == y_test) / np.sqrt(y_prediction.shape[0])


print(f'jsi score: {jsi_acc}')
print(f'std score: {std_acc}')


# Görselleştirme

plt.figure(figsize=(10,7))
plt.plot(range(1, k_neighboor), jsi_acc, color='green')
plt.fill_between(range(1, k_neighboor), jsi_acc + 1 * std_acc, alpha=0.3)
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()  #  alt grafiklerin ve eksen etiketlerinin çakışmasını önlemek için otomatik olarak alt grafiklerin arasındaki boşluğu ayarlar
plt.show()

print(f'En iyi doğruluk değeri {jsi_acc.max()}, k = {jsi_acc.argmax()+1}')  # .argmax() en büyük değerin indexini verir. + 1 koymamızın sebeni sıfırsız saysın diye
# endregion



# region veri setindeki müşterilerin özelliklerine dayanarak bir müşterinin bir telekomünikasyon şirketinden ne tür bir hizmet planı alacağını tahmin etmek için bir KNN modeli oluştur
df.columns

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values
print(X)

y = df['custcat'].values
print(y)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

komsu = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
y_sonucu = komsu.predict(X_test)
print(y_sonucu)

print(f'Test: {metrics.accuracy_score(y_test, komsu.predict(X_test))}')
print(f'Train: {metrics.accuracy_score(y_train, komsu.predict(X_train))}')


k_sayisi = int(input("K sayısı gir: "))
accuracy_lenght = k_sayisi - 1
jsi_acc = np.zeros(accuracy_lenght)
std_acc = np.zeros(accuracy_lenght)

for k in range(1, k_sayisi):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_predict = neigh.predict(X_test)

    jsi_acc[k-1] = metrics.accuracy_score(y_test, y_predict)
    std_acc[k-1] = np.std(y_predict == y_test) / np.sqrt(y_predict.shape[0])


print(f'jis skoru: {jsi_acc}')
print(f'std skoru: {std_acc}')


plt.figure(figsize=(10,7))
plt.plot(range(1, k_sayisi), jsi_acc, color='b')
plt.fill_between(range(1,k_sayisi), jsi_acc + 1 * std_acc, alpha=0.2, color='r')
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En iyi doğruluk değeri {jsi_acc.max()}, k = {jsi_acc.argmax()+1}')
# endregion



# region Veri setindeki müşterilerin çeşitli demografik ve hizmetle ilgili özelliklerine dayanarak, bir müşterinin belirli bir hizmet türüne abone olup olmayacağını tahmin et
df.columns

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values

print(X)

y = df['custcat'].values
print(y)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


neighboor = KNeighborsClassifier(n_neighbors=12).fit(X_train, y_train)
y_result = neighboor.predict(X_test)
print(y_result)

print(f'Train sets: {metrics.accuracy_score(y_train, neighboor.predict(X_train))}')
print(f'Test sets: {metrics.accuracy_score(y_test, neighboor.predict(X_test))}')

neighboor_sayisi = int(input("Neighbor sayısı giriniz: "))
jsi_acc = np.zeros(neighboor_sayisi - 1)
std_acc = np.zeros(neighboor_sayisi - 1)

for k in range(1, neighboor_sayisi):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_tahmini = neigh.predict(X_test)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, y_tahmini)
    std_acc[k - 1] = np.std(y_tahmini == y_test) / np.sqrt(y_tahmini.shape[0])


print(f'jis skoru: {jsi_acc}')
print(f'std skoru: {std_acc}')


plt.figure(figsize=(10,7))
plt.plot(range(1, neighboor_sayisi), jsi_acc, color='b')
plt.fill_between(range(1, neighboor_sayisi), jsi_acc + 1 * std_acc, color='green', alpha=0.25)
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En iyi doğruluk değeri: {jsi_acc.max()}\n'
        f'Sırası: {jsi_acc.argmax() + 1}')
# endregion



# region Veri setindeki müşterilerin bütün özellik ve etkili parametrelere dayanarak, bir müşterinin hangi hizmet türüne ne kadar oranla abone olacağını tahmin et
print(df.columns)

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values

print(X)

y = df['custcat'].values
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)


komsu_katsayisi = int(input('Katsayı giriniz: '))
array_lenght = komsu_katsayisi - 1
jacard_similarity_index = np.zeros(array_lenght)
standart_sapma = np.zeros(array_lenght)

for k in range(1, komsu_katsayisi):
    sinif = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_prediction = sinif.predict(X_test)

    jacard_similarity_index[k - 1] = metrics.accuracy_score(y_test, y_prediction)
    standart_sapma[k - 1] = np.std(y_prediction == y_test) / np.sqrt(y_prediction.shape[0])


print(f'jacard similartity index: {jacard_similarity_index}')
print(f'Standart Sapma: {standart_sapma}')


plt.figure(figsize=(10,7))
plt.plot(range(1, komsu_katsayisi), jacard_similarity_index, color='green')
plt.fill_between(range(1, komsu_katsayisi), jacard_similarity_index + 1 * standart_sapma, alpha=0.25, color='brown')
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En iyi doğruluk oranın skoru: {jacard_similarity_index.max()}\n'
        f' En iyi doğruluk oranının sırası: {jacard_similarity_index.argmax() + 1}')
# endregion



# region kişinin hangi müşteri kategorisine (custcat) ait olduğunu tahmin et.
df.columns

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values

print(X)

y = df['custcat'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

neighboor = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
y_predict = neighboor.predict(X_test)
print(y_predict)

k_neigh = int(input("Katsayı giriniz: "))
jsi_acc = np.zeros(k_neigh-1)
std_acc = np.zeros(k_neigh-1)

for k in range(1, k_neigh):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    prediction_y = neigh.predict(X_test)
    print(prediction_y)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, prediction_y)
    std_acc[k - 1] = np.std(prediction_y == y_test) / np.sqrt(prediction_y.shape[0])


print(f'jsi score: {jsi_acc}')
print(f'std score: {std_acc}')


plt.figure(figsize=(10,7))
plt.plot(range(1, k_neigh), jsi_acc, color='yellow')
plt.fill_between(range(1, k_neigh), jsi_acc + 1 * std_acc, alpha=0.25, color='green')
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En iyi doğruluk değeri: {jsi_acc.max()}\n'
        f'Sırası: {jsi_acc.argmax() + 1}')
# endregion



# region Kullanıcıların hangi custcat'i seçmeye daha yatkın olduğunun tahminini yap
df.columns

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values
print(X)

y = df['custcat'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

X = preprocessing.StandardScaler().fit_transform(X)  # Normalizasyon işlemi yaptık
print(X)


neighboor = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
y_sonucu = neighboor.predict(X_test)
print(y_sonucu)

print(f'test set: {metrics.accuracy_score(y_test, neighboor.predict(X_test))}')
print(f'train set: {metrics.accuracy_score(y_train, neighboor.predict(X_train))}')


k_katsayisi = int(input("K katsayısını giriniz: "))
array_uzunlugu = k_katsayisi - 1
jsi_acc = np.zeros(array_uzunlugu)
std_acc = np.zeros(array_uzunlugu)

for k in range(1, k_katsayisi):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    predictions = neigh.predict(X_test)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, predictions)
    std_acc[k - 1] = np.std(predictions == y_test) / np.sqrt(predictions.shape[0])


print(f'jsi score: {jsi_acc}')
print(f'std score: {std_acc}')


plt.figure(figsize=(10,7))
plt.plot(range(1, k_katsayisi), jsi_acc, color='r')
plt.fill_between(range(1, k_katsayisi), jsi_acc + 1 * std_acc, alpha=0.20)
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()


print(f'En iyi doğruluk oranı: {jsi_acc.max()}\n'
        f'Sırası: {jsi_acc.argmax() + 1}')
# endregion



# region Veri setindeki müşterilerin yaşına, gelirine veya eğitim seviyesine dayanarak, belirli bir kategoriye ait olma olasılıklarını tahmin et
df.columns

X = df[['age', 'income', 'ed']].values
print(X)

y = df['custcat'].values
print(y)


X_train , X_test, y_train, y_test = train_test_split(X, y ,random_state=42, test_size=0.2)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)


neighbor = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_sonuc = neighbor.predict(X_test)
print(y_sonuc)


komsuluk_kat_s = int(input("Komşuluk kat sayısı giriniz: "))
uzunluk = komsuluk_kat_s - 1
jsi = np.zeros(uzunluk)
std = np.zeros(uzunluk)

for k in range(1, komsuluk_kat_s):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_tahmin = neigh.predict(X_test)

    jsi[k - 1] = metrics.accuracy_score(y_test, y_tahmin)
    std[k - 1] = np.std(y_tahmin == y_test) / np.sqrt(y_tahmin.shape[0])


print(f'jsi:  {jsi}')
print(f'std: {std}')


plt.figure(figsize=(10,7))
plt.plot(range(1, komsuluk_kat_s), jsi, color='purple')
plt.fill_between(range(1, komsuluk_kat_s), jsi + 1 * std, color='pink', alpha=0.5)
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()


print(f'En iyi doğruluk değeri: {jsi.max()}\n'
        f'Sırası: {jsi.argmax() + 1}')
# endregion



# region Veri setindeki özellikler arasında hangileri, bir müşterinin kategorisini belirlemek için daha belirleyici olabilir (Bu soruyu görsel araçlarla çeşitli gösterimler yap)

features = ['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']
target = 'custcat'

plt.figure(figsize=(18, 10))
for i, feature in enumerate(features, 1):  # enumerate(features, 1) kullanılarak, her özellik için bir sayaç oluşturulur ve bu sayaç i değişkenine atanır, özellik de feature değişkenine atanır.
    plt.subplot(3, 4, i)
    sns.histplot(data=df, x=feature, hue=target, multiple="stack")
    plt.title(f'{feature} Dağılımı')
    plt.xlabel(feature)
    plt.ylabel("Frekans")
plt.tight_layout()
plt.show()


# Özellikler Arasındaki Korelasyon Haritası'nın gösterimi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Özellikler Arasındaki Korelasyon Haritası")
plt.show()
# endregion



# region Bütün features matrixlerin custcat ile olan ilişkisine bak
df.columns

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values
print(X)

y = df['custcat'].values
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

komsu = KNeighborsClassifier(n_neighbors=6).fit(X_train,y_train)
y_sonuc = komsu.predict(X_test)
print(y_sonuc)

print(f'Train accuracy score: {metrics.accuracy_score(y_test, komsu.predict(X_test))}')
print(f'Test accuracy score: {metrics.accuracy_score(y_train, komsu.predict(X_train))}')


komsu_katsayi = int(input("Katsayı giriniz: "))
array_uzunluk = komsu_katsayi - 1
jsi_acc = np.zeros(array_uzunluk)
std_acc = np.zeros(array_uzunluk)

for k in range(1, komsu_katsayi):
    ev = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
    y_predict_ciktisi = ev.predict(X_test)
    print(y_predict_ciktisi)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, y_predict_ciktisi)
    std_acc[k - 1] = np.std(y_predict_ciktisi == y_test) / np.sqrt(y_predict_ciktisi.shape[0])


print(f'jsi acc: {jsi_acc}')
print(f'std_acc: {std_acc}')


plt.figure(figsize=(10,7))
plt.plot(range(1, komsu_katsayi), jsi_acc, color='green')
plt.fill_between(range(1, komsu_katsayi), jsi_acc + 1 * std_acc, alpha=0.33, color='b')
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En iyi doğruluk değeri: {jsi_acc.max()}\n'
        f'Sırası: {jsi_acc.argmax() + 1}')
# endregion



# region featuresların custcat ile olan ilşkisini elbow method kullanarak ve error rate hesaplaması yaparak bul
X = df.drop('custcat',axis=1).values
print(X)

y = df['custcat'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

standard_scaler = preprocessing.StandardScaler()
X = standard_scaler.fit_transform(X)
print(X)

error_rates = []
for k in range(1, 32):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pre = knn.predict(X_test)
    error_rates.append(1 - accuracy_score(y_test, y_pre))

print(f'Error Rates = {error_rates}')

plt.figure(figsize=(10,7))
plt.plot(range(1, 32), error_rates, color='green')
plt.legend(('error_rate'), prop={'size': 8})
plt.xlabel('Neighboors Number', color='r')
plt.ylabel('Error Rate', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En düşük error rate değeri = {min(error_rates)}')
print(f'En düşük error rate değerine sahip K sayısı = {np.argmin(error_rates) + 1}')
# endregion