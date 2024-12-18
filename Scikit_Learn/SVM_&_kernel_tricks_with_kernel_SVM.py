

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



# Veri setinin oluşturulması
X,y = make_blobs(n_samples=100, centers=2, random_state=42)  # n_samples=100 -> 100 örneklem oluşturdum.   centers=2 -> 2 farklı sınıf oluşturdum.


# SVM modelinin eğitimi
model = SVC(kernel='linear', C=1.0)
model.fit(X,y)


# Izgara (Grid) oluşturma
# Burada amaç 2 boyutlu bir harita üzerinde her noktayı tahmin etmek için bir ızgara oluşturulur.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # X[:, 0].min() - 1 ve X[:, 0].max() + 1 -> İlk özelliğin (x ekseni) minimum ve maksimum değerlerinin 1 birim dışına çıkarılır.
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))  # np.meshgrid -> X ve Y eksenlerinde her bir noktayı temsil eden bir ızgara matrisi oluşturur.
# 0.01 değeri ızgaranın çözünürlüğünü belirler. Daha küçük değer (adım boyutu) daha detaylı harita oluşturur.


# Tahminler (predictions)
# Burada amaç ızgara üzerindeki her noktayı model kullanarak tahmin etmek.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # np.c_[xx.ravel(), yy.ravel()] -> Izgaradaki her bir noktayı, modelin tahmin edebilmesi için düz bir liste haline getirir.
Z = Z.reshape(xx.shape)  # Z.reshape(xx.shape) -> Tahmin edilen sonuçları, orjinal ızgara boyutuna geri dönüştürür.
# np.c_[xx.ravel(), yy.ravel()] ifadesiyle xx ve yy'yi sütunlar olarak birleştirip her noktanın X ve y koordinatları çiftler halinde bir matris oluşuturldu.


# Karar sınırının (decision boundary) çizilmesi
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)  # plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired) -> Karar sınırlarını 2D harita üzerine çizer.
# cmap=plt.cm.Paired -> Her sınıf için farklı renk belirler.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')  # plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k') -> Veri noktalarını (X) scatter plot (dağılım grafiği) ile çizer.
plt.title("SVM Karar Sınırı")  # c=y: Her sınıfa (etikete) göre farklı renkler kullanır.
plt.xlabel("Özellik 1")  # s=50: Nokta boyutlarını ayarlar.
plt.ylabel("Özellik 2")  # edgecolor='k': Noktaların kenarlarını siyah yapar.
plt.show()