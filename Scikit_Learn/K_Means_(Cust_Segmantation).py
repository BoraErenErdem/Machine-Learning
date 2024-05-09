

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D



df = pd.read_csv('Data/Cust_Segmentation.csv')
print(df.head().to_string())


# region Veri setini önce gereksiz dosyalardan ayıkla, sonra işlemeye hazır hale getirip K_Means algoritmasını uygula.
# Adress sütununu sil
df.drop(['Address'], axis=1, inplace=True)
print(df.head().to_string())



# NaN olan değerleri bul
print(df.isnull().sum())



# NaN olan değerlerin yerine ortalama alop onları yaz yani veri setini düzenle
df['Defaulted'] = df['Defaulted'].replace(to_replace=np.nan, value=df['Defaulted'].astype(float).mean())
print(df['Defaulted'])



# Veri setini düzenlemek için Normalizasyon yap
X = df.values[:,1:]  # df.values ifadesi, pandas DataFrame'inin numpy dizisine dönüştürülmesini sağlar. [:, 1:] kısmı ise bu dizideki tüm satırları ve sütunların 1'inden başlayarak sonuna kadar olan sütunları seçer. Yani, DataFrame'deki ilk sütun atlanır ve geri kalan tüm sütunlar alınır..!
print(X)

X = StandardScaler().fit_transform(X)
print(X)


k_means = KMeans(init='k-means++', n_clusters=3, n_init=12)  # n_clusters= Train sonucunda kaç tane küme oluşturulacağını gösterir
k_means.fit(X)  # n_init= KMeans algoritmasında en uygun sonucu vericek olan merkez noktalarını saptamak için kaç deneme yapmak istiyorsan o kadar yaz
labels = k_means.labels_  # init= Amprik olasılık dağılımına (veri kümesinin olasılık dağılımı hesaplama yöntem) göre örneklemeyi kullanarak başlangıç küme merkezlerini seçmek için kullanılır
print(labels)  # init'in greedy ve vanilla diye iki farklı argümanı vardır. Bunların her ikisi de en iyi ağırlık merkezini seçmek için kullanılır
# Vanilla k-means++ merkez belirlemede olasılık yaklaşımını kullanır
# Greedy k-means++ her bir veri setindeki featureslere göre tek tek deneyerek merkezi bulur (default olarak init='k-means++' Greedy gelir)

df['Clus_km'] = labels
print(df.head().to_string())


# Küme merkez noktalarının koordinatları
print(k_means.cluster_centers_)  # .cluster_centers_ ile küme merkez noktalarının koordinatları bulunur


# NOT: X[:, 1] Numpy Index Notation, yani numpy 2d array'inden bu yapı ile data extracts edebiliriz..!


# Görselleştirme
fig = plt.figure(num=1, figsize=(10,7))
plt.clf()  # noktaların üstündeki rakamları kaldırmak için yazdık (bu grafikte çıkmıyor ama normalde bunun için kullanılır)
ax = fig.add_subplot(111, projection='3d')  # 3D projeksiyon için subplot ekleyin
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float_))  # bu kısımda datayı verdik ve bizden tip bekledi biz de tip verdik(astype ile float verdik)
ax.set_xlabel('Education', color='r')
ax.set_ylabel('Age', color='r')  # bu kısımda labellara atadık
ax.set_zlabel('Income', color='r')
plt.show()

# Çalışma sonunda 3 kümeye ayrılan müşterilerimizin demografik bilgilerine göre onları kümeledik.
# Alt yoğun grup ==> Genç ve düşük gelirli
# Orta grup ==> Orta yaş ve orta gelir
# Üst yoğun grup ==> Zengin, eğitimli, ve yaşlı

# endregion