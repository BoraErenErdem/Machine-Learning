

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_sample_images  # sklearn.datasets'den veri setini yükledik
import tensorflow as tf




# region CNN
# Convolution yani Evrişim aslında ağırlıklardan oluşan bir filtredir. Bu filtre ile resimlerdeki öznitelikleri analiz edebiliriz. Matematik olarak her bir pixeli karşısındaki ağırlık değeriyle çarpıp sonuçları topluyoruz. Sonuç olarak pikselleri dönüştürmüş oluyoruz.

# CNN genel olarak görüntü sınıflandırma, nesne tespiti ve görüntü segmentasyonu gibi alanlarda kullanılır. CNN'ler özellikle geleneksel yapay sinir ağlarına kıyasla (ANN) görsel veri işleme kısmında daha etkilidir.

# CNN'lerin temel bileşenleri:
# 1- Evrişim Katmanları: Girdi görüntüsünden özellik haritalarını çıkarma işlemini uygular. Bu katmanlar özellik haritalarını oluşturmak için filtreler veya çekirdekler kullanır, özelliklerin lokaliteye (bölgeye) göre algılanmasına izin verir.

# 2- Aktivasyon Fonksiyonları: Bu kısım modelin derin öğrenme yeteneğini arttırır ve doğrusal olmayan özellikleri öğrenmesini sağlar. Genelde 'relu' kullanılır.

# 3- Havuzlama Katmanları (Pooling Layers): Özellik haritalarını küçültmek ve özellik konumuna daha az duyarlı olmasını sağlar. Max Pooling veya Average Pooling gibi teknikler kullanılır ancak en yaygın olarak kullanılan Max Pooling. (max pooling = piksellerin ağırlıklarının max olanları alır)

# 4- Tam Bağlantılı Katman: Özellik haritalarından elde edilen özellikleri sınıflandırmak için kullanılır. Bu katmanlar geleneksel sinir ağlarına (ANN) benzer şekilde çalışır.

# NOT: CNN'ler görsel veri işlemede çok başarılı çünkü özellik haritalarını otomatik olarak çıkarabilir ve özellik konumuna göre genelleme yapabilir..!
# endregion



images = load_sample_images()['images']  # değişkende iki tane resim var
print(images)

images[0].shape  # değişkendeki ilk resmin boyutu
# output: (427, 640, 3) ==> 427'ye 640 boyutlu ve resim renkli olduğu için 3 kanaldan oluşuyor demek

images[1].shape  # değişkendeki ikinci resmin boyutu


# resmin boyutlarını 80'e 120 yap
images = tf.keras.layers.CenterCrop(height=80, width=120)(images)  # değişkendeki yani veri setindeki resimlerin boyutunu küçülttük

images = tf.keras.layers.Rescaling(scale=1/255)(images)  # değişkendeki yani veri setindeki boyutunu küçülttüğümüz resimleri ölçeklendirdik

images.shape  # değişkenin yeni boyutu
# output: [2, 80, 120, 3] ==> # ilk değer (2) batch_size (örnek sayısı) ifade eder.  # ikinci (80 yükseklik) ve üçüncü (120 genişlik) değer resimlerin boyutunu ifade eder.
# dördüncü değer (3) resimler renkli olduğu için kanal sayısını ifade eder.


# resimleri 2 boyutlu convolution yani evrişim katmanından geçirme işlemi
cov_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)  # filters= kullanmak istediğimiz filtre sayısını belirliyoruz  # kernel_size= filtrenin boyutu

fmaps = cov_layer(images)  # resmin boyutları ve katmanları değişti çünkü 7 x 7 boyutlarına filtre uygulanırken yanadki boşluklardan dolayı filtre tam uygulanamadı. resmin kenarlarına da filtre uygulamak için resmin çevresine 'padding' eklenebilir. orjinal girdide 3 katman varken burada filters=32 kullandığımız için filtre 32 oldu.

fmaps.shape

cov_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same')  # padding='same' ifadesi evrişim işlemlerinde görüntülerin boyutunu korumak için kullanılır ve çıkış görüntüsünün giriş görüntüsü ile aynı boyutta olmasını sağlar. Bilgi kaybını önlemek ve daha doğru sonuçları elde etmek için kullanışlı bir tekniktir.

fmaps = cov_layer(images)  # resim boyutları bozulmadan korunarak çıktı

fmaps.shape  # output: [2, 80, 120, 32]


# Kernels (Filtreler):
# kernels.shape çıktısı, evrişim katmanında kullanılan filtrelerin şeklini belirtir.
# (7, 7, 3, 32) şeklinde bir çıktı, filtrelerin boyutunu ve sayısını belirtir.
# İlk iki boyut, her bir filtre için boyutları belirtir. Bu durumda, 7x7 boyutunda bir filtre kullanıldığı belirtilir.
# Üçüncü boyut, giriş kanallarının sayısını belirtir. Bu durumda, 3 kanallı (RGB gibi) giriş görüntüleri için bir filtre kullanıldığı belirtilir.
# Dördüncü boyut, filtre sayısını belirtir. Bu durumda, 32 farklı filtre kullanıldığı belirtilir.


# Biases (Sapmalar):
# biases.shape çıktısı, evrişim katmanında kullanılan sapmaların şeklini belirtir.
# (32,) şeklinde bir çıktı, her bir filtre için bir sapmanın olduğunu belirtir.
# Bu, evrişim işleminden geçen her bir öznitin (filtrenin) sonuçlarına eklenen bir sabit değeri ifade eder.
# Bias vektörü, her bir filtre için bir sapma içerir. Bu durumda, 32 farklı filtre için 32 farklı sapma olduğu belirtilir.

# NOT: Genellikle her bir filtre için bir bias kullanmak daha yaygın bir yaklaşımdır. Bazı durumlarda filtreden farklı sapma, bias kullanmak işlemin karmaşıklığını ve bakımını arttırabilir.

kernels, biases = cov_layer.get_weights()
kernels.shape
biases.shape

max_pool = tf.keras.layers.MaxPool2D(pool_size=2)  # Max Pooling tekniğini kullandık
output = max_pool(images)

output.shape

global_avg_pool = tf.keras.layers.GlobalAvgPool2D()  # Average Pooling tekniğini kullandık
global_avg_pool(images)