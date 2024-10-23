

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#region Veri temizleme ve veri manipülasyonları
df = pd.read_csv('Data/AmesHousing.tsv', sep='\t')
print(df.head())

df.info()

df.shape

df.describe()

df['SalePrice'].describe()

df['Sale Condition'].describe()
df['Sale Condition'].value_counts()


# corelasyon corr()
df_num = df.select_dtypes(include=['int64','float64'])
print(df_num)

df_num_cor = df_num.corr()['SalePrice'].sort_values(ascending=False)
print(df_num_cor)

en_iyi_num_cor = df_num_cor[abs(df_num_cor) > 0.5]
print(en_iyi_num_cor)
print(en_iyi_num_cor.count())

len(df_num.columns)

for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['SalePrice'])


untransformed_saleprice = sns.displot(df['SalePrice'])
print(f'Skewness yani Çarpıklık değeri {df['SalePrice'].skew()}')  # Yani çarpıklık değeri (skewness) çok fazla. Bunu azaltmak için logaritma yapıcaz. (np.log())

logaritma_saleprice = np.log(df['SalePrice'])  # logaritmik işlem yapıldı ve çarpıklık değeri azaldı

transformed_saleprice = sns.displot(logaritma_saleprice)
print(f'Skewness yani Çarpıklık değeri {logaritma_saleprice.skew()}')


df['Lot Area']
lot_area_untransformed = sns.displot(df['Lot Area'])
print(f'Skewness = {df['Lot Area'].skew()}')

logaritma_lot_area = np.log(df['Lot Area'])

lot_area_transformed = sns.displot(logaritma_lot_area)
print(f'Skewness = {logaritma_lot_area.skew()}')


# Tekrarlanan Verileri Temizleme
# Tekrarlanan Order sütununun kaldırılması
remove_sub = df.drop_duplicates(subset=['Order'])
print(remove_sub)
df.index.is_unique


# Eksik Değerleri Bulmak
toplam_eksik_degerler = df.isnull().sum().sort_values(ascending=False).head(20)
print(toplam_eksik_degerler)

toplam_eksik_degerler.plot(kind='bar', figsize=(10,7), fontsize = 10)
plt.xlabel('Columns', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title("Toplam Eksik Veriler", fontsize = 20)

# Eksik (null) değerlere sahip olan satırları .dropna() ile kaldırmak
df['Lot Frontage'].isnull().sum()
df1 = df.dropna(subset=['Lot Frontage'])
print(df1['Lot Frontage'].isnull().sum())

# Eksik (null) değerler içeren tüm özelliği (sütunu) drop() yöntemi kullanarak kaldırmak
df['Lot Frontage'].isnull().sum()
df2 = df.drop('Lot Frontage',axis=1)
print(df2)

# Boş değerleri fillna() ile doldurma (sıfır, ortalama, medyan vb)
medyan = df['Lot Frontage'].median()
medyan

df['Lot Frontage'] = df['Lot Frontage'].fillna(medyan)
df['Lot Frontage'].tail()


df['Mas Vnr Area'].isnull().sum()
ortalama = df['Mas Vnr Area'].mean()
ortalama

df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(ortalama)
print(df['Mas Vnr Area'])


# Özellik Ölçekleme (Scaling)

# Min-max ölçekleme (ya da normalizasyon) en basit olanıdır: değerler kaydırılır ve yeniden ölçeklendirilerek 0 ile 1 arasında bir aralığa ulaşır. Bu, minimum değerin çıkarılması ve maksimum değerden minimum değerinin çıkarılmasına bölünmesiyle yapılır.

normalizasyon_df = MinMaxScaler().fit_transform(df_num)
normalizasyon_df


# Standartlaştırma ise farklıdır: önce ortalama değer çıkarılır (bu yüzden standartlaştırılmış değerler her zaman sıfır ortalamaya sahiptir), ardından standart sapmaya bölünerek sonuçta birim varyansa sahip bir dağılım elde edilir.

standartizasyon_df = StandardScaler().fit_transform(df_num)
standartizasyon_df


df['SalePrice']
saleprice_standartizasyon = StandardScaler().fit_transform(df['SalePrice'].to_numpy().reshape(-1,1))
saleprice_standartizasyon
# endregion