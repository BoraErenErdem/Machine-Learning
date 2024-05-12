

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing,metrics


# diamonds veri seti Seaborn kütüphanesinde bulunan ve elmasların özelliklerini içeren bir veri setidir. Bu veri seti elmasların kesim kalitesi, renk, berraklık, ağırlık gibi özelliklerini içerir.



df = sns.load_dataset('diamonds')
print(df.head().to_string())

sns.lineplot(df)

# region Elmasların fiyatlarını tahmin eden lineer model kur

df = pd.get_dummies(df,columns=['cut', 'color', 'clarity'], drop_first=True, dtype=int)  # drop_first=True ilk değeri düşür demek.  # dtype=int True ve False yerine 1 ve 0 yaz demek.
print(df.head().to_string())

df.columns

X = df[['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Premium',
        'cut_Very Good', 'cut_Good', 'cut_Fair', 'color_E', 'color_F',
        'color_G', 'color_H', 'color_I', 'color_J', 'clarity_VVS1',
        'clarity_VVS2', 'clarity_VS1', 'clarity_VS2', 'clarity_SI1',
        'clarity_SI2', 'clarity_I1']].values

print(X)


y = df['price'].values
print(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = linear_model.LinearRegression()
model.fit(X_train, y_train)

model.score(X_test, y_test)  # 0.9189
model.score(X_train, y_train)  # 0.9199

# Train ve Test scoreları arasında fark ne kadar az ise o kadar uyumlu öğremiştir. Burada hem test ettiklerimizin hem de train ettiklerimizin scoruna baktık. Underfitting durumunda, model eğitim verileri üzerinde düşük performans gösterirken, test verileri üzerinde de düşük performans gösterir. Overfitting durumunda ise, model eğitim verileri üzerinde yüksek performans gösterirken, test verileri üzerinde düşük performans gösterir.

df.info()


neighboor = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
y_result = neighboor.predict(X_test)
print(y_result)

print(f'Test: {metrics.accuracy_score(y_test, neighboor.predict(X_test))}')  # 0.02
print(f'Train: {metrics.accuracy_score(y_train, neighboor.predict(X_train))}')  # 0.10

k_katsayisi = int(input("Katsayı giriniz: "))
array_lenght = k_katsayisi - 1
jsi_acc = np.zeros(array_lenght)
std_acc = np.zeros(array_lenght)

for k in range(1, k_katsayisi):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_predict = neigh.predict(X_test)
    print(y_predict)

    jsi_acc[k - 1] = metrics.accuracy_score(y_test, y_predict)
    std_acc[k - 1] = np.std(y_predict == y_test) / np.sqrt(y_predict.shape[0])


print(f'jsi_acc: {jsi_acc}')
print(f'std_acc: {std_acc}')


plt.figure(figsize=(10,7))
plt.plot(range(1, k_katsayisi), jsi_acc, color='green')
plt.fill_between(range(1, k_katsayisi), jsi_acc + 1 * std_acc, alpha=0.38, color='blue')
plt.legend(('accuracy', 'std'), prop={'size': 8})
plt.xlabel('Number of Neighboors', color='r')
plt.ylabel('Accuracy', color='r')
plt.grid()
plt.tight_layout()
plt.show()

print(f'En iyi doğruluk değeri {jsi_acc.max()}, k_sırası = {jsi_acc.argmax()+1}')
# endregion