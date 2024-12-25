

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree  # plot_tree fonksiyonu bir karar ağacının görselleştirilmesini sağlar
from sklearn import preprocessing, metrics, tree
from six import StringIO
import pydotplus
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier  # ensemble modülünden RandomForestClassifier'ı çağırdık
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate



df = pd.read_csv('Data/drug200.csv')
print(df.head().to_string())


# NOT: Entropy ve Gini genellikle benzer sonuçlar üretir. Bu algoritmanın uygulanmasına ve veri setine göre değişebilir. Eğer criterion= verilmezse default olarak Gini hesaplaması yapılır..!


# region Hangi ilaçları kullanacaklarını seçmek için karar ağacı yap ve en son hangi etkenle daha önemli önem sırasına göre sırayıp göster
df.columns

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X)

# Yukarıda oluşturduğumuz X features matrix'inde Sex, BP, Cholestrol gibi features'ların değerleri categorical yani sözel ifadeler olduğunu görebiliriz. Bu çalışmada karar ağacının algoritmasında entropy hesabı yapacağımız için yani aritmatiksel işlemler yapacağımız için bu categorical değerlerden faydalanmalıyız. İlgili features'leri scaler büyüklüklere dönüştüreceğiz.

df['Sex'].value_counts()
le_Sex = preprocessing.LabelEncoder().fit(['F','M'])  # LabelEncoder() 0 ve 1 verir. Buradaki fit() F ve M yani isimlerine 0 ve 1 gelecek olan isimleri belirtir.
X[:,1] = le_Sex.transform(X[:,1])  # Burada Sex adlı sütunu aldık ve onu dönüştürmesini istedik (0 ve 1)
print(X)


le_BP = preprocessing.LabelEncoder().fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])
print(X)


le_Cholesterol = preprocessing.LabelEncoder().fit(['NORMAL', 'HIGH'])
X[:,3] = le_Cholesterol.transform(X[:,3])
print(X)

# NOT: OneHotEncoder ile LabelEncoder() benzer işleve sahip fakat LabelEncoder() sınıf türünde döndürür o yüzden ML algoritmalarında daha mantıklı..!

X = preprocessing.StandardScaler().fit_transform(X)  # Entropide 0 yutan eleman olduğu için bütün datayı Normalizasyon yaptık.
print(X)

y = df['Drug'].values
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Decision Tree algoritmasında oluşturulan yapraklarda (node) bilgi kazancına bakılacaksa 'entropy' yada 'log_loss' yani bir başka değişle Shanon İnformation gain yapılırken, yapraktaki saflığa bakılacaksa 'gini' hesaplaması yapılır.

drug_tree_entropy = DecisionTreeClassifier(criterion='entropy')
drug_tree_entropy.fit(X_train, y_train)
prediction_entropy = drug_tree_entropy.predict(X_test)


print(f'Decision Tree Accuracy Score (entropy): {metrics.accuracy_score(y_test, prediction_entropy)}')


drug_tree_gini = DecisionTreeClassifier(criterion='gini')
drug_tree_gini.fit(X_train, y_train)
prediction_gini = drug_tree_gini.predict(X_test)


print(f'Decision Tree Accuracy Score (gini): {metrics.accuracy_score(y_test, prediction_gini)}')


# Özelliklerin önem sırasını belirleme
importance_entropy = drug_tree_entropy.feature_importances_
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

importance_df_entropy = pd.DataFrame({'Özellikler':feature_names, 'Önemlilik Derecesi': importance_entropy}).sort_values('Önemlilik Derecesi', ascending=False)
print(importance_df_entropy)


importance_gini = drug_tree_gini.feature_importances_
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

importance_df_gini = pd.DataFrame({'Özellikler':feature_names, 'Önemlilik Derecesi': importance_gini}).sort_values('Önemlilik Derecesi', ascending=False)
print(importance_df_gini)



plt.figure(figsize=(10, 7))
plt.barh(importance_df_gini['Özellikler'], importance_df_gini['Önemlilik Derecesi'], color='skyblue')
plt.xlabel('Önemlilik Derecesi')
plt.ylabel('Özellikler')
plt.title('Özelliklerin Önem Sırası')
plt.gca().invert_yaxis()  # En yüksek öneme sahip özelliklerin en üstte görünmesi için yatay eksenin sıralamasını tersine çevirme
plt.show()
# endregion



# region Veri setindeki kişilerin özelliklerine dayanarak hangi ilacı reçete edeceğimizi tahmin etmek için bir karar ağacı modeli oluştur
df.columns

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X)

y = df['Drug'].values
print(y)


le_sex = preprocessing.LabelEncoder().fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])
print(X)

le_bp = preprocessing.LabelEncoder().fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_bp.transform(X[:,2])
print(X)

le_cholesterol = preprocessing.LabelEncoder().fit(['NORMAL','HIGH'])
X[:,3] = le_cholesterol.transform(X[:,3])
print(X)


X = preprocessing.StandardScaler().fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

karar_agaci_entropy = DecisionTreeClassifier(criterion='entropy')
karar_agaci_entropy.fit(X_train,y_train)
entropy_tahmini = karar_agaci_entropy.predict(X_test)
print(entropy_tahmini)
print(f'Karar ağacı doğruluk skoru (entropy): {metrics.accuracy_score(y_test, entropy_tahmini)}')


karar_agaci_gini = DecisionTreeClassifier(criterion='gini')
karar_agaci_gini.fit(X_train, y_train)
gini_tahmini = karar_agaci_gini.predict(X_test)
print(gini_tahmini)
print(f'Karar ağacı doğruluk skoru tahmini (gini): {metrics.accuracy_score(y_test, gini_tahmini)}')


importance_entropy = karar_agaci_entropy.feature_importances_
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

importance_df_entropy = pd.DataFrame({'Özellikler':feature_names, 'Önemlilik Derecesi': importance_entropy}).sort_values('Önemlilik Derecesi', ascending=False)
print(importance_df_entropy)


importance_gini = karar_agaci_gini.feature_importances_
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

importance_df_gini = pd.DataFrame({'Özellikler':feature_names, 'Önemlilik Derecesi': importance_gini}).sort_values('Önemlilik Derecesi', ascending=False)
print(importance_df_gini)


plt.figure(figsize=(10, 7))
plt.barh(importance_df_entropy['Özellikler'], importance_df_entropy['Önemlilik Derecesi'], color='skyblue')
plt.xlabel('Önemlilik Derecesi')
plt.ylabel('Özellikler')
plt.title('Özelliklerin Önem Sırası')
plt.gca().invert_yaxis()
plt.show()
# endregion



# region veri setindeki kişilerin yaşlarına göre hangi ilacı kullanacaklarını tahmin etmek için bir karar ağacı modeli oluştur
df.columns

X = df[['Age']].values
print(X)

y = df['Drug'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

decision_tree_entropy = DecisionTreeClassifier(criterion='entropy')
decision_tree_entropy.fit(X_train, y_train)
decision_tree_entropy_prediction = decision_tree_entropy.predict(X_test)
print(decision_tree_entropy_prediction)
print(f'Decision tree entropy score: {metrics.accuracy_score(y_test, decision_tree_entropy_prediction)}')


decision_tree_gini = DecisionTreeClassifier(criterion='gini')
decision_tree_gini.fit(X_train, y_train)
decision_tree_gini_prediction = decision_tree_gini.predict(X_test)
print(decision_tree_gini_prediction)
print(f'Decision tree gini score: {metrics.accuracy_score(y_test, decision_tree_gini_prediction)}')



# Decision Tree modelinden özellik önemlilik değerlerini al
importance_entropy = decision_tree_entropy.feature_importances_
importance_gini = decision_tree_gini.feature_importances_
feature_names = ['Age']  # Sadece 'Age' özelliği olduğu için bu özelliğin adını kullanıyoruz

# Özelliklerin önem sırasını DataFrame'e dönüştür
importance_df = pd.DataFrame({'Özellikler': feature_names,
                                'Entropy Önemlilik Derecesi': importance_entropy,
                                'Gini Önemlilik Derecesi': importance_gini})

print(importance_df)


# Görselleştirme
plt.figure(figsize=(10, 7))
plt.barh(importance_df['Özellikler'], importance_df['Entropy Önemlilik Derecesi'], color='skyblue', label='Entropy')
plt.barh(importance_df['Özellikler'], importance_df['Gini Önemlilik Derecesi'], color='salmon', label='Gini')
plt.xlabel('Önemlilik Derecesi')
plt.ylabel('Özellikler')
plt.title('Özelliklerin Önem Sırası')
plt.legend()
plt.gca().invert_yaxis()
plt.show()
# endregion



# region kişilerin yaşlarına göre hangi ilacı kullandıklarının tahminin yapıp görselleştir
le_sex = LabelEncoder().fit(df['Sex'])
le_bp = LabelEncoder().fit(df['BP'])
le_cholesterol = LabelEncoder().fit(df['Cholesterol'])

df['Sex'] = le_sex.transform(df['Sex'])
df['BP'] = le_bp.transform(df['BP'])
df['Cholesterol'] = le_cholesterol.transform(df['Cholesterol'])


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(f"Karar Ağacı Doğruluk Skoru: {metrics.accuracy_score(y_test, y_pred)}")

# Tahminlerin bölgesel dağılımını görselleştirme
plt.figure(figsize=(10, 7))
sns.countplot(x='Age', hue='Drug', data=df)
plt.title('Yaş Aralıklarına Göre İlaç Kullanımı')
plt.xlabel('Yaş Aralığı')
plt.ylabel('Sayı')
plt.legend(title='İlaç')
plt.xticks(rotation=45)
plt.show()
# endregion



# region Yaş değişkeninin ilaç seçimi üzerindeki etkisini görselleştirmip karar ağacı üzerinde görselleştir
plt.figure(figsize=(10, 7))
df.groupby('Age')['Drug'].value_counts().unstack().plot(kind='bar', stacked=True)  # unstack() fonksiyonuyla çok endeksli bir Seriyi veya DataFrame'i yeniden yapılandırır. Bu, yaş gruplarına göre ilaç sayılarını içeren bir tablo oluşturur. Yani indexleri sütuna çevirerek daha rahat ve anlaşışır bir görsellik sunar.
plt.xlabel('Yaş')
plt.ylabel('İlaç Sayısı')
plt.title('Yaşa Göre İlaç Seçimi')
plt.legend(title='İlaç')
plt.xticks(rotation=0)
plt.show()


X = df[['Age']].values
y = df['Drug'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


decision_tree = DecisionTreeClassifier(random_state=42)  # criterion= belirtmezsen default olarak 'gini' katsayısı kullanılır..!
decision_tree.fit(X_train, y_train)


plt.figure(figsize=(15, 10))
plot_tree(decision_tree, filled=True, feature_names=['Age'], class_names=decision_tree.classes_)
plt.title('Karar Ağacı')
plt.show()
# filled=True: Karar ağacı düğümlerinin renginin sınıfın çoğunluk değerine göre doldurulmasını sağlar.
# feature_names=['Age']: Karar ağacında kullanılan özelliklerin isimlerini belirtir. Burada sadece 'Age' özelliği kullanıldığı için sadece bu özelliğin adı verilmiştir
# class_names=decision_tree.classes_: Sınıf adlarını belirtir. Karar ağacının sınıf adları, modelin .classes_ özelliği aracılığıyla alınmıştır. Bu sayede her sınıfın adı doğru bir şekilde gösterilir.

y_pred = decision_tree.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Test Doğruluğu:", accuracy)
# endregion



# region BP ve Cholesterol değişkeninin ilaç seçimi üzerindeki etkisini görselleştirip karar ağacı üzerinde görselleştir
plt.figure(figsize=(10,7))
df.groupby(['BP', 'Cholesterol'])['Drug'].value_counts().unstack().plot(kind='bar', stacked=True)
plt.xlabel('BP', color='r')
plt.ylabel('İlaç Sayısı', color='r')
plt.suptitle('BP sonucuna Göre İlaç Seçimi', color='r')
plt.legend(title='İlaç', prop={'size': 8})
plt.xticks(rotation=360)
plt.show()

X = df[['BP','Cholesterol']].values
print(X)

y = df['Drug'].values
print(y)

le_BP = preprocessing.LabelEncoder().fit(['LOW','NORMAL','HIGH'])
X[:,0] = le_BP.transform(X[:,0])
print(X)

le_Cholesterol = preprocessing.LabelEncoder().fit(['NORMAL', 'HIGH'])
X[:, 1] = le_Cholesterol.transform(X[:,1])
print(X)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.2)

decision_tree_entropy = DecisionTreeClassifier(criterion='entropy')
decision_tree_entropy.fit(X_train, y_train)
decision_tree_entropy_prediction = decision_tree_entropy.predict(X_test)
print(decision_tree_entropy_prediction)
print(f'Decision Tree Entropy Score: {metrics.accuracy_score(y_test, decision_tree_entropy_prediction)}')


decision_tree_gini = DecisionTreeClassifier(criterion='gini')
decision_tree_gini.fit(X_train, y_train)
decision_tree_gini_prediction = decision_tree_gini.predict(X_test)
print(decision_tree_gini_prediction)
print(f'Decision Tree Gini Score: {metrics.accuracy_score(y_test, decision_tree_gini_prediction)}')



importance_entropy = decision_tree_entropy.feature_importances_
feature_names = ['BP', 'Cholesterol']
importance_df_entropy = pd.DataFrame({'Özellikler':feature_names, 'Önemlilik Derecesi': importance_entropy}).sort_values('Önemlilik Derecesi', ascending=False)
print(importance_df_entropy)


importance_gini = decision_tree_gini.feature_importances_
feature_names = ['BP','Cholesterol']
importance_df_gini = pd.DataFrame({'Özellikler':feature_names, 'Önemlilik Derecesi': importance_gini}).sort_values('Önemlilik Derecesi', ascending=False)
print(importance_df_gini)


plt.figure(figsize=(10, 14))

plt.subplot(2,1,1)
plot_tree(decision_tree_entropy, filled=True, feature_names=['BP','Cholesterol'], class_names=decision_tree_entropy.classes_)
plt.title('Decision Tree Entropy', color='r')
plt.show()

plt.subplot(2,1,2)
plot_tree(decision_tree_gini, filled=True, feature_names=['BP','Cholesterol'], class_names=decision_tree_gini.classes_)
plt.title('Decision Tree Gini', color='r')
plt.show()
# endregion



# region random forest ile kişilerin hangi ilacı kullanacağını belirleyip tahmin yap
df.columns

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X)

y = df['Drug'].values
print(y)


le_sex = preprocessing.LabelEncoder().fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])
print(X)

le_BP = preprocessing.LabelEncoder().fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])
print(X)

le_Cholesterol = preprocessing.LabelEncoder().fit(['NORMAL','HIGH'])
X[:,3] = le_Cholesterol.transform(X[:,3])
print(X)

X = preprocessing.StandardScaler().fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

random_forrest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=6)  # n_estimators= Ağaç sayısını belirler. Daha fazla ağaç daha fazla çeşitlilik ve iyi sonuç demektir ancak daha fazla hesaplama maliyeti olur.
# criterion= Ağaçların bölünme kriterini belirler. "gini" veya "entropy" olabilir.
# max_depth= Her ağacın maksimum derinliğini sınırlar. Ağaçların derinleşmesini önler, bu da aşırı uydurmaya karşı koruma sağlayabilir.
# n_jobs= Model eğitimi sırasında kullanılacak iş parçacığı sayısını belirler. Eğer -1 olarak ayarlanırsa, tüm işlemciler kullanılır.
random_forrest.fit(X_train, y_train)
y_prediction = random_forrest.predict(X_test)
print(y_prediction)

print(f'Modelin doğruluğunun skoru: {metrics.accuracy_score(y_test, y_prediction)}')



plt.figure(figsize=(10, 7))
plt.bar(y_test, metrics.accuracy_score(y_test, y_prediction))
plt.xlabel('Sınıflar')
plt.ylabel('Doğruluk')
plt.title('Sınıf Doğruluk Oranları')
plt.xticks(rotation=45)
plt.show()
# endregion