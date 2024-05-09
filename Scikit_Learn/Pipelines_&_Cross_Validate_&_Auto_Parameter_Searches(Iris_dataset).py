

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing, metrics, tree
from six import StringIO
import pydotplus
import matplotlib.image as mpimg
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import cross_validate  # cross_validate yani çapraz sorgulama temel olarak, mevcut veri setini eğitim ve test setlerine ayırmak yerine, veri setini birden fazla alt küme (katman) halinde böler ve her bir alt küme için modelin performansını değerlendirir. Bu, modelin genel performansını daha güvenilir bir şekilde ölçmeye yardımcı olur. Ayrıca cross_validate aşırı uyum sağlanmasını engeller.
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris  # (data)
from sklearn.pipeline import make_pipeline  # Veri işleme ve modelleme süreçlerini bir araya getirerek bir dizi adımı birleştiren kullanışlı bir araçtır. Pipeline'lar, her bir adım arasında veri sızıntısını önlemek için otomatik olarak geçici belleği temizler. Bu, özellikle çapraz doğrulama gibi tekniklerle çalışırken modelin gerçek performansını daha doğru bir şekilde değerlendirmeye yardımcı olur. Özetle  model geliştirme sürecini daha düzenli, yeniden kullanılabilir ve daha az hata yapmaya eğilimli hale getiren güçlü bir araçtır.
from sklearn.datasets import fetch_california_housing  # (data)
from sklearn.model_selection import RandomizedSearchCV  # Rastgele değerler kullanılır. # belirli bir hiperparametre aralığında rastgele seçilen kombinasyonları değerlendirir ve en iyi performansı sağlayan hiperparametreleri bulur
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor



# Iris_dataset & pipelines
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)  # şu an test_size vermediğimiz için ön tanımlı olarak % 75 train, % 25 test olarak veri setini parçaladı.

pipe = make_pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())

X, y = load_iris(return_X_y=True)  # girdi ve çıktı değişkenleri oluşturuyorum

pipe.fit(X_train, y_train)

accuracy_score(pipe.predict(X_test), y_test)




# cross_validate (çapraz doğrulama)
X, y = make_regression(n_samples=1000, random_state=0)  # n_samples= örneklem sayısıdır.

regression = linear_model.LinearRegression()
result = cross_validate(regression, X, y)
print(result['test_score'])



# Automatic Parameter Searches (otomatik pararmetre seçimi)
# modellerin ayaralanabilir parametrelerine 'hiper parametre' denir

X,y = fetch_california_housing(return_X_y=True)  # girdi ve çıktı değişkenleri oluşturuyorum

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

param_distributions = {'n_estimators': randint(1,5), 'max_depth': randint(5,10)}

search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5, param_distributions=param_distributions, random_state=0)  # estimator= sklearn'de makina öğrenmesi algoritmalarına estimator denir..!

search.fit(X_train, y_train)  # en iyi model için parametreleri gördük

search.best_params_  # Outputs: 'max_depth': 9 (yani en iyi model için max derinlik 9), 'n_estimators': 4

search.score(X_test, y_test)