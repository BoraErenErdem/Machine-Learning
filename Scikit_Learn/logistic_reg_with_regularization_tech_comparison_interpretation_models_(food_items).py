

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve


df = pd.read_csv('Data/food_items.csv')
df.head()


# region EDA preprocessing işlemleri

df.dtypes.value_counts()

# 1.yol iloc[] ile sütunları seçme
feature_columns = df.iloc[:, :-1].columns
feature_columns

# 2.yol select_dtypes ile sütunları seçme
feature_cols = df.select_dtypes(include=['number']).columns
feature_cols

# 3.yol drop() ile istenilen sütunları seçme
feature_col = df.drop('class', axis=1).columns
feature_col


# 1.yol
df.select_dtypes(include=['number']).describe()

# 2.yol
df.iloc[:, :-1].describe()

# 3.yol
df.drop('class', axis=1).describe()


# feature engineering ile class sütun değişkenini kontrolü
# 1.yol
df.select_dtypes(include=['object']).value_counts(normalize=True)  # NOT: normalize=True parametresi her bir değerin oranını döndürerek verinin dağılımını yüzdesel olarak görmeyi sağlar..!

# 2.yol
df.iloc[:, -1:].value_counts(normalize=True)

# class değerlerini görselleştirme
df.iloc[:, -1:].value_counts().plot.bar(figsize=(10,7), color=['blue', 'green', 'red'])

# class sınıfındaki 3 label da dengesizdir. Bu göz önünde bulundurup model oluşturmaya öyle devam edilir.

X = df.drop(columns='class').values
X

y = df['class'].values
y

# MinMaxScaler() ile ölçekleme
min_max_scaler = MinMaxScaler()

X = min_max_scaler.fit_transform(X)
X

print(f'X sütunun aralık kontorlü = {X.min(), X.max()}')

# y sütunu kategorik 3 değere sahip olduğu için LabelEncoder() kullanıldı
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)  # y sütununu oluştururken .values kullandığım için zaten numpy array formatındaydı. Ancak .values kullanılmasaydım o zaman tek boyutlu dizeye indirgenirdi.
y                                   # örnek olarak -> y = label_encoder.fit_transform(y.values.ravel()) ile hem numpy formatına dönüştürülür hem de .ravel() ile çok boyutlu numpy arrayleri                                        (2D, 3D gibi) tek boyutlu şekilde düzleştirir..!

# sınıflar -> 0=In Moderation, 1=Less Often, 2=More Often
np.unique(y, return_counts=True)  # Bu ifade np.unique() kullanılarak dizideki benzersiz ögeleri [0, 1, 2] ve her benzersiz ögelerin kaç defa tekrar ettiğini gösterir.[6649, 5621,  990]
# endregion




# region Logistic Regression modelini L2, L1 ve Elastic Net regularization teknikleriyle oluşturup karşılaştırma, metriclerine bakma ve yorumlama
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y) # stratify=y parametresi veri setindeki target column'da dengesizlik varsa kullanılır. Bu parametre sınıf dağılımını hem train hem de test veri setinde belirtilen oranda koruyarak sınıfların oranlarının korunmasını sağlar.

print(f'Train shapes = {X_train.shape}, {y_train.shape}')
print(f'Test shapes = {X_test.shape}, {y_test.shape}')

# region L2 regularization tekniği ile logistic regression modeli tanımlama
logistic_reg_l2 = LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs', random_state=42, multi_class='multinomial')

logistic_reg_l2.fit(X_train,y_train)
l2_predict = logistic_reg_l2.predict(X_test)

class_probability_distribution_l2 = logistic_reg_l2.predict_proba(X_test[:1, :])[0]  # predict_proba(X_test[:1, :])[0] ifadesi tüm sınıflarda bulunan olasılıkların ilk satır ve sütunu alır..!
class_probability_distribution_l2

# Bütün metricleri görebileceğim bir fonksiyon oluşturdum.
def evaluate_metric(y_test, y_predict):
    results = {}
    results['Accuracy'] = accuracy_score(y_test, y_predict)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_predict)
    results['Precision'] = precision
    results['Recall'] = recall
    results['F1-Score'] = f1_score
    return results

evaluate_metric(y_test, l2_predict)  # Metirclerden anlaşılabileceği gibi class 2 sınıfına (More Often) ait değerler iyi değil. Bu bize dengesiz veri kümesinin olduğunun kanıtıdır.

cm_l2 = confusion_matrix(y_test, l2_predict, normalize='true')
cm_l2

# ConfusionMatrixDisplay() ile karışıklık matrisini görüntüleme
sns.set('talk')
display = ConfusionMatrixDisplay(confusion_matrix=cm_l2, display_labels=logistic_reg_l2.classes_)
display.plot()
plt.show()


# Katsayıları görüntüleme ve yorumlama (interpretation) için görselleştirme
logistic_reg_l2.coef_


def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict


def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals


def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()


coef_dict = get_feature_coefs(logistic_reg_l2, 1, feature_cols)
visualize_coefs(coef_dict)

coef_dict = get_feature_coefs(logistic_reg_l2, 2, feature_cols)
visualize_coefs(coef_dict)
# endregion




# region L1 regularization tekniği ile logistic regression modeli tanımlama. Burada amacım sınıflandırma performansının iyileşip iyileşmeyeceğini görmek
logistic_reg_l1 = LogisticRegression(penalty='l1', max_iter=1000, solver='saga', random_state=42, multi_class='multinomial')

logistic_reg_l1.fit(X_train,y_train)
l1_predict = logistic_reg_l1.predict(X_test)


class_probability_distribution_l1 = logistic_reg_l1.predict_proba(X_test[:1, :])[0]
class_probability_distribution_l1

evaluate_metric(y_test, l1_predict)  # oluşturmuş olduğum fonksiyonu kullandım.

cm_l1 = confusion_matrix(y_test, l1_predict, normalize='true')
cm_l1


sns.set('talk')
display = (ConfusionMatrixDisplay(confusion_matrix=cm_l1, display_labels=logistic_reg_l1.classes_))
display.plot()
plt.show()


logistic_reg_l1.coef_


def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict


def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals


def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()


coef_dict = get_feature_coefs(logistic_reg_l1, 1, feature_cols)
visualize_coefs(coef_dict)

coef_dict = get_feature_coefs(logistic_reg_l1, 2, feature_cols)
visualize_coefs(coef_dict)
# endregion



# Modeller hakkında yorum --->  Modeller karşılaştırıldığında L1 tekniği kullanılan logistic regression'un L2'den çok daha iyi performans verdiği görülür. Bunun sebebi L1'in katsayıları sıfırlaması olabilir. Böylece feature selection yapılmış olup modeli etkilemeyen veya az etkileyen özelliklerin modelden çıkarılması sağlanmış olabilir.



# region Elastic net regularization tekniği ile Logistic Regression oluşturup modeli karşılaştır, metriclerine bak ve yorumla
logistic_reg_elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, l1_ratio=0.5, multi_class='multinomial')

logistic_reg_elastic_net.fit(X_train,y_train)
elastic_net_predict = logistic_reg_elastic_net.predict(X_test)

class_probability_distribution_elastic_net = logistic_reg_elastic_net.predict_proba(X_test[:1, :])[0]
class_probability_distribution_elastic_net

print(f'Elastic net modelinin tahmin ettiği class -> {logistic_reg_elastic_net.predict(X_test[:1, :])[0]}')

def evaluate_scores(y_test, y_predicts):
    values = []
    values.append(f'Accuracy_score = {accuracy_score(y_test,y_predicts)}')
    precision, recall, F1_score, _ = precision_recall_fscore_support(y_test, y_predicts)
    values.append(f'Precision_score = {precision}')
    values.append(f'Recall_score = {recall}')
    values.append(f'F1_score = {F1_score}')
    return values


evaluate_scores(y_test, elastic_net_predict)  # sadece accuracy, precision, recall ve f1_score metriklerine baktım.


def all_average_params(y_test, y_pred):
    for avg in [None,'micro','macro','weighted']:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=avg)
        print(f"Average: {[avg]}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f'____________')

all_average_params(y_test, elastic_net_predict)  # precision, recall ve f1_score metricleri ile beraber çoklu sınıfların (multiclass) alabileceği average parametrelerine bakıp yorumladım..!


cm_elastic_net = confusion_matrix(y_test, elastic_net_predict, normalize='true')
cm_elastic_net

sns.set('talk')
display = ConfusionMatrixDisplay(confusion_matrix=cm_elastic_net, display_labels=logistic_reg_elastic_net.classes_)
display.plot()
plt.show()


logistic_reg_elastic_net.coef_


def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef

    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict


def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals


def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()


coef_dict = get_feature_coefs(logistic_reg_elastic_net, 1, feature_cols)
visualize_coefs(coef_dict)

coef_dict = get_feature_coefs(logistic_reg_elastic_net, 2, feature_cols)
visualize_coefs(coef_dict)
# endregion

# L2, L1 ve Elastic Net regularization tekniği sonucunda bu veri seti için en iyi performansın L1 tekniğinde olduğu görülür. Ancak bu parametrelere ve benimsenen yaklaşım türlerine göre değişkenlik gösterir..!

# endregion