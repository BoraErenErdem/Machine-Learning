import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştırma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge regresyonu modelini oluşturma
ridge_model = Ridge()

# GridSearchCV ile alpha parametresini optimize et
param_grid = {'alpha': np.logspace(-6, 6, 13)}  # Logaritmik bir param_grid aralığı belirledim çünkü Ridge ve Lasso düzenlileştirmeler,nde genellikle logaritmik aralıklar kullanılır. Bu yüzden np.geomspace() yerine np.logspace() kullandım..!

grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# GridSearchCV ile modelin en iyi alpha değerini bul
grid_search.fit(X_train_scaled, y_train)

# En iyi alpha parametresive model skoru
best_alpha = grid_search.best_params_['alpha']
print("En iyi alpha değeri:", best_alpha)
print("En iyi modelin skoru:", np.sqrt(-grid_search.best_score_))  # MSE'nin kökünü alarak RMSE'yi gösterdim

# Test seti ile değerlendirme
best_ridge_model = grid_search.best_estimator_  # En iyi modeli al
y_pred = best_ridge_model.predict(X_test_scaled)

# Test seti için RMSE hesaplama
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
print("Test seti RMSE:", rmse_test)

# Cross-validation ile modelin doğruluğunu değerlendirilmesi
cross_val_scores = cross_val_score(best_ridge_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_rmse = np.sqrt(-cross_val_scores.mean())  # MSE'nin kökünü alarak RMSE'yi al
print("Cross-validation ortalama RMSE:", mean_cv_rmse)


# Görselleştirme
plt.figure(figsize=(14, 6))

# Alpha değerlerine göre model performansını görselleştirme
alphas = np.logspace(-6, 6, 13)
rmse_scores = [np.sqrt(-score) for score in grid_search.cv_results_['mean_test_score']]
plt.subplot(1, 2, 1)
plt.plot(alphas, rmse_scores, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel("Alpha Değerleri (log scaled)")
plt.ylabel("Cross-validation RMSE")
plt.title("Alpha'ya Göre Ridge Regresyon RMSE Değerleri")
plt.grid(True)

# Test seti tahminleri ve gerçek değerler arasındaki ilişkiyi görselleştirme
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Test Seti: Gerçek vs. Tahmin Edilen Değerler")
plt.tight_layout()

plt.show()