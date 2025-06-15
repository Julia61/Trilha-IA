# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np

# %% [markdown]
# ### Carregar e visualizar os dados - Regressão

# %%
# Carregar os dados - Regressão
df = pd.read_csv('./dataset/dataset_colesterol.csv')

# %%
# Visualiar estrutura dos dados
df.info()

# %%
# Visualizar DataFrame
df.head(10)

# %%
# Ajustar DataFrame
df.drop('Id', axis=1, inplace=True)

# %%

# Aplicar OneHotEncoding nas variáveis categoricas
df = pd.get_dummies(df, columns=['Grupo Sanguíneo', 'Fumante', 'Nível de Atividade'])

# %%
# DataFrame atualizado
df

# %%
df.info()

# %% [markdown]
# ### Treinar Modelo de Regressão com Lasso (L1)

# %%
# Separar X e y
X = df.drop('Colesterol', axis=1)
y = df['Colesterol']

# %%
# Separar Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# %%
# Treinar o Modelo de regressão linear múltipla com Lasso
# Quanto maior o alpha, maior a penalização e mais coeficientes tendem a ser reduzios a zero
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_train, y_train)

# %%
# Mostrar importância de Features
def importancia_features(modelo):
    importance = np.abs(modelo.coef_)
    print('Importância das Features')
    for i, feature in enumerate(modelo.feature_names_in_):
        print(f'{feature}: {importance[i]}')

# %%
# Importância das Features - Lasso
importancia_features(model_lasso)

# %%
# Função para a avaliar performance
def performace_regressao(modelo, X_test, y_test):
    # Faz a predição com o modelo no conjunto de testes
    y_pred = modelo.predict(X_test)
    # Avaliar desempenho
    return root_mean_squared_error(y_test, y_pred)

# %%
# Performance Regressão com Lasso
performace_regressao(model_lasso, X_test, y_test)

# %% [markdown]
# ### Treinar com LassoCV

# %%
model_lasso_cv = LassoCV(alphas=[0.1, 0.5, 1], cv=5, random_state=51)
model_lasso_cv.fit(X, y)

# %%
# Importância das Features - LassoCV
importancia_features(model_lasso_cv)

# %%
# Performance Regressão com LassoCV
performace_regressao(model_lasso_cv, X_test, y_test)

# %% [markdown]
# ### Treinar Modelo de Regressão com Ridge (L2)

# %%
# Treinar o Modelo de regressão linear múltipla com Lasso
# Quanto maior o alpha, maior a penalização e mais coeficientes tendem a ser reduzios a zero
model_ridge = Ridge(alpha=0.1)
model_ridge.fit(X_train, y_train)

# %%
# Importância das Features - Lasso
importancia_features(model_ridge)

# %%
# Performance Regressão com Lasso
performace_regressao(model_ridge, X_test, y_test)

# %%
# Treinar com RidgeCV
model_ridge_cv = RidgeCV(alphas=[0.1, 0.5, 1], cv=5)
model_ridge_cv.fit(X, y)

# %%
# Importância das Features - RidgeCV
importancia_features(model_ridge_cv)

# %%
# Performance Regressão com RidgeCV
performace_regressao(model_ridge_cv, X_test, y_test)


