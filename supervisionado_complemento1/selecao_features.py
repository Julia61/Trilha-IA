# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, f1_score

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
# ### Treinar modelo de regressão linear multipla com RFE
# 

# %%
# Separar X e y
X = df.drop('Colesterol', axis=1)
y = df['Colesterol']

# %%
# Separar Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# %%
# Treinar o modelo com RFE
# RFE (Recursice Feature Elimination)
# Uso um estimador e defino uma quantidade de features (hiperparâmetros)
# O RFE faz iterações iniciando com todas as features e eliminando a cada iteração até atingir a qtde. definido
# Elimina características/features menos importantes
rfe_method = RFE(estimator=LinearRegression(), n_features_to_select=6)
rfe_method.fit(X_train,y_train)

# %%
# Quais features foram selecionadas
X_train.columns[(rfe_method.get_support())]

# %%
# Ranking de Features
def mostrar_ranking(metodo_fs, X_train):

    # Obter o Ranking de Features
    ranking = rfe_method.ranking_

    # Obter os nomes das Features
    nomes_features = X_train.columns.to_list()

    # Criar um DataFrame com os rankings e os nomes das features
    df_ranking = pd.DataFrame({'Feature': nomes_features, 'Ranking': ranking})

    # Ordene o DataFrame pelo ranking
    df_ranking = df_ranking.sort_values(by='Ranking')

    # Exibir Ranking
    print(df_ranking)


# %%
# Ranking de Features do RFE Regressão
mostrar_ranking(rfe_method, X_train)

# %%
# Função para a avaliar performance
def performace_regressao(modelo, X_test, y_test):
    # Faz a predição com o modelo no conjunto de testes
    y_pred = modelo.predict(X_test)
    # Avaliar desempenho
    return root_mean_squared_error(y_test, y_pred)

# %%
performace_regressao(rfe_method, X_test, y_test)

# %% [markdown]
# ### Treinar Modelo sem RFE

# %%
# Treinar Modelo de Regressão sem RFE
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)


# %%
# Performance Regressão sem RFE
performace_regressao(model_reg, X_test, y_test)

# %% [markdown]
# ### Treinar Modelo de Regressão Linear com RFECV

# %%
rfe_method_cv = RFECV(estimator=LinearRegression(), min_features_to_select=6, cv=5)
rfe_method_cv.fit(X_train, y_train)

# %%
performace_regressao(rfe_method_cv, X_test, y_test)

# %%
# Quais features foram selecionadas
X_train.columns[(rfe_method_cv.get_support())]

# %%
# Quantas features foram selecionadas
rfe_method_cv.n_features_

# %% [markdown]
# ### Treinar Modelo  de Regressão com SelectFromModel

# %%
sfm_method = SelectFromModel(estimator=model_reg, max_features=4, threshold=0.5)
sfm_method.fit(X_train, y_train)

# %%
# Quais features foram selecionadas
X_train.columns[(sfm_method.get_support())]

# %%
# Treinar modelo com as features selecionadas
X_train_ajustado_reg = sfm_method.transform(X_train)
X_test_ajustado_reg = sfm_method.transform(X_test)
model_reg.fit(X_train_ajustado_reg, y_train)

# %%
# Performance do Modelo com SelectFromModel
performace_regressao(model_reg, X_test_ajustado_reg, y_test)

# %% [markdown]
# ### Carregar e preparar dados - Classificação

# %%
# Carregar o dataset
df2 = pd.read_csv('./dataset/fruit_quality.csv')

# %%
# Visualizar estrutura
df2.info()

# %%
# Visualizar DF
df2.head(10)

# %%
# Ajustar DataFrame

# Remover a coluna A_id
df2.drop('A_id', axis=1, inplace=True)

df2['Quality'] = (df2['Quality'] == 'good').astype(int)

df2

# %% [markdown]
# ### Carregar e preparar dados - Classificação

# %%
# Carregar o dataset
df2 = pd.read_csv('./dataset/fruit_quality.csv')

# %%
# Visualizar estrutura
df2.info()

# %%
# Visualizar DF
df2.head(10)

# %%
# Ajustar DataFrame

# Remover a coluna A_id
df2.drop('A_id', axis=1, inplace=True)

df2['Quality'] = (df2['Quality'] == 'good').astype(int)

df2

# %% [markdown]
# ### Treinar modelo de regressão logistica com RFE
# 

# %%
# Separar X e y
X = df2.drop('Quality', axis=1)
y = df2['Quality']

# %%
# Separar Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# %%
# Treinar o modelo com RFE
# RFE (Recursice Feature Elimination)
# Uso um estimador e defino uma quantidade de features (hiperparâmetros)
# O RFE faz iterações iniciando com todas as features e eliminando a cada iteração até atingir a qtde. definido
# Elimina características/features menos importantes
rfe_method = RFE(estimator=LogisticRegression(), n_features_to_select=5)
rfe_method.fit(X_train,y_train)

# %%
# Quais features foram selecionadas
X_train.columns[(rfe_method.get_support())]

# %%
# Ranking de Features do RFE Regressão
mostrar_ranking(rfe_method, X_train)

# %%
# Função para a avaliar performance
def performace_classificacao(modelo, X_test, y_test):
    # Faz a predição com o modelo no conjunto de testes
    y_pred = modelo.predict(X_test)
    # Avaliar desempenho
    return f1_score(y_test, y_pred)

# %%
performace_classificacao(rfe_method, X_test, y_test)

# %% [markdown]
# ### Treinar Modelo sem RFE

# %%
# Treinar modelo sem RFE
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# %%
# Validar Performance
performace_classificacao(model_lr, X_test, y_test)

# %% [markdown]
# ### Treinar Modelo de Regressão Logística com RFECV

# %%
rfe_method_cv = RFECV(estimator=LogisticRegression(), min_features_to_select=4, cv=5, scoring='f1_weighted')
rfe_method_cv.fit(X_train, y_train)

# %%
performace_classificacao(rfe_method_cv, X_test, y_test)

# %%
# Quais features foram selecionadas
X_train.columns[(rfe_method_cv.get_support())]

# %%
# Quantas features foram selecionadas
rfe_method_cv.n_features_

# %% [markdown]
# ### Treinar Modelo  de Regressão logística com SelectFromModel

# %%
sfm_method = SelectFromModel(estimator=model_lr, max_features=5, threshold=0.01)
sfm_method.fit(X_train, y_train)

# %%
# Quais features foram selecionadas
X_train.columns[(sfm_method.get_support())]

# %%
# Treinar modelo com as features selecionadas
X_train_ajustado_class = sfm_method.transform(X_train)
X_test_ajustado_class = sfm_method.transform(X_test)
model_lr.fit(X_train_ajustado_class, y_train)

# %%
# Performance do Modelo com SelectFromModel
performace_classificacao(model_lr, X_test_ajustado_class, y_test)

# %%



