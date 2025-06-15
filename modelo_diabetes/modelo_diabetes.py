# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Carregar arquivo para um DataFrame
df_exames = pd.read_csv('./datasets/exame_diabetes.csv')

# %%
# Visualizar Estrutura
df_exames.info()

# %%
# Apresentar as 10 primeiras linhas
df_exames.head(10)

# %%
df_exames['genero'].unique()

# %% [markdown]
# 

# %%
# Remover coluna id_paciente
df_exames.drop(columns=['id_paciente'], axis=1, inplace=True)

# %%
# Converter uma variável categorica (genero) em númerica, usando One-Hot Enconding
df_exames = pd.get_dummies(df_exames, columns=['genero'], dtype='int64')

# %%
df_exames.head(10)

# %%
# Apresentar Mapa de Calor com Correlação entre as variáveis 
sns.heatmap(df_exames.corr(),vmin=-1, vmax=1, annot=True)

# %%
# Mapa de Correlação só com a variável Target (resultado)
sns.heatmap(df_exames.corr()[['resultado']].sort_values(by='resultado', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

# %%
# Plot de Scatter (Dispersão) com Distribuição
pd.plotting.scatter_matrix(df_exames, alpha=0.2, figsize=(6,6), diagonal='kde')

# %%
# Histograma de todas as variáveis
df_exames.hist(layout=(2,4), figsize=(10,5))

# %%
# Criar uma Feature nova
# IMC (Índice de Massa Corporal)
# IMC = peso (Kg) / altura (m) ^2
df_exames['imc'] = (df_exames['peso'] / ((df_exames['altura']/100)**2))

# %%
df_exames.head(10)

# %%
df_exames.info()

# %%
dict_regressao = {'tempo_casa': [1, 3, 6, 9, 10, 14, 18], 'salario': [1500, 3000, 4500, 6000, 7000, 8500, 10000]}

# %%
df_refressao_simples = pd.DataFrame.from_dict(dict_regressao)

# %%
sns.regplot(data=df_refressao_simples, x="tempo_casa", y="salario")

# %%
# Importar bibliotecas so sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# %%
# Modelo 1 - Sem IMC
X = df_exames.drop(columns=['imc', 'resultado'])
y = df_exames['resultado']

# %%
X

# %%
# Dividir conjunto entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=51)


# %%
y_test



# %%
# Treinar o algoritmo de Regressão Linear - Modelo 1
model_1 = LinearRegression().fit(X_train, y_train)

# %%
# Gerar Predições do conjunto de testes com base no Modelo 1
y_pred = model_1.predict(X_test)

# %%
# Equação da Reta - Regressão Linear
# y = ax + b
model_1.coef_

# %%
model_1.intercept_

# %%
# R2 Score - Conjunto de Treinamento
model_1.score(X_train, y_train)

# %%
# R2 Score - Conjunto de Testes
model_1.score(X_test,y_test)

# %%
# R2 Score - Testes
r2_score(y_test, y_pred)

# %%
# MAE (Mean Absolute Error)
mean_absolute_error(y_test, y_pred)

# %%
# Segundo Modelo - Apenas IMC
X = pd.DataFrame(df_exames['imc'])
y = df_exames['resultado']

# %%
# Dividir conjunto entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=51)

# %%
# Treinar o algoritmo de Regressão Linear - Modelo 2
model_2 = LinearRegression().fit(X_train, y_train)

# %%
# Gerar Predição com base no modelo para o conjunto de testes
y_pred = model_2.predict(X_test)

# %%
model_2.coef_

# %%
model_2.intercept_

# %%
# R2 Score no Conjuntoi de Treinamento
model_2.score(X_train, y_train)

# %%
# R2 Score no Conjunto de Testes
model_2.score(X_test,y_test)

# %%
# MAE (Mean Absolute Error)
mean_absolute_error(y_test, y_pred)

# %%
# Mostrar como a reta foi calculada
plt.scatter(X_test, y_test, color='g')
plt.plot(X_test, y_pred, color='k')


