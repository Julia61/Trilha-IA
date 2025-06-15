# %%
import pandas as pd
import seaborn as sns
import ppscore as pps
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

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

# %% [markdown]
# ### Calcular PPS - Regressão

# %%
# Calcular PPS entre as variáveis
pps_matrix_reg = pps.matrix(df)
pps_matrix_reg

# %%
# Ajustar Matriz para fazer o Plot
pps_matrix_reg_pivot = pps_matrix_reg[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
pps_matrix_reg_pivot

# %%
# Plotar a matriz de PPS
plt.figure(figsize=(10,8))
sns.heatmap(pps_matrix_reg_pivot, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Predictive Power Score (PPS) - Modelo Colesterol')
plt.show()

# %%
# Calcular PPS entre 2 variáveis específicas
pps.score(df, 'Peso', 'Colesterol')

# %% [markdown]
# ### Matriz de Correlação - Regressão

# %%
# Calcular a matriz de correlação
corr_matrix_reg = df.corr()
corr_matrix_reg

# %%
# Plotar a matriz de PPS
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix_reg, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação - Modelo Colesterol')
plt.show()

# %% [markdown]
# ### Treinar Modelo Linear - Regressão

# %%
# Separar X e y
X = df.drop('Colesterol', axis=1)
y = df['Colesterol']

# %%
# Treinar modelo regressão linear multipla
model_reg = LinearRegression()
model_reg.fit(X, y)

# %%
# Avaliar a importância das features com base nos coeficientes do modelo
feat_importance_reg = pd.Series(model_reg.coef_, index=X.columns)
feat_importance_reg.plot(kind='barh')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.title('Importância das Features na Regressão Linear')
plt.show()

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
# ### Calcular PPS - Classificação

# %%
# Calcular PPS entre as variáveis
pps_matrix_class = pps.matrix(df2)
pps_matrix_class

# %%
# Ajustar Matriz para fazer o Plot
pps_matrix_class_pivot = pps_matrix_class[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
pps_matrix_class_pivot

# %%
# Plotar a matriz de PPS
plt.figure(figsize=(10,8))
sns.heatmap(pps_matrix_class_pivot, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Predictive Power Score (PPS) - Modelo Qualidade de Frutas')
plt.show()

# %%
# Calcular PPS entre 2 variáveis específicas
pps.score(df2, 'Size', 'Quality')

# %% [markdown]
# ### Matriz de Correlação - Classificação

# %%
# Calcular a matriz de correlação
corr_matrix_class = df2.corr()
corr_matrix_class

# %%
# Plotar a matriz de PPS
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix_class, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação - Modelo Qualidade de Frutas')
plt.show()

# %% [markdown]
# ### Treinar Modelo Regressão Logística

# %%
# Separar X e y
X = df2.drop('Quality', axis=1)
y = df2['Quality']

# %%
# Treinar modelo regressão linear multipla
model_lr = LogisticRegression()
model_lr.fit(X, y)

# %%
# Avaliar a importância das features com base nos coeficientes do modelo
feat_importance_class = pd.Series(model_lr.coef_[0], index=X.columns)
feat_importance_class.plot(kind='barh')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.title('Importância das Features na Regressão Logística')
plt.show()

# %%



