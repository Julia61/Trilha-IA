# %%
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

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
# ### Treinar Modelo Regressão Linear

# %%
# Separar X e y
X = df.drop('Colesterol', axis=1)
y = df['Colesterol']

# %%
# Separar Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# %%
# Treinar Modelo de Regressão sem RFE
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

# %% [markdown]
# ### Apresentar Plots Shapley Values - Regressão

# %%
# Rodar o explainer no conjunto de treinamento
explainer_reg = shap.Explainer(model_reg.predict, X_train)

# %%
# Calcular Shapley values no conjunto de testes
shap_values_reg = explainer_reg(X_test)

# %%
# Plotar a contribuição geral por Shap Values
shap.plots.bar(shap_values_reg)

# %%
# Mostrar 1a instância do conjunto de testes
X_test.iloc[0,:]

# %%
# Plotar os Shap Values para um exemplo específico
shap.plots.waterfall(shap_values_reg[0], max_display=13)

# %%
# Plotar Hetmap Geral
shap.plots.heatmap(shap_values_reg, max_display=13)

# %%
# Plot de Beewswarm Geral
shap.plots.beeswarm(shap_values_reg, max_display=13)

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
# ### Treinar o Modelo de Regressão Logística

# %%
# Separar X e y
X = df2.drop('Quality', axis=1)
y = df2['Quality']

# %%
# Separar Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# %%
# Treinar Modelo de Regressão sem RFE
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# %% [markdown]
# ### Apresentar Plots Shapley Values - Classificação

# %%
# Rodar o explainer no conjunto de treinamento
explainer_class = shap.Explainer(model_lr.predict, X_train)

# %%
# Calcular Shapley values no conjunto de testes
shap_values_class = explainer_class(X_test)

# %%
# Plotar a contribuição geral por Shap Values
shap.plots.bar(shap_values_class)

# %%
# Mostrar 1a instância do conjunto de testes
X_test.iloc[0,:]

# %%
# Plotar os Shap Values para um exemplo específico
shap.plots.waterfall(shap_values_class[0], max_display=13)

# %%
# Plotar Hetmap Geral
shap.plots.heatmap(shap_values_class, max_display=13)

# %%
# Plot de Beewswarm Geral
shap.plots.beeswarm(shap_values_class, max_display=13)


