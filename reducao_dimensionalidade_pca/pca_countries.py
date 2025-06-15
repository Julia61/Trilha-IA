# %%

# EDA e Visualização de Dados
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

# ML
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# %% [markdown]
# ### Carregar os Dados

# %%
# Carregar DataFrame
df_countries = pd.read_csv('./datasets/Country_Data.csv', sep=';', decimal=',')

# %%
# Analisar a estrutura
df_countries.info()

# %%
# Visualizar os primeiros registros
df_countries.head(10)

# %%
# Visualizar os últimos registros
df_countries.tail(10)

# %% [markdown]
# ### EDA

# %% [markdown]
# ### Análise Univariada

# %%
# Estatística das variáveis
df_countries.describe()

# %%
# Distribuição da variável income_category
percentual_income_category = df_countries.value_counts('income_category') / len(df_countries) * 100
px.bar(percentual_income_category, color=percentual_income_category.index)

# %%
# Distribuição da variável income
px.histogram(df_countries['income'], title='Histograma da variável Income')

# %%
# Distribuição da variável GDPP (PIB per capita)
px.histogram(df_countries['gdpp'], title='Histograma da variável GDPP')

# %%
# Distribuição da variável inflation
px.histogram(df_countries['inflation'], title='Histograma da variável Inflation')

# %%
# Distribuição da variável Life Expectarion
px.histogram(df_countries['life_expec'], title='Histograma da variável Life Expectarion')

# %% [markdown]
# ### Análise Bivariada

# %%
# Plot de Distribuição (BoxPlot) por income e income_category
px.box(df_countries, x='income', y='income_category', color='income_category', orientation='h', hover_data=['country'])

# %%
# Plot de Distribuição (BoxPlot) por gdpp e income_category
px.box(df_countries, x='gdpp', y='income_category', color='income_category', orientation='h', hover_data=['country'])

# %%
# Plot de Distribuição (BoxPlot) por inflation e income_category
px.box(df_countries, x='inflation', y='income_category', color='income_category', orientation='h', hover_data=['country'])

# %%
# Plot de Distribuição (BoxPlot) por life_expec e income_category
px.box(df_countries, x='life_expec', y='income_category', color='income_category', orientation='h', hover_data=['country'])

# %%
# Scatterplot de Income e GDPP
px.scatter(df_countries, x='income', y='gdpp', color='income_category', hover_data=['country'])

# %%
# Gerar Matriz de Correlação
matiz_correlacao_countries = df_countries.corr(numeric_only=True)

# %%
# Plotar Matriz de Correlação
sns.heatmap(matiz_correlacao_countries, vmin=-1, vmax=1, annot=True)

# %% [markdown]
# ### Treinar o algoritmo PCA

# %%
# Selecionar as colunas para PCA
X = df_countries.copy()

# Remover colunas desnecessárias
X.drop(columns=['country', 'income_category'], axis=1, inplace=True)

# %%
# Separar variáveis quantitativas
numeric_features = ['child_mort', 'exports', 'health', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']

# %%
# Definir tranformações
numeric_tranformer = StandardScaler()

# %%
# Criar Pre Processador de Tranformações
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_tranformer, numeric_features)
    ]
)

# %%
# Tranformar os dados
X_tranformed = preprocessor.fit_transform(X)

# %%
# Visualizar X_tranformed
X_tranformed

# %%
# Matriz de covariância
np.cov(X_tranformed)

# %%
# Criar modelo PCA
modelo_pca = PCA(n_components=2)

# %%
# Executar PCA
X_pca = modelo_pca.fit_transform(X_tranformed)

# %%
# Gerar um DataFrame com base nos componetes principais
#df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])


# %%
# Visualizar componentes principais
df_pca.head(10)

# %%
# Verificar tamanho do DataFrame de PCA
len(df_pca)

# %%
# Incluir componentes principais no dataframe original
df_countries['PC1'] = df_pca['PC1']
df_countries['PC2'] = df_pca['PC2']
#df_countries['PC3'] = df_pca['PC3']

# %%
# Mostrar DataFrame original com Componetes Principais
df_countries.head(10)

# %% [markdown]
# ### Visualizar Resultados

# %%
# Autovalores 
autovetores = modelo_pca.explained_variance_

# Autovetores
autovalores = modelo_pca.components_

print(f'Autovetores:')
print(autovetores)
print("\nAutovalores:")
print(autovalores)

# %%
# Mostrar Chart 3D com os componentes principais
fig = px.scatter_3d(df_countries, x='PC1', y='PC2', z='PC3', color='income_category',
                    title='Visualização PCA', width=800, height=600, hover_data=['country'])
fig.show()

# %%
# Gráfico 2D (PCA com 2 componentes principais)
px.scatter(df_countries, x='PC1', y='PC2', color='income_category', hover_data=['country'])

# %%
# Reconstruir os dados com base no PCA
X_recovered = modelo_pca.inverse_transform(X_pca)

# %%
# Visualizar Recovered
X_recovered.shape[1]

# %%
# Calcular o erro de reconstrução
reconstruction_error = mean_squared_error(X_tranformed, X_recovered)
print("Erro de Reconstrução", reconstruction_error)

# %% [markdown]
# ### Salvar Modelo

# %%
import joblib

# Salvar Modelo
joblib.dump(modelo_pca, './modelo_pca_countries.pkl')

# Salvar Preprocessor
joblib.dump(preprocessor, './preprocessor_pca_countries.pkl')




