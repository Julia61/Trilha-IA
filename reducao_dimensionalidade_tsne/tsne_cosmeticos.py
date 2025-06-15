# %%
# EDA
import pandas as pd
import plotly.express as px
import seaborn as sns

# ML / tSNE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.manifold import TSNE

# %% [markdown]
# ### Carregar os Dados

# %%
# Carregar DataFrame
df_cosmeticos = pd.read_csv('./datasets/cosmeticos.csv')

# %%
# Visualizar a Estrutura
df_cosmeticos.info()

# %%
# Visualizar os primeiros registros
df_cosmeticos.head(10)

# %%
# Visualizar os últimos registros
df_cosmeticos.tail(10)

# %% [markdown]
# ### EDA

# %% [markdown]
# Tranformar valores da coluna "Ingredientes" em um novo DataFrame

# %%
df_cosmeticos_eda = df_cosmeticos.copy()
df_cosmeticos_eda = df_cosmeticos_eda['Ingredientes'].str.split(',')

# %%
df_cosmeticos_eda

# %%
df_ingredientes = df_cosmeticos_eda.explode('Ingredientes')

# %%
df_ingredientes

# %% [markdown]
# Análise Univariada
# 

# %%
# Estatisticas das VAriávris
df_cosmeticos.describe()

# %%
# Distribuição da variável Tipo
percentual_tipo = df_cosmeticos.value_counts('Tipo') / len(df_cosmeticos) * 100
percentual_tipo = percentual_tipo.reset_index()
percentual_tipo.columns = ['Tipo', 'Percentual']
fig = px.bar(percentual_tipo, x='Tipo', y='Percentual', color='Tipo', text='Percentual')

# Atulizar o Plot para melhor visualizar os labels
fig.update_traces(texttemplate='%{text:.4s}%', textposition='outside')
fig.show()



# %%
# Distribuição da variável Marca
percentual_marca = df_cosmeticos.value_counts('Marca') / len(df_cosmeticos) * 100
percentual_marca = percentual_marca.reset_index()
percentual_marca.columns = ['Marca', 'Percentual']
fig = px.bar(percentual_marca.head(10), x='Percentual', y='Marca', color='Marca', text='Percentual', orientation='h')

# Atulizar o Plot para melhor visualizar os labels
fig.update_traces(texttemplate='%{text:.4s}%', textposition='outside')
fig.show()

# %%
# Quantidade de Marcas
len(df_cosmeticos['Marca'].unique())

# %%
# Distribuição da variável Ingrediente
percentual_ingrediente = df_ingredientes.value_counts('Ingredientes') / len(df_ingredientes) * 100
percentual_ingrediente = percentual_ingrediente.reset_index()
percentual_ingrediente.columns = ['Ingrediente', 'Percentual']
fig = px.bar(percentual_ingrediente.head(10), x='Percentual', y='Ingrediente', color='Ingrediente', text='Percentual')

# Atulizar o Plot para melhor visualizar os labels
fig.update_traces(texttemplate='%{text:.4s}%', textposition='outside')
fig.show()

# %%
# Quantidade de Marcas
len(percentual_ingrediente['Ingrediente'].unique())

# %%
# Distribuição da variável Preço
px.histogram(df_cosmeticos['Preco'], title='Histograma da variável Preço')

# %%
# Distribuição da variável Rating
px.histogram(df_cosmeticos['Rating'], title='Histograma da variável Rating')

# %% [markdown]
# ### Análise Bivariada

# %%
# Plot de Distribuição (BoxPlot) por Tipo de Preço
px.box(df_cosmeticos, x='Preco', y='Tipo', color='Tipo', orientation='h', hover_data=['Marca', 'Nome'])

# %%
# Plot de Distribuição (BoxPlot) por Tipo de Rating
px.box(df_cosmeticos, x='Rating', y='Tipo', color='Tipo', orientation='h', hover_data=['Marca', 'Nome'])

# %%
# Correlação Preço e rating
px.scatter(df_cosmeticos, x='Preco', y='Rating', color='Tipo', hover_data=['Marca'])

# %%
# Gerar Matriz de Correlação
matriz_correlacao_cosmeticos = df_cosmeticos.corr(numeric_only=True)

# %%
# Plotar a matriz de correlação
sns.heatmap(matriz_correlacao_cosmeticos, vmin=-1, vmax=1, annot=True)

# %% [markdown]
# ### Treinar o algoritmo t-SNE

# %%
# Copiar DataFrame original
X = df_cosmeticos.copy()
X.drop(columns=['Nome', 'Ingredientes'], axis=1, inplace=True)


# %%
# Separando Variáveis numéricas e categoricas
numeric_features = ['Rating', 'Preco']
categorical_features = ['Marca', 'Tipo']

# %%
# Definir as tranformações
numeric_tranformer = StandardScaler()
categorical_tranformer = OneHotEncoder()

# %%
# Criar Pre Processor
preprocessor = ColumnTransformer(
    transformers= [
        ('num', numeric_tranformer, numeric_features),
        ('cat', categorical_tranformer, categorical_features)
    ],
    remainder='passthrough'
)

# %%
# Tranformar os dados
X_transformed = preprocessor.fit_transform(X)

# %%
# Visualizar Dados
X_transformed

# %%
# Armazenar resultados do t-SNE em DataFrame
results_df = pd.DataFrame()

# %%
# Loop de treinamento do algoritmo, mudando o parâmetro Perplexity
for perplexity in range(5, 51,1):

    # Criar e treinar modelo
    tsne = TSNE(n_components=3, perplexity=perplexity, init="random", max_iter=250, random_state=51)
    tsne_results = tsne.fit_transform(X_transformed)

    # Armazenar resultados
    temp_df = pd.DataFrame(tsne_results, columns=['Componente 1', 'Componente 2', 'Componente 3'])
    temp_df['Perplexity'] = perplexity
    results_df = pd.concat([results_df, temp_df], axis=0)


# %%
results_df.head(10)

# %%
# Reset no Indice para realizar o plot
results_df.reset_index(drop=True, inplace=True)

# %% [markdown]
# ### Visualizar Resultados

# %%
# Criar um ScatterPlot Animado com variação no Perplexity
fig = px.scatter_3d(results_df, x='Componente 1', y='Componente 2', z='Componente 3', animation_frame='Perplexity', title='Visualização do t-SNE com variação do Perplexity')
fig.show()

# %%



