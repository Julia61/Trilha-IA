# %%
import pandas as pd
import plotly.express as px
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ML
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Otimização HP
import optuna

# %% [markdown]
# ### Carregar os Dados

# %%
# Carga de Dados
df_clientes = pd.read_csv('./datasets/dataset_clientes_pj.csv')

# %%
# Visualizar a estrutura
df_clientes.info()

# %%
# Visualizar os primeiros registros
df_clientes.head(10)

# %% [markdown]
# ### EDA

# %%
# Distribuição da variável inovação
percentual_inovacao = df_clientes.value_counts('inovacao') / len(df_clientes) * 100
px.bar(percentual_inovacao, color=percentual_inovacao.index)

# %%
# Teste ANOVA (Análise de Variância)
# Verificar se há variações significativas na média de faturamento mensal para diferentes níveis de inovaçãp
# Suposições / Pressupostos:
# - Observações independentes
# - Variável dependente é contínua
# - Segue uma distribuição normal
# - Homogeneidade das variâncias
# - Amostrar sejam de tamanhos iguais

# %%
# Chegar se as variâncias (faturamento) entre os grupos (inovação) são homogêneas
# Aplicar Teste de Bartlett
# H0 - Variância são iguais
# H1 - Variânciasnão são iguais

from scipy.stats import bartlett

# Separando os dados de faturamento em frupos com base na coluna 'inovação'
dados_agrupados = [df_clientes['faturamento_mensal'][df_clientes['inovacao'] == grupo] for grupo in df_clientes['inovacao'].unique()]

# Executar o teste de Bartlett
bartlett_test_statistic, bartlett_p_value = bartlett(*dados_agrupados)

# Exibindo os resultados
print(f'Estatistica do Teste Bartlett: {bartlett_test_statistic} ')
print(f'P-Value do Teste Bartlett: {bartlett_p_value} ')


# %%
# Executar o Teste de Shapiro-Wilk
# Verificar se os dados seguem uma distribuição normal
# H0 - Seguem uma distribuição normal
# H1 - Não seguem uma distribuição normal

from scipy.stats import shapiro

# Executar o teste
# Executar o teste de Bartlett
shapiro_test_statistic, shapiro_p_value = shapiro(df_clientes['faturamento_mensal'])

# Exibindo os resultados
print(f'Estatistica do Teste SW: {shapiro_test_statistic} ')
print(f'P-Value do Teste SW: {shapiro_p_value} ')


# %%
# Aplicar a ANOVA de Welch, pois as amostras são de tamanhos diferentes
# H0 - Não há diferenças significativas entre as médias dos grupos
# H1 - Há pelo menos uma diferença significativa entre as médias dos grupos
from pingouin import welch_anova

aov = welch_anova(dv='faturamento_mensal', between='inovacao', data=df_clientes)

# Exibindo os resultados

print(f'Estatistica do Teste de ANOVA de Welch: {aov.loc[0, 'F']} ')
print(f'P-Value do Teste de ANOVA de Welch: {aov.loc[0, 'p-unc']} ')

# %% [markdown]
# ### Treinar o algoritmo K-Means

# %%
# Selecionar as colunas para cluesterização 
X = df_clientes.copy()

# Separando variáveis numericas, categoricas e ordinais
numeric_features = ['faturamento_mensal', 'numero_de_funcionarios', 'idade']
categorical_features = ['localizacao', 'atividade_economica']
ordinal_features = ['inovacao']

# Aplicar Tranformação por tipo
numeric_tranformer = StandardScaler()
categorical_tranformer = OneHotEncoder()
ordinal_tranformer = OrdinalEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_tranformer, numeric_features),
        ('cat', categorical_tranformer, categorical_features),
        ('ord', ordinal_tranformer, ordinal_features)
    ]
)

# Tranformar os dados
X_tranformed = preprocessor.fit_transform(X)

# %%
X_tranformed

# %%
# Optuna para Otimização de Hiperparâmetros
def kmeans_objective(trial):
    # Definindo os hiperparâmetros a serem ajustados
    n_clusters = trial.suggest_int('n_clusters', 3, 10)
    distance_metric = trial.suggest_categorical('distance_metric', ['euclidean', 'minkowski'])

    # Criar o Modelo
    modelo_kmeans = KMeans(n_clusters=n_clusters, random_state=51)

    # Treinar o modelo
    modelo_kmeans.fit(X_tranformed)

    # Calcular o Silhouette Score
    distances = pairwise_distances(X_tranformed, metric=distance_metric)
    silhouette_avg = silhouette_score(distances,  modelo_kmeans.labels_)

    return silhouette_avg

# %%
# Criar um estudo do Optuna
search_space = {'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10], 'distance_metric': ['euclidean', 'minkowski']}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_kmeans = optuna.create_study(direction='maximize', sampler=sampler)

# Rodar o estudo
estudo_kmeans.optimize(kmeans_objective, n_trials=100)

# %%
# Melhor configuração encontrada pelo Optuna
best_params = estudo_kmeans.best_params

# Instanciando o modelo K-Means com melhores parâmetros
best_kmeans = KMeans(n_clusters=best_params['n_clusters'], random_state=51)
best_kmeans.fit(X_tranformed)

# Calcular o Silhouette Score
distances = pairwise_distances(X_tranformed, metric=best_params['distance_metric'])
best_silhouette = silhouette_score(distances, best_kmeans.labels_)

print(f'K (Numero de Clusters): {best_params['n_clusters']}')
print(f'Métrica de Distância Selecionada: {best_params['distance_metric']}')
print(f'Silhouette Score: {best_silhouette}')


# %%
# Criar coluna com cluster escolhido
df_clientes['cluster'] = best_kmeans.labels_

# %%
# Visualizar os primeiros registros
df_clientes.head(10)

# %% [markdown]
# ### Visualizar Resultados
# 

# %%
# Cruzar idade e faturamento, apresentando os clusters
px.scatter(df_clientes, x='idade', y='faturamento_mensal', color='cluster')

# %%
# Cruzar inovação e faturamento, apresentando os clusters
px.scatter(df_clientes, x='inovacao', y='faturamento_mensal', color='cluster')

# %%
# Cruzar numero funcionarios e faturamento, apresentando os clusters
px.scatter(df_clientes, x='numero_de_funcionarios', y='faturamento_mensal', color='cluster')

# %% [markdown]
# ### Salvar o Modelo e o Pipilene de Tranformação

# %%
import joblib

# Salvar o modelo
joblib.dump(best_kmeans, 'modelo_clusterizacao_clientes.pkl')

# Salvar o Pipeline
joblib.dump(preprocessor, 'pipeline_clusterizacao_clientes.pkl')

# %% [markdown]
# ### Aplicação Bath no Gradio

# %%
import gradio as gr
modelo = joblib.load('./modelo_clusterizacao_clientes.pkl')
preprocessor = joblib.load('./pipeline_clusterizacao_clientes.pkl')

def clustering(arquivo):
    # Carregar o CSV em um DataFrame
    df_empresas = pd.read_csv(arquivo.name)

    # Tranformar os dados do DF para o formato que o KMeans precisa
    X_tranformed = preprocessor.fit_transform(df_empresas)

    # Treinar modelo
    modelo.fit(X_tranformed)

    # Criar a coluna cluster no DF
    df_empresas['cluster'] = modelo.labels_
    df_empresas.to_csv('./cluster.csv', index=False)

    return './cluster.csv'

# %%
# Criar a interface 
app = gr.Interface(
    clustering,
    gr.File(file_types=[".csv"]),
    "file"
)

# Rodar a aplicação
app.launch()

# %%



