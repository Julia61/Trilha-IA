# %%
# EDA e Visualização de Dados
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt 

# Ml 
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

# Otimização
import optuna

# %% [markdown]
# ### Carregar Dados
# 

# %%
df_laptops = pd.read_csv('./datasets/laptops_new.csv')

# %%
# Analisar a Estrutura
df_laptops.info()

# %%
# Visualizar os primeiros registros
df_laptops.head(10)

# %%
# Visualizar os ultimos registros
df_laptops.tail(10)

# %% [markdown]
# ### EDA

# %%
# Estatística das variáveis
df_laptops.describe()

# %%
# Checar variável year_of_warranty
df_laptops['year_of_warranty'].unique()

# %%
# Ajustar variável year_of_warranty
df_laptops.loc[df_laptops['year_of_warranty'] == 'No information', 'year_of_warranty'] = 1
df_laptops['year_of_warranty'] = df_laptops['year_of_warranty'].astype(int)

# %%
# Transformar variável booleana em int (is_touch_screen)
df_laptops['is_touch_screen'] = df_laptops['is_touch_screen'].astype(int)

# %%
# Distribuição da variável brand
percentual_brand = df_laptops.value_counts('brand') / len(df_laptops) * 100
px.bar(percentual_brand, color=percentual_brand.index)

# %%
# Distribuição da variável processor_brand
# Distribuição da variável brand
percentual_processor_brand = df_laptops.value_counts('processor_brand') / len(df_laptops) * 100
px.bar(percentual_processor_brand, color=percentual_processor_brand.index)

# %%
# Distribuição da variável price
sns.histplot(df_laptops['price'], kde=True, color='lightblue')

# %%
# Distribuição da variável rating
sns.histplot(df_laptops['rating'], kde=True, color='red')

# %%
# Plot de Distribuição (BoxPlot) por Brand e Price
px.box(df_laptops, x='price', y='brand', color='brand', orientation='h')

# %%
# Plot de Distribuição (BoxPlot) por Brand e Rating
px.box(df_laptops, x='rating', y='brand', color='brand', orientation='h')

# %%
# Scatterplot de Price e Rating
px.scatter(df_laptops, x='price', y='rating', color='brand')

# %% [markdown]
# ### Treinar Modelo Clustering hierarquico

# %%
# Selecionar as colunas para clusterização
X = df_laptops.copy()

# Remvover colunas desnecessárias
X.drop(columns=['index', 'model'], axis= 1, inplace=True)

# %%
# Separando variáveis numericas e categoriacas
numeric_features = ['price', 'rating', 'num_cores', 'num_threads', 'ram_memory', 'primary_storage_capacity', 'display_size', 'resolution_width',        'resolution_height']
categorical_features = ['brand', 'processor_brand', 'gpu_brand', 'gpu_type', 'os']

# %%
# Definir tranformações
numeric_tranformer = StandardScaler()
categorical_tranformer = OneHotEncoder()

# %%
# Criar Pre Processor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_tranformer, numeric_features),
        ('cat', categorical_tranformer, categorical_features)
    ]
)

# %%
# Tranformar os dados
X_transformed = preprocessor.fit_transform(X)

# %%
# Visualizar X_transformed
X_transformed

# %%
def hierarchical_aglomerative_objective(trial):
    # Definindo os hiperparâmetros a serem ajustados
    n_clusters = trial.suggest_int('n_clusters', 10, 150)
    # Linkage = Critério de distância entre dois conjuntos para formar os clusters
    # Ward = Variância
    # Average = Media
    # Complete = Máxima
    # Single = Miníma
    linkage = trial.suggest_categorical('linkage', ['ward', 'average', 'complete', 'single'])

    # Instanciar o modelo 
    hierarchical_model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)

    # Treinar o modelo e já executar a clusterização
    y = hierarchical_model.fit_predict(X_transformed.toarray())

    # Calcular o Silhoueette Score
    silhouette_avg = silhouette_score(X_transformed, y)

    return silhouette_avg

# %%
# Criar um estudo no Optuna
search_space_ag = {'n_clusters': range(10, 151), 'linkage': ['ward', 'average', 'complete', 'single']}
sampler_ag = optuna.samplers.GridSampler(search_space=search_space_ag)
estudo_ag = optuna.create_study(direction='maximize', sampler=sampler_ag)

# %%
# Executar estudo do Optuna para Aglomerative 
estudo_ag.optimize(hierarchical_aglomerative_objective, n_trials=600)

# %%
# Mostrar melhor configuração do Optuna (Aglomerative)
best_params_ag = estudo_ag.best_params

print(f'Clusters = {best_params_ag['n_clusters']}')
print(f'Linkage = {best_params_ag['linkage']}')


# %%
def hierarchical_diviside_objective(trial):
    # Definindo os hiperparâmetros a serem ajustados
    n_clusters = trial.suggest_int('n_clusters', 10, 150)
    
    # Instanciar o modelo 
    hierarchical_model = BisectingKMeans( n_clusters=n_clusters)

    # Treinar o modelo e já executar a clusterização
    y = hierarchical_model.fit_predict(X_transformed.toarray())

    # Calcular o Silhoueette Score
    silhouette_avg = silhouette_score(X_transformed, y)

    return silhouette_avg

# %%
# Criar um estudo no Optuna
search_space_di = {'n_clusters': range(10, 151)}
sampler_di = optuna.samplers.GridSampler(search_space=search_space_di)
estudo_di = optuna.create_study(direction='maximize', sampler=sampler_di)

# %%
# Executar estudo do Optuna para Aglomerative 
estudo_di.optimize(hierarchical_diviside_objective, n_trials=200)

# %%
# Mostrar melhor configuração do Optuna (Divisive)
best_params_di = estudo_di.best_params

print(f'Clusters = {best_params_di['n_clusters']}')


# %%
# Criar modelo com melhor configuração (Aglomerative)

best_model = AgglomerativeClustering(
    n_clusters=best_params_ag['n_clusters'],
    linkage=best_params_ag['linkage']
)

# %%
# Treinar Modelo 
best_model.fit(X_transformed.toarray())

# %%
# Veficar Silhouette Score
best_score = silhouette_score(X_transformed, best_model.labels_)
best_score

# %%
# Criar Coluna com cluster escolhido no Dataframe original
df_laptops['cluster'] = best_model.labels_

# %% [markdown]
# ### Visualizar Resultados

# %%
# Mostrar Chart com Trial do Optuna
fig = optuna.visualization.plot_optimization_history(estudo_ag)
fig.show()

# %%
# Treinar modelo com Scipy
modelo_de = linkage(X_transformed.toarray(), method=best_params_ag['linkage'], optimal_ordering=True)

# %%
# Mostrar Dendograma
plt.figure(figsize=(20, 12))
dendrogram(modelo_de, truncate_mode='lastp', p=50, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrograma Clustering Hierarquico Aglomerativo')
plt.xlabel('Tamanho do Cluster ou Objeto de Dado')
plt.ylabel('Distância')
plt.show()

# %%
# Cortar o dendrograma
clusters_de_scipy = cut_tree(modelo_de, height=32)
len(np.unique(clusters_de_scipy))

# %%
# Cruzamento entre brand e price, apresentando os clusters
px.scatter(df_laptops, x='cluster', y='price', color='brand')

# %%
# Cruzamento entre brand e price, apresentando os clusters
px.scatter(df_laptops, x='brand', y='price', color='cluster')

# %%
# Cruzamento entre brand e rating, apresentando os clusters
px.scatter(df_laptops, x='brand', y='rating', color='cluster')

# %%
# Cruzamento entre brand e rating, apresentando os clusters
px.scatter(df_laptops, x='cluster', y='rating', color='brand')

# %%
# Distribuição da variável cluster
percentual_cluster = df_laptops.value_counts('cluster') / len(df_laptops) * 100
px.bar(percentual_cluster, color=percentual_cluster.index)

# %%
# Distribuição da variável cluster
qtde_cluster = df_laptops.value_counts('cluster') 
px.bar(qtde_cluster, color=qtde_cluster.index)

# %% [markdown]
# ### Salvar Modelo, Prepocessor e CSV Atualizado

# %%
# Salvar Modelo e Preprocessor
import joblib

# Salvar Modelo
joblib.dump(best_model, './modelo_clusterizacao.pkl')

# Salvar o Prepocessor
joblib.dump(preprocessor, './preprocessor_clusterizacao.pkl')


# %%
# Salvar CSV atualizado com dados de cluster
df_laptops.to_csv('./datasets/clusterizacao_laptops.csv', index=False)



