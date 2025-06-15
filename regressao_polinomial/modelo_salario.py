# %%
# pipenv install scikit-learn scipy pandas matplotlib seaborn ipykernel pingouin fastapi pydantic streamlit uvicorn requests

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%
df_salario = pd.read_csv('./datasets/dataset_salario.csv')

# %%
# Inpecionar a estrutura do dataframe
df_salario.info()

# %%
# Copiar DF oara DF EDA
df_salario_eda = df_salario.copy()

# %%
df_salario_bucketing = df_salario.copy()

# %% [markdown]
# ### EDA

# %%
# Visualizar os dados
df_salario_eda.head(10)

# %%
# Detectar valores ausentes
df_salario_eda.isna().sum()

# %%
# Médidas Estatísticas do DF
df_salario_eda.describe()

# %%
# Verificar / detectar outliers
sns.boxplot(data=df_salario_eda, x='tempo_na_empresa')

# %%
sns.boxplot(data=df_salario_eda, x='nivel_na_empresa')

# %%
sns.boxplot(data=df_salario_eda, x='salario_em_reais')

# %%
# Cruzamento variaveis númericas com salario em reais (variavel dependente)
sns.scatterplot(data=df_salario_eda, x='tempo_na_empresa', y='salario_em_reais')

# %%
sns.scatterplot(data=df_salario_eda, x='nivel_na_empresa', y='salario_em_reais')

# %%
sns.scatterplot(data=df_salario_eda, x='nivel_na_empresa', y='tempo_na_empresa')

# %%
# Histrogramas das variáveis
sns.pairplot(df_salario_eda)

# %%
# Mapa de Calor
plt.figure(figsize=(15,6))
sns.heatmap(df_salario_eda.corr('spearman'), vmin=-1, vmax=1, annot=True)

# %%
# Formato de Ranking
sns.heatmap(df_salario_eda.corr('spearman')[['salario_em_reais']].sort_values(by='salario_em_reais', ascending=False), vmin=-1, vmax=1,annot=True, cmap='BrBG')

# %%
# Bucketing Tempo de Casa
bins_tempo_casa = [0,10,20,30,40,50,60,70,80,90,100,110,120, 130]
labels_tempo_csa = ['0-9', '10-19','20-29', '30-39', '40-49','50-59','60-69', '70-79','80-89','90-99','100-109','110-119', '120-129']
df_salario_bucketing['escala_tempo'] = pd.cut(x=df_salario_bucketing['tempo_na_empresa'], bins=bins_tempo_casa, labels=labels_tempo_csa,include_lowest=True)

# %%
df_salario_bucketing.head(20)

# %%
plt.figure(figsize=(14,8))
sns.boxplot(df_salario_bucketing, x='escala_tempo', y='salario_em_reais')

# %% [markdown]
# ### Treinar Modelo Linear

# %%
# Importar Bibliotecas
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

import numpy as np

# %%
# Criar o Dataset de Treino e Teste
X = df_salario.drop(columns='salario_em_reais', axis=1)
y = df_salario['salario_em_reais']

# %%
# Split usando Kfold com 5 pastas/splits
kf = KFold(n_splits=5, shuffle=True, random_state=51)

# %%
kf.split(X)

# %%
# Bloco para Treinamento do Modelo

# Pipeline
# Padronizar variáveis numericas - tempo_na_empresa, nivel_na_empresa

# Nomes das colunas
colunas_numericas = ['tempo_na_empresa', 'nivel_na_empresa']

# Tranformer para Colunas Númericas
tranformer_numericas = Pipeline(steps=[
    ('scaler', StandardScaler())
])


# Criar um ColumnTranformer
preprocessor =  ColumnTransformer(
    transformers=[
        ('num', tranformer_numericas, colunas_numericas)
    ]
)

# Criando o Pipeline principal = Pré_Processamento + Treinamento
model_regr = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', LinearRegression())])

# Armazenar RMSE Treino e Teste
rmse_scores_fold_train = []
rmse_scores_fold_test = []

# Armazenar R2 Score de Teste
r2score_fold_test = []

# Armazenar Resíduos
residuos = []

# Armazenar Predições
y_pred_total = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Treine o modelo no conjunto de treinamento
    model_regr.fit(X_train,y_train)

    # Faça predições nos conjuntos de treinamento e teste
    y_train_pred = model_regr.predict(X_train)
    y_test_pred = model_regr.predict(X_test)

    # Calcule o RMSE para os conjuntos de treinamento e teste
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    # Calcular R2Score e Residuos para conjunto de teste
    r2score_test = r2_score(y_test, y_test_pred)
    residuos_test = np.array(y_test - y_test_pred)
    
    # Armazeno as metricas da iteração na lista
    rmse_scores_fold_train.append(rmse_train)
    rmse_scores_fold_test.append(rmse_test)
    r2score_fold_test.append(r2score_test)
    residuos.append(residuos_test)
    y_pred_total.append(y_test_pred)

rmse_train_final = np.mean(rmse_scores_fold_train)
rmse_test_final = np.mean(rmse_scores_fold_test)
r2score_test_final = np.mean(r2score_fold_test)
percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
residuos = np.array(residuos).reshape(-1)
y_pred_total = np.array(y_pred_total).reshape(-1)


# %% [markdown]
# ### Análise de Métricas - Modelo Linear

# %%
# Métricas 
print(f'RMSE Treino: {rmse_train_final}')
print(f'RMSE Teste: {rmse_test_final}')
print(f'% Dif. RMSE Treino e Teste: {percentual_rmse_final}')
print(f'R2Score Teste: {r2score_test_final}')


# %% [markdown]
# ### Análise de Resíduos - Modelo Linear

# %%
# Tranformar residuos na escala padrão
# (X - media) / desvio_padrao
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
# Verificar linearidade dos resíduos: Valores entre -2 e +2 (Escala Padrão)
# Verificar homocedasticidade: Valores em torno da reta sem nenhuma tendência ou formato
sns.scatterplot(x=y_pred_total, y=residuos_std)
plt.axhline(y=0)
plt.axhline(y=-2)
plt.axhline(y=2)

# %%
# Chegar se rediduos seguem uma distribuição normal
# QQ Plot
import pingouin as pg
plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.show()

# %%
# Teste de Normalidade - Shapiro-Wilk
from scipy.stats import shapiro, kstest
from statsmodels.stats.diagnostic import lilliefors
stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f'Estat. Teste {stat_shapiro} e P-Value {p_value_shapiro}')

# %%
# Teste de Normalidade - Kolmogorov-Smirnov
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f'Estat. Teste {stat_ks} e P-Value {p_value_ks}')

# %%
# Teste de Normalidade - Lilliefors
stat_ll, p_value_ll = lilliefors(residuos, dist='norm', pvalmethod='table')
print(f'Estat. Teste {stat_ll} e P-Value {p_value_ll}')

# %% [markdown]
# ### Treinar Modelo Polinomial

# %%
# Exemplo de criação de Features Polinomiais
feat_poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = feat_poly.fit_transform(X)
feat_poly.feature_names_in_


# %%
feat_poly.get_feature_names_out(feat_poly.feature_names_in_)

# %%
# Bloco para Treinamento do Modelo

# graus_polynomial = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
graus_polynomial = [4]

rmse_train_values = []
rmse_test_values = []
percentual_rmse_values = []
r2score_test_values = []

# Split usando Kfold com 5 pastas/splits
kf = KFold(n_splits=5, shuffle=True, random_state=51)

for grau in graus_polynomial: 
    # Pipeline
    # Padronizar variáveis numericas - tempo_na_empresa, nivel_na_empresa


    # Nomes das colunas
    colunas_numericas = ['tempo_na_empresa', 'nivel_na_empresa']

    # Tranformer para Colunas Númericas
    tranformer_numericas = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])


    # Criar um ColumnTranformer
    preprocessor =  ColumnTransformer(
        transformers=[
            ('num', tranformer_numericas, colunas_numericas)
        ]
    )

    # Criar Features Polinomiais
    poly_feat = PolynomialFeatures(degree=grau, include_bias=False)

    # Criando o Pipeline principal = Pré_Processamento + Treinamento
    model_poly = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('poly_features', poly_feat),
                                ('regressor', LinearRegression())])

    # Armazenar RMSE Treino e Teste
    rmse_scores_fold_train = []
    rmse_scores_fold_test = []

    # Armazenar R2 Score de Teste
    r2score_fold_test = []

    # Armazenar Resíduos
    residuos = []

    # Armazenar Predições
    y_pred_total = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Treine o modelo no conjunto de treinamento
        model_poly.fit(X_train,y_train)

        # Faça predições nos conjuntos de treinamento e teste
        y_train_pred = model_poly.predict(X_train)
        y_test_pred = model_poly.predict(X_test)

        # Calcule o RMSE para os conjuntos de treinamento e teste
        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)

        # Calcular R2Score e Residuos para conjunto de teste
        r2score_test = r2_score(y_test, y_test_pred)
        residuos_test = np.array(y_test - y_test_pred)
        
        # Armazeno as metricas da iteração na lista
        rmse_scores_fold_train.append(rmse_train)
        rmse_scores_fold_test.append(rmse_test)
        r2score_fold_test.append(r2score_test)
        residuos.append(residuos_test)
        y_pred_total.append(y_test_pred)

    rmse_train_final = np.mean(rmse_scores_fold_train)
    rmse_test_final = np.mean(rmse_scores_fold_test)
    r2score_test_final = np.mean(r2score_fold_test)
    percentual_rmse_final = ((rmse_test_final - rmse_train_final) / rmse_train_final) * 100
    residuos = np.array(residuos).reshape(-1)
    y_pred_total = np.array(y_pred_total).reshape(-1)

    rmse_train_values.append(rmse_train_final)
    rmse_test_values.append(rmse_test_final)
    r2score_test_values.append(r2score_test_final)
    percentual_rmse_values.append(percentual_rmse_final)



# %%
# Plotar Gráfico para comparar RMSE por Grau de Polinômio
plt.figure(figsize=(12,8))
plt.plot(graus_polynomial, rmse_train_values, label='RMSE (Treino)')
plt.plot(graus_polynomial, rmse_test_values, label='RMSE (Teste)')
plt.xlabel('Grau do Polinômio')
plt.ylabel('RMSE')
plt.title('RMSE por Grau do Polinômio')
plt.legend()
plt.grid(True)
plt.show()


# %%
# Plotar Gráfico para comparar %Dif RMSE (Treino e Teste)
plt.figure(figsize=(12,8))
plt.plot(graus_polynomial, percentual_rmse_values, label='%Dif RMSE Treino e Teste')
plt.xlabel('Grau do Polinômio')
plt.ylabel('%Dif RMSE')
plt.title('%Dif RMSE por Grau do Polinômio')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Análise de Métricas - Modelo Polinomial

# %%
# Métricas 
print(f'RMSE Treino: {rmse_train_final}')
print(f'RMSE Teste: {rmse_test_final}')
print(f'% Dif. RMSE Treino e Teste: {percentual_rmse_final}')
print(f'R2Score Teste: {r2score_test_final}')

# %% [markdown]
# ### Análise de Resíduos - Modelo Polinomial

# %%
# Tranformar residuos na escala padrão
# (X - media) / desvio_padrao
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
# Verificar linearidade dos resíduos: Valores entre -2 e +2 (Escala Padrão)
# Verificar homocedasticidade: Valores em torno da reta sem nenhuma tendência ou formato
sns.scatterplot(x=y_pred_total, y=residuos_std)
plt.axhline(y=0)
plt.axhline(y=-2)
plt.axhline(y=2)

# %%
# Chegar se rediduos seguem uma distribuição normal
# QQ Plot
import pingouin as pg
plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.show()

# %%
# Teste de Normalidade - Shapiro-Wilk
from scipy.stats import shapiro, kstest
from statsmodels.stats.diagnostic import lilliefors
stat_shapiro, p_value_shapiro = shapiro(residuos)
print(f'Estat. Teste {stat_shapiro} e P-Value {p_value_shapiro}')

# %%
# Teste de Normalidade - Kolmogorov-Smirnov
stat_ks, p_value_ks = kstest(residuos, 'norm')
print(f'Estat. Teste {stat_ks} e P-Value {p_value_ks}')

# %%
# Teste de Normalidade - Lilliefors
stat_ll, p_value_ll = lilliefors(residuos, dist='norm', pvalmethod='table')
print(f'Estat. Teste {stat_ll} e P-Value {p_value_ll}')

# %% [markdown]
# ## Realizar Predições Individuais

# %%
input_features = {
    'tempo_na_empresa': 80,
    'nivel_na_empresa': 10
}

pred_df = pd.DataFrame(input_features, index=[1])

# %%
# Predição
model_poly.predict(pred_df)

# %%
import joblib

# %%
# Salvar Modelo
joblib.dump(model_poly, './modelo_salario.pkl')

# %%



