# %%
# pipenv install scikit-learn scipy pandas matplotlib seaborn ipykernel gradio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Carregar o arquivo
df_colesterol = pd.read_csv('./datasets/dataset_colesterol.csv')

# %%
# Checar arquivo
df_colesterol.info()

# %%
# Remover coluna Id
df_colesterol.drop(columns=['Id'], axis=1, inplace=True)

# %%
# Renomear as colunas
df_colesterol.columns = [
    'grupo_sanguineo',
    'fumante',
    'nivel_atividade_fisica',
    'idade',
    'peso',
    'altura',
    'nivel_colesterol'
    
]

# %%
# Copiar DF para DF EDA
df_colesterol_eda = df_colesterol.copy()

# %%
# Copiar DF para DF bucketing
df_colesterol_bucketing = df_colesterol.copy()

# %% [markdown]
# ### EDA
# 

# %%
# Visualizar os dados
df_colesterol_eda.head(10)

# %%
df_colesterol_eda.nivel_atividade_fisica.unique()

# %%
# Detectar valores ausentes
df_colesterol_eda.isna().sum()

# %%
# Medidas estatísticas do DF
df_colesterol_eda.describe()

# %%
# Coletar medidas das variáveis categoricas
moda_grupo_sang = df_colesterol_eda.grupo_sanguineo.mode()
moda_fumante = df_colesterol_eda.fumante.mode()
moda_nivel_atividade = df_colesterol_eda.nivel_atividade_fisica.mode()

# %%
# Coletar medidas das variáveis numericas
mediana_idade = df_colesterol_eda.idade.median()
mediana_peso = df_colesterol_eda.peso.median()
mediana_altura = df_colesterol_eda.altura.median()

# %%
# Imputar valores ausentes
df_colesterol_eda.fillna(value={'grupo_sanguineo': moda_grupo_sang[0],
                                'fumante': moda_fumante[0],
                                'nivel_atividade_fisica': moda_nivel_atividade[0],
                                'idade': mediana_idade,
                                'peso': mediana_peso,
                                'altura': mediana_altura
                                }, inplace=True)

# %%
# Converter idade e altura para inteiro
df_colesterol_eda.idade = df_colesterol_eda.idade.astype(int)

# %%
df_colesterol_eda.altura = df_colesterol_eda.altura.astype(int)

# %%
# Estrutura di Df
df_colesterol_eda.info()

# %%
# Visualizar os dados
df_colesterol_eda.head(10)

# %%
# Verificar / detectar outliers
sns.boxplot(data=df_colesterol_eda, x='idade')

# %%
sns.boxplot(data=df_colesterol_eda, x='peso')

# %%
# Filtrar o público a ser removido
df_colesterol_eda[df_colesterol_eda['peso'] < 40].peso.count()

# %%
# Remover público do DataFrame
df_colesterol_eda.drop(df_colesterol_eda[df_colesterol_eda['peso'] < 40].index, axis=0, inplace=True)

# %%
sns.boxplot(data=df_colesterol_eda, x='altura')

# %%
sns.boxplot(data=df_colesterol_eda, x='nivel_colesterol')

# %%
# Cruzamento de Variáveis Categoricas com Nivel Colesterol
sns.boxplot(data=df_colesterol_eda, x='grupo_sanguineo', y='nivel_colesterol')

# %%
sns.boxplot(data=df_colesterol_eda, x='fumante', y='nivel_colesterol')

# %%
sns.boxplot(data=df_colesterol_eda, x='nivel_atividade_fisica', y='nivel_colesterol')

# %%
# Cruzamento Variáveis numéricas com nível colesterol
sns.scatterplot(data=df_colesterol_eda, x='idade', y='nivel_colesterol')

# %%
sns.scatterplot(data=df_colesterol_eda, x='peso', y='nivel_colesterol')

# %%
sns.scatterplot(data=df_colesterol_eda, x='altura', y='nivel_colesterol')

# %%
sns.pairplot(df_colesterol_eda)

# %%
# Converter variáveis categoricas nominais em numericas, usando One-Hot Encoder do Pandas
df_colesterol_eda = pd.get_dummies(df_colesterol_eda, columns=['grupo_sanguineo', 'fumante'], dtype='int')

# %%
df_colesterol_eda.head(10)

# %%
# Converter variavel categorica ordinal em numerica, usando factorize do Pandas
df_colesterol_eda['nivel_atividade_fisica'] = pd.factorize(df_colesterol_eda.nivel_atividade_fisica)[0] + 1


# %%
# Mapa de Calor com Correlação entre as variaveis
plt.figure(figsize=(15,6))
sns.heatmap(df_colesterol_eda.corr(), vmin=-1, vmax=1, annot=True)

# %%
# Formato de Ranking, somente correlação com a varivel target (nivel_colesterol)
sns.heatmap(df_colesterol_eda.corr()[['nivel_colesterol']].sort_values(by='nivel_colesterol', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')

# %%
# Bucketing Idade
bins_idade = [20, 30, 40, 50, 60, 70, 80]
labels_idade = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df_colesterol_bucketing['escala_idade'] = pd.cut(x=df_colesterol_bucketing['idade'], bins= bins_idade, labels= labels_idade, include_lowest=True)


# %%
df_colesterol_bucketing.head(10)

# %%
sns.boxplot(df_colesterol_bucketing, x='escala_idade', y='nivel_colesterol')

# %%
# Bucketing Peso
bins_peso = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
labels_peso = ['40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100-109', '110-119','120-129','130-139', '140-149','150-159']
df_colesterol_bucketing['escala_peso'] = pd.cut(x=df_colesterol_bucketing['peso'], bins= bins_peso, labels= labels_peso, include_lowest=True)

# %%
plt.figure(figsize=(15,8))
sns.boxplot(df_colesterol_bucketing, x='escala_peso', y='nivel_colesterol')

# %% [markdown]
# ### Treinar Modelo

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


# %%
# Criar DataSet de Treino e Teste

df_colesterol.drop(df_colesterol[df_colesterol['peso'] < 40].index, axis=0, inplace=True)


# %%
df_colesterol.info()

# %%
X = df_colesterol.drop(columns='nivel_colesterol', axis=1)
y = df_colesterol['nivel_colesterol']

# %%
# Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=51)

# %%
y_test.shape

# %%
# Pipeline
# Imputar moda nas variáveis categoricas - grupo_sanguineo, fumante, nivel_atividade_fisica
# Padronizar variáveis numericas - idade, altura, peso
# OneHotEncode nas variáveis categoricas nominais -  grupo_sanguineo, fumante
# OrdinalEncode nas variáveis categóricas ordinais - nivel_atividade_fisica
# Imputar mediana nas variáveis numericas - idade, altura, peso

# Nomes das Colunas
colunas_categoricas = ['grupo_sanguineo', 'fumante']
colunas_numericas = ['idade', 'altura', 'peso']
colunas_ordinais = ['nivel_atividade_fisica']

# %%
# Tranformer Categoricas
tranformer_categorias = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %%
# Tranformer Ordinais

tranformer_ordinais = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[['Baixo', 'Moderado', 'Alto']], handle_unknown='error'))
])

# %%
# Trasformer Numericas
tranformer_numericas = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# %%
# Criar um ColumTranformer que encapsula todas as tranformações
preprocessor = ColumnTransformer(
    transformers=[
        ('num', tranformer_numericas, colunas_numericas),
        ('cat', tranformer_categorias, colunas_categoricas),
        ('ord', tranformer_ordinais, colunas_ordinais)
    ]
)

# %%
# Criando o Pipeline principal = Pre Processamento + Treinamento
model_reg = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])

# %%
# Treinar Modelo
model_reg.fit(X_train, y_train)

# %% [markdown]
# ### Análise de Métricas

# %%
# Gerar Predição
y_pred = model_reg.predict(X_test)

# %%
# Calcular R2 Score
r2_score(y_test,y_pred)

# %%
# Calcular MAE (Mean Absolute Error)
mean_absolute_error(y_test, y_pred)

# %%
# Calcular RMSE (Root Mean Absolute Error)
root_mean_squared_error(y_test, y_pred)

# %% [markdown]
# ### Análise de Residuos

# %%
# Calcular residuos
residuos = y_test - y_pred

# %%
# Tranformar residuos na escala padrão
# (X - media) / desvio_parao
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
# Verificar Linearidade dos residuos: Valores entre -2 e +2 (Escala Padrão)
# Verificar homocedasticidade: Valores em torno da reta
sns.scatterplot(x=y_pred, y=residuos_std)
plt.axhline(y=0)
plt.axhline(y=-2)
plt.axhline(y=2)

# %%
# Chegar se resíduos seguem uma distribuição normal
# QQ Plot
import pingouin as pg
plt.figure(figsize=(14,8))
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Resíduos na escala padrão')
plt.show()

# %%
# Teste de Normalidade de Shapiro-Wilk
from scipy.stats import shapiro, kstest, anderson
from statsmodels.stats.diagnostic import lilliefors, het_goldfeldquandt
stat_shapiro, p_value_shapiro = shapiro(residuos)
print("Estatistica do Teste: {} e P-Value: {}".format(stat_shapiro,p_value_shapiro))

# %%
# Teste de Kolmogorov_smirnov

stat_ks, p_value_ks = kstest(residuos, 'norm')
print("Estatistica do Teste: {} e P-Value: {}".format(stat_ks,p_value_ks))

# %%
# Teste de Lilliedors

stat_ll, p_value_ll = lilliefors(residuos, dist='norm', pvalmethod='table')
print("Estatistica do Teste: {} e P-Value: {}".format(stat_ll,p_value_ll))

# %%
# Teste de Anderson-Darling

stat_and, critical_and, significante_and = anderson(residuos,  dist='norm')


# %%
critical_and

# %%
significante_and

# %%
print("Estatistica do Teste: {} e Valor Crítico: {}".format(stat_and,critical_and[2]))

# %%
# Teste de Homocedasticidade de Goldfeld-Quandt
pipe = Pipeline(steps=[('preeprocessor', preprocessor)])
X_test_tranformed = pipe.fit_transform(X_test)

# %%
X_test_tranformed

# %%
test_goldfeld = het_goldfeldquandt(residuos, X_test_tranformed)
stat_goldfeld = test_goldfeld[0]
p_value_goldfeld = test_goldfeld[1]
print("Estatistica do Teste: {} e P-Value: {}".format(stat_goldfeld,p_value_goldfeld))

# %% [markdown]
# ### Realizar Predições individuais
# 

# %%
predicao_individual = {
    'grupo_sanguineo': 'O',
    'fumante': 'Não',
    'nivel_atividade_fisica': 'Alto',
    'idade': 40,
    'peso': 70,
    'altura': 180
}
sample_df = pd.DataFrame(predicao_individual, index=[1])

# %%
sample_df

# %%
# Predição
model_reg.predict(sample_df)

# %%
import joblib


# %%
# Salvar Modelo
joblib.dump(model_reg, './modelo_colesterol.pkl')


