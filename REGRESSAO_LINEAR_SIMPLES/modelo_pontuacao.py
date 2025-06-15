# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, kstest, probplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# %% [markdown]
# ### Carga dos dados
# 

# %%
# Abrir o dataset
df_pontuacao = pd.read_csv('./datasets/pontuacao_teste.csv')

# %%
# Chegar estrutura do Dataset
df_pontuacao.info()

# %%
# Visualizar Topo do DF
df_pontuacao.head(10)

# %%
# Visualizar final do dataset
df_pontuacao.tail(10)

# %% [markdown]
# ### EDA

# %%
# Medidas estatísticas das variáveis
df_pontuacao.describe()

# %%
# Plot de Dispersão
# X = horas_estudo
# y = pontuacao_teste
sns.scatterplot(data=df_pontuacao, x='horas_estudo', y='pontuacao_teste')

# %%
# Verificar se temos outliers 
# Plot BoxPlot
sns.boxplot(df_pontuacao, y='horas_estudo')

# %%
sns.boxplot(df_pontuacao, y='pontuacao_teste')

# %%
# Verificar Correlação - Pearson
sns .heatmap(df_pontuacao.corr('pearson'), annot=True)

# %%
# Verificar Correlação - Spearman
sns .heatmap(df_pontuacao.corr('spearman'), annot=True)

# %%
# Histograma das variáveis
sns.displot(df_pontuacao, x='horas_estudo')

# %%
# Histograma das variáveis
sns.displot(df_pontuacao, x='pontuacao_teste')

# %% [markdown]
# ### Treinar Modelo

# %%
# Dividir dataset entre treino e teste
# Quando temos apenas uma feature, precisamos ajustar o shape
X = df_pontuacao['horas_estudo'].values.reshape(-1,1)
y = df_pontuacao['pontuacao_teste'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)

# %%
# Instanciar o modelo a ser treinado
reg_model = LinearRegression()

# %%
# Treinar o modelo
reg_model.fit(X_train, y_train)

# %%
# Imprimir a equação da reta
# y = aX + b
print("A equação da reta é y = {:4f}x + {:4f}".format(reg_model.coef_[0][0], reg_model.intercept_[0]))

# %% [markdown]
# ### Validar Modelo - Métricas

# %%
# Predição dos valores com base no conjunto de testes
y_pred = reg_model.predict(X_test)

# %%
# Calcular métrica R-squared ou Coeficiente de Determinação
# R2 representa a proporção na variação na variável dependente que é explicada pela variável indeprendente
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error, mean_squared_error
r2_score(y_test, y_pred)

# %%
# Calcular métrica MAE (Mean Absolute Error)
# MAE = Media (y_test - y_pred)
# É uma métrica fácil de interpretar 
# MAE é menos sensível a outliers
mean_absolute_error(y_test, y_pred)

# %%
# Calcular métricas MSE (Mean Squared Error)
# MSE = Média (y_test - y_pred)2
# Não é uma metrica fácil de interpretar
# MSE é mais sensível a outliers e penaliza grandes erros
mean_squared_error(y_test, y_pred)

# %%
# Calcular métricas RMSE (Square Root Squared Error)
# MSE = Raiz (Média (y_test - y_pred)2)
# É uma metrica fácil de interpretar
# MSE é mais sensível a outliers e penaliza grandes erros
root_mean_squared_error(y_test, y_pred)



# %%
# Analise Gráfica
x_axis = range(len(y_test))
plt.figure(figsize=(10,6))
sns.scatterplot(x=x_axis, y=y_test.reshape(-1), color='blue', label='valores Reais')
sns.scatterplot(x=x_axis, y=y_pred.reshape(-1), color='red', label='valores Preditos')
plt.legend()
plt.show()

# %% [markdown]
# ### Análise de Resíduos
# 

# %%
# Calcular residuos
residuos = y_test - y_pred

# %%
# Calcular os resíduos padronizados (standardization)
# Para cada elemento de um conjunto (X - média) / desvio_padrão
from scipy.stats import zscore
residuos_std = zscore(residuos)

# %%
# Verificar Linearidade do modelo:
# Se os resíduos estiver entre -1 e +2 (na escala padrão) - indica linearidade

# Verificar homogeneidade das variâncias (Homocedasticidade)
# Valores estiver em torno da reta, temos homocedasticidade, caso contrário
# Se tivermos alguma tendência ou padrão (formam um cone, funil), há 
# heterocedasticidade

sns.scatterplot(x=y_pred.reshape(-1), y=residuos_std.reshape(-1))
plt.axhline(y=0)

# %%
# Checar se residuos seguem uma destribuição normal
# QQ (Quantile-Qunatile) Plot, que avlia de uma amostra segue uma distribuição 
# normal
import pingouin as pg
pg.qqplot(residuos_std, dist='norm', confidence=0.95)
plt.xlabel('Quantis Teóricos')
plt.ylabel('Residuos na escala padrão')
plt.show()


# %%
# Teste de Normalidade - Shapiro Wilk
# H0 - Segue distribuição normal
# H1 - não segue distribuição normal
# Se o p>valor > 0.05 não rejeita H0, caso contrário rejeitamos
stat_shapiro, p_valor_shapiro = shapiro(residuos.reshape(-1))
print("Estatística do teste: {} e P-Valor: {}". format(stat_shapiro, p_valor_shapiro))

# %%
# Teste de Normalidade - Kolmogrov-Smirnov
# H0 - Segue distribuição normal
# H1 - não segue distribuição normal
# Se o p>valor > 0.05 não rejeita H0, caso contrário rejeitamos
stat_ks, p_valor_ks = kstest(residuos.reshape(-1), 'norm')
print("Estatística do teste: {} e P-Valor: {}". format(stat_ks, p_valor_ks))

# %% [markdown]
# ### Fazer predições com o modelo

# %%
# Se eu estudar 30.4 horas, qual a pontuação prevista pelo modelo?
reg_model.predict([[30.4]])

# %%
# Quantas horas estudar para obter 600 pontos (pelo modelo)?
# y = ax + b
# y - b = ax
# (y - b) / a = x
# x = (y - b) / a
(600 - reg_model.intercept_[0]) / reg_model.coef_[0][0]

# %% [markdown]
# ## Salvar modelo para usar depois

# %%
import joblib 
joblib.dump(reg_model, './modelo_regressao.pkl')

# %%



