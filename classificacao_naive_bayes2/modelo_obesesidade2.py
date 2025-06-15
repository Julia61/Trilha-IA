# %%
# EDA
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt 
import sweetviz as sv

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score

# Otimização de Hiperparâmetros
import optuna

# %%
# Carregar Dataset
df_obsesidade = pd.read_csv('./datasets/dataset_obesidade.csv')

# %%
df_obsesidade.head(5)

# %%
# Mostrar os ultimos registros do Dataframe
df_obsesidade.tail(5)

# %%
# Mostrar estrutura / schema do Dataframe
df_obsesidade.info()

# %%
# Converter colunas para tipo inteiro
lista_colunas = ['Idade', 'Consumo_Vegetais_Com_Frequencia', 'Refeicoes_Dia', 'Consumo_Agua', 'Nivel_Atividade_Fisica', 'Nivel_Uso_Tela']

df_obsesidade[lista_colunas] = df_obsesidade[lista_colunas].astype(int)

# %%
# Detectar valores ausentes
df_obsesidade.isna().sum()

# %% [markdown]
# ### EDA

# %%
# Distribuição da variável Target - Obsesidade
px.bar(df_obsesidade.value_counts('Obesidade'))

# %%
# Distribuição da variável Target - Obsesidade
px.bar(df_obsesidade.value_counts('Obesidade') / len(df_obsesidade) * 100)

# %%
# Analise Univariada
px.histogram(df_obsesidade, x='Idade')

# %%
# Analise Univariada - Idade
px.box(df_obsesidade, y='Idade')

# %%
# Analise Univariada - Genero
px.bar(df_obsesidade.value_counts('Genero_Masculino') / len(df_obsesidade) * 100)

# %%
# Analise Univariada - Historico Obsesidade na familia
px.bar(df_obsesidade.value_counts('Historico_Familiar_Sobrepeso') / len(df_obsesidade) * 100)

# %%
# Analise Univariada - Nivel_Atividade_Fisica
px.bar(df_obsesidade.value_counts('Nivel_Atividade_Fisica') / len(df_obsesidade) * 100)

# %%
# Analise Univariada - Nivel de Uso de Tela
px.bar(df_obsesidade.value_counts('Nivel_Uso_Tela') / len(df_obsesidade) * 100)

# %%
# Formulação de Hipótese
# Faixa Etária influencia em Obesidade
df_obsesidade['Idade'].describe()

# %%
# Bucketing de Idade
bins = [10, 20, 30, 40, 50, 60, 70]
bins_ordinal = [0, 1, 2, 3, 4, 5]
labels_faixa_etaria = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70']
df_obsesidade['Faixa_Etaria_String'] = pd.cut(x = df_obsesidade['Idade'], bins=bins, labels=labels_faixa_etaria, include_lowest=True)
df_obsesidade['Faixa_Etaria'] = pd.cut(x = df_obsesidade['Idade'], bins=bins, labels=bins_ordinal, include_lowest=True)


# %%
df_obsesidade

# %%
# Criar uma Tabela de Contigência - Faixa Etária e Obesidade
tabela_contingencia_faixa_etaria = sm.stats.Table.from_data(df_obsesidade[['Obesidade', 'Faixa_Etaria_String']])

# %%
tabela_contingencia_faixa_etaria.table_orig

# %%
# Teste de Chi_square de Pearson
print(tabela_contingencia_faixa_etaria.test_nominal_association())

# %%
# p_value < 0.05, rejeitamos H0, portanto as variáveis não são independentes

# %%
# Automatizar EDA
sv_obesidade_report = sv.analyze(df_obsesidade, target_feat='Obesidade')

# %%
sv_obesidade_report.show_notebook()

# %% [markdown]
# ### Treinamento do Modelo - Baseline

# %%
# Dividir Dados de Treino e Teste
X = df_obsesidade.drop(columns=['Obesidade', 'Idade', 'Faixa_Etaria_String'], axis=1)
y = df_obsesidade['Obesidade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51, shuffle=True)

# %%
# Treinamento do Modelo
model_baseline = GaussianNB()
model_baseline.fit(X_train, y_train)


# %% [markdown]
# ### Métricas Modelo Baseline

# %%
# Predizer valores do conjunto de testes
y_pred = model_baseline.predict(X_test)

# %%
# Avaliando o desempenho do modelo
classification_report_str = classification_report(y_test, y_pred)
recall_baseline = recall_score(y_test, y_pred, average='macro')
print(f'Relatório de Classificação\n{classification_report_str}')
print(f'Recall\n{recall_baseline}')


# %%
# Mostrar Matriz de Confusão
confusion_matrix_modelo_baseline = confusion_matrix(y_test, y_pred)
disp_modelo_baseline = ConfusionMatrixDisplay(confusion_matrix_modelo_baseline)
disp_modelo_baseline.plot()

# %% [markdown]
# ### Treinamento Modelo - Automated Feature Selection

# %%
# Select KBest - Seleciona as K melhores features, baseado em um teste
kbest = SelectKBest(score_func=chi2, k= 8)

# %%
X_train_kbest = kbest.fit_transform(X_train, y_train)

# %%
X_train_kbest

# %%
# Features Selecionadas
kbest_features = kbest.get_support(indices=True)
X_train_best_features = X_train.iloc[:, kbest_features]
X_train_best_features.info()

# %%
kbest_features

# %%
# Treinar Modelo com 5 melhores features
modelo_kbest = GaussianNB()
modelo_kbest.fit(X_train_best_features, y_train)

# %% [markdown]
# ### Métricas Modelo Select KBest

# %%
# Filtrar as features nos dados de teste
X_test_kbest = kbest.transform(X_test)
X_test_best_features = X_test.iloc[:, kbest_features]

# %%
# Predizer valores do conjunto de testes
y_pred_kbest = modelo_kbest.predict(X_test_best_features)

# %%
# Avaliando o desempenho do modelo
classification_report_str = classification_report(y_test, y_pred_kbest)
recall_baseline = recall_score(y_test, y_pred_kbest, average='macro')
print(f'Relatório de Classificação (KBest)\n{classification_report_str}')
print(f'Recall (KBest)\n{recall_baseline}')


# %%
# Mostrar Matriz de Confusão
confusion_matrix_modelo_kbest = confusion_matrix(y_test, y_pred_kbest)
disp_modelo_kbest = ConfusionMatrixDisplay(confusion_matrix_modelo_kbest)
disp_modelo_kbest.plot()

# %% [markdown]
# ### Tuning de Hiperparâmetros
# 

# %%
# Ajustar hiperparâmetros de SelectKBest
# k = k melhores features conforme chi2

def naivebayes_optuna(trial):

    k = trial.suggest_int('k', 1, 18)

    kbest = SelectKBest(score_func=chi2, k= k)

    X_train_kbest = kbest.fit_transform(X_train, y_train)

    kbest_features = kbest.get_support(indices=True)
    X_train_best_features = X_train.iloc[:, kbest_features]

    # Treinar modelo com melhores features
    modelo_kbest_optuna = GaussianNB()
    modelo_kbest_optuna.fit(X_train_best_features, y_train)

    # Aplicar o seletor de features no conjunto de testes
    X_test_kbest = kbest.transform(X_test)
    X_test_best_features = X_test.iloc[:, kbest_features]

    # Predizer valores
    y_pred_kbest = modelo_kbest_optuna.predict(X_test_best_features)

    # Avaliar Recall
    recall_optuna = recall_score(y_test, y_pred_kbest, average='macro')

    return k, recall_optuna

# %%
# Rodar o estudo dos hiperparametros
search_space = {'k': range(1,19)}
estudo_naivebayes = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space=search_space), directions=['minimize', 'maximize'])
estudo_naivebayes.optimize(naivebayes_optuna, n_trials=50)

# %%
# Mostrar melhor resultado
trial_com_melhor_recall = max(estudo_naivebayes.best_trials, key=lambda t: t.values[1])
print("Trial com maior recall e menor k:")
print(f'\ttrial number: {trial_com_melhor_recall.number}')
print(f'\tparam: {trial_com_melhor_recall.params}')
print(f'\tvalues (k, recall): {trial_com_melhor_recall.values}')



# %%
# Mostrar Chart com Trials
fig = optuna.visualization.plot_pareto_front(estudo_naivebayes)
fig.show()

# %% [markdown]
# ### Salvar Modelo

# %%
import joblib

# Salvar o modelo 
joblib.dump(modelo_kbest, 'modelo_obesesidade.pkl')


