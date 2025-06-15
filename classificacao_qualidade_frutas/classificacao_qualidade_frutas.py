# %%
# EDA
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pingouin import ttest

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_curve, auc, log_loss,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Otimização de Hiperparâmetros
import optuna


# %% [markdown]
# ### Carregar os Dados

# %%
df_frutas = pd.read_csv('./datasets/fruit_quality.csv')

# %%
# Visualizar DataFrame
df_frutas.head(10)

# %%
# Visualizar DataFrame
df_frutas.tail(10)

# %%
# Estrutura do Dataframe
df_frutas.info()

# %% [markdown]
# ### EDA

# %%
# Distribuição da Variável Target - Percentual 
px.bar(df_frutas.value_counts('Quality') / len(df_frutas) * 100)

# %%
# Tranformar a variável Quality em numérica (0 e 1)
df_frutas['Quality'] = (df_frutas['Quality'] == 'good').astype(int)

# %%
# Remover a coluna A_id, pois não tem poder preditivo
df_frutas.drop(columns=['A_id'], axis=1, inplace=True)

# %%
# Verificar distribuição e a correlação de variáveis numa forma visual
sns.pairplot(df_frutas, diag_kind='hist')

# %%
# BoxPlot Quality x Weight
px.box(df_frutas, x='Quality', y='Weight', color='Quality')

# %%
# BoxPlot Quality x Sweetness
px.box(df_frutas, x='Quality', y='Sweetness', color='Quality')

# %%
# BoxPlot Quality x Size
px.box(df_frutas, x='Quality', y='Size', color='Quality')

# %%
# Teste de T-Student
# Um teste estatístico para verificar se existe uma diferença significativa entre as médias de 2 grupos
# H0 - Não há diferença significativa entre as médias dos grupos
# H1 - Há há diferença significativa entre as médias dos grupos
grupo_good_weight = df_frutas[df_frutas['Quality'] == 1]['Weight']
grupo_bad_weight = df_frutas[df_frutas['Quality'] == 0]['Weight']
ttest(x=grupo_good_weight, y=grupo_bad_weight, paired=False)



# %% [markdown]
# Não há diferença significatica das médias de peso entre frutas boas ruins

# %%
grupo_good_sweetness = df_frutas[df_frutas['Quality'] == 1]['Sweetness']
grupo_bad_sweetness = df_frutas[df_frutas['Quality'] == 0]['Sweetness']
ttest(x=grupo_good_sweetness, y=grupo_bad_sweetness, paired=False)

# %% [markdown]
# Há diferença significatica das médias de doçura entre frutas boas ruins

# %%
grupo_good_size = df_frutas[df_frutas['Quality'] == 1]['Size']
grupo_bad_size = df_frutas[df_frutas['Quality'] == 0]['Size']
ttest(x=grupo_good_size, y=grupo_bad_size, paired=False)

# %% [markdown]
# Há diferença significatica das médias de tamanho entre frutas boas ruins

# %%
# Cor Matrix
corr_matrix = df_frutas.corr()
corr_matrix

# %%
# plot Heatmap
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr_matrix.columns,
        y = corr_matrix.index,
        z = np.array(corr_matrix),
        text= corr_matrix.values,
        texttemplate='%{text:.2f}',
        colorscale=px.colors.diverging.RdBu,
        zmin=-1,
        zmax=1
    )
)
fig.show()

# %% [markdown]
# ### Treinar Modelo Baseline

# %%
X = df_frutas.drop(columns=['Quality'], axis=1)
y = df_frutas['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)

# %%
# Criar o Objeto do Algoritmo Logistico Regression
# Para datasets pequenos, o solver liblinear é um dos indicados na documentação do sklearn
model_lr = LogisticRegression(solver='liblinear')

# %%
model_lr.fit(X_train, y_train)

# %% [markdown]
# ### Métricas Baseline

# %%
# Retornar a classificação predita com base no conjunto de testes
y_pred = model_lr.predict(X_test)

# %%
# Decision Function retorna o valor calculado (score) de cada instância, considerando os coeficientes obtidos da reta de regressão
y_decision = model_lr.decision_function(X_test)

# %%
# Retornar as propabilidades de cada classes para cada instâncias no conjunto de testes
y_prob = model_lr.predict_proba(X_test)

# %%
y_prob

# %%
# Retornar os valores da curva ROC - TPR (True Positive Rate), FPR (False Positive Rate), Threshold
fpr, tpr, thresholds = roc_curve(y_test, y_decision)

# %%
# Calcular AUC (Area Under The Curve), com base nos valores da curva ROC
roc_auc = auc(fpr, tpr)
roc_auc

# %%
# Plotar Curva ROC com o valor de AUC
fig = px.area(
    x=fpr, y=tpr,
    title=f'Curva ROC (AUC={roc_auc:.4f})',
    labels=dict(x='FPR', y='TPR'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

fig.show()


# %%
# Apresentar a importância das Features (com base nos coeficientes obtidos na regressão)
importance = np.abs(model_lr.coef_)

# Exibir a importância das Features
print("Importância das Features")
for i, feature in enumerate(model_lr.feature_names_in_):
    print(f'{feature}: {importance[0][i]}')

# %%
# F1 Score é uma média harmônica entre Precisão e Recall 
f1_score_baseline = f1_score(y_test, y_pred)
f1_score_baseline

# %%
# Apresentar BCE (Binary Cross Entropy) - Log Loss
log_loss(y_test, y_pred)

# %%
# Mostrar Matriz de Confusão
confusion_matrix_modelo = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix_modelo)
disp.plot()


# %% [markdown]
# ### Otimizar Hiperparâmetros

# %%
# Ajustar hiperparâmetros com Optuna

# Hiperparâmetro penalty
# Regularização controla a complexidade do modelo, reduzindo overfitting
# L1 (Ridge) é útil para fins de Feature Selection e para modelos esparsos. Soma dos valos absolutos dos coeficientes
# L2 (Lasso) é útil para evitar o overfitting, principalmente quando há multicolinearidade. Soma dos quadrados dos coeficientes

# Hiperparâmetro C
# Valores maiores de C, indica uma regularização mais fraca
# Valores menores de C, indica uma regularização mais forte


def lr_optuna(trial):

    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    c_values = trial.suggest_categorical('c', [100, 10, 1.0, 0.1, 0.01])

    # Treinar modelo
    model_lr_optuna = LogisticRegression(solver='liblinear', penalty=penalty, C=c_values)
    model_lr_optuna.fit(X_train, y_train)

    # Retornar a Decision Function
    y_decision_optuna = model_lr_optuna.decision_function(X_test)

    # Calcular Curva ROC
    fpr, tpr, threshols = roc_curve(y_test, y_decision_optuna)

    # Calcular AUC
    roc_auc_optuna = auc(fpr, tpr)

    # Retornar Predição
    y_pred_optuna = model_lr_optuna.predict(X_test)

    # Calcular F1 Score
    f1_score_optuna = f1_score(y_test, y_pred_optuna, average='macro')

    # Calcular BCE (Binary Cross Entropy - Erro)
    log_loss_optuna = log_loss(y_test, y_pred_optuna)

    return roc_auc_optuna, f1_score_optuna, log_loss_optuna

# %%
# Criação do Estudo e Rodar Otimizador
search_space = {'penalty': ['l1', 'l2'], 'c': [100, 10, 1.0, 0.1, 0.01]}
sampler = optuna.samplers.GridSampler(search_space=search_space)
estudo_lr = optuna.create_study(directions=['maximize', 'maximize', 'minimize'])
estudo_lr.optimize(lr_optuna, n_trials=20)

# %%
# Mostrar melhor resultado e melhor conjunto de hiperparâmetros
melhor_trial = max(estudo_lr.best_trials, key=lambda t: t.values[1])
print(f'Trial com melhor AUC e F1 e menor BCE:')
print(f'\nnumber: {melhor_trial.number}')
print(f'\nnumber: {melhor_trial.params}')
print(f'\nnumber: {melhor_trial.values}')



# %%
# mostrar Chart com Trial do Estudo
fig = optuna.visualization.plot_pareto_front(estudo_lr)
fig.show()

# %%
# Comparação entre melhor resultado da otimização e baseline
print(f'AUC: baseline={roc_auc} optuna={melhor_trial.values[0]}')

# %%
# Comparação entre melhor resultado da otimização e baseline - F1 Score
print(f'F1: baseline={f1_score_baseline} optuna={melhor_trial.values[1]}')

# %%
# Comparação entre melhor resultado da otimização e baseline - BCE
print(f'BCE: baseline={log_loss(y_test, y_pred)} optuna={melhor_trial.values[2]}')

# %% [markdown]
# ### Verificar métricas com Thresholds diferentes

# %%
# Fazer um loop e apresentar o F1 Score para cada Threshold
lista_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

lista_resultados = {'cenario': [], 'resultado': []}
lista_resultados['cenario'].append('baseline')
lista_resultados['resultado'].append(f1_score_baseline)
lista_resultados['cenario'].append('optuna')
lista_resultados['resultado'].append(melhor_trial.values[1])

for novo_threshold in lista_thresholds:
    y_pred_threshold = (model_lr.predict_proba(X_test)[:, 1] >= novo_threshold).astype(int)
    f1_score_threshold = f1_score(y_test, y_pred_threshold, average='macro')
    lista_resultados['cenario'].append(str(novo_threshold))
    lista_resultados['resultado'].append(f1_score_threshold)

# %%
# Criar DataFrame com Resultados
df_resultados_thresholds = pd.DataFrame(lista_resultados)

# %%
df_resultados_thresholds

# %%
# Apresentar resultado dos F1 scores
px.line(df_resultados_thresholds    , x='cenario', y='resultado')

# %% [markdown]
# ### Salvar Modelo

# %% [markdown]
# 

# %%
import joblib

# Salvar Modelo
joblib.dump(model_lr, 'modelo_qualidade_frutas.pkl')

# %%



