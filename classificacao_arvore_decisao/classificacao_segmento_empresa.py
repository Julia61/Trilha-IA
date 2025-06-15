# %%
# pipenv install pandas plotly matplotlib pingouin nbformat ipykernel scikit-learn optuna ipywidgets gradio

#EDA
import pandas as pd
import pingouin as pg
import plotly.express as px 
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


# Machine Learning
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay

##Otimização de Hiperpâmetros
import optuna

# %%
# pipenv install pandas plotly matplotlib pingouin nbformat ipykernel scikit-learn optuna ipywidgets gradio

#EDA
import pandas as pd
import pingouin as pg
import plotly.express as px 
import plotly.figure_factory as ff
import matplotlib.pyplot as plt


# Machine Learning
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,classification_report, ConfusionMatrixDisplay

##Otimização de Hiperpâmetros
import optuna

# %%
# Carregar Dataset
df_segmento = pd.read_csv('./datasets/dataset_segmento_clientes.csv')

# %% [markdown]
# ### EDA

# %%
# Visualizar os dados
df_segmento.head(10)

# %%
# Estrutura do Dataset
df_segmento.info()

# %%
# Valores possíveis - Variáveis Categóricas
df_segmento['atividade_economica'].unique()

# %%
# Valores possíveis - Variáveis Categóricas
df_segmento['localizacao'].unique()

# %%
# Valores possíveis - Variáveis Categóricas
df_segmento['segmento_de_cliente'].unique()

# %%
# Valores possíveis - Variáveis Categóricas
df_segmento['inovacao'].unique()

# %%
# Distribuição da Variável Segmento de Cliente (Target)
contagem_target = df_segmento.value_counts('segmento_de_cliente')
contagem_target

# %%
# Criar uma lista ordenada do target
lista_segmentos = ['Starter', 'Bronze', 'Silver', 'Gold']

# %%
# Distribuição da Variável Target - Contagem
px.bar(contagem_target, color=contagem_target.index, category_orders={'segmento_de_cliente': lista_segmentos})

# %%
# Distribuição da Variável Target - Percentual
percentual_target = contagem_target / len(df_segmento) * 100
px.bar(percentual_target, color=percentual_target.index, category_orders={'segmento_de_cliente': lista_segmentos})


# %%
# Distribuição da Variável Localização
percentual_localizacao = df_segmento.value_counts('localizacao') / len(df_segmento) * 100
px.bar(percentual_localizacao, color=percentual_localizacao.index)


# %%
# Distribuição da Variável Atividade Econônimica
percentual_atividade = df_segmento.value_counts('atividade_economica') / len(df_segmento) * 100
px.bar(percentual_atividade, color=percentual_atividade.index)

# %%
# Distribuição da Variável Inovação
percentual_inovacao = df_segmento.value_counts('inovacao') / len(df_segmento) * 100
px.bar(percentual_inovacao, color=percentual_inovacao.index)

# %%
# Tabela de Contingência entre Localização e Target
crossstab_localizacao = pd.crosstab(df_segmento['localizacao'], df_segmento['segmento_de_cliente'], margins=True)[lista_segmentos].reset_index()

tabela_localizacao = ff.create_table(crossstab_localizacao)

# Mostrar a Crosstab
tabela_localizacao.show()

# %%
# Tabela de Contingência entre Atividade  e Target
crossstab_atividade = pd.crosstab(df_segmento['atividade_economica'], df_segmento['segmento_de_cliente'], margins=True)[lista_segmentos].reset_index()

tabela_atividade= ff.create_table(crossstab_atividade)

# Mostrar a Crosstab
tabela_atividade.show()

# %%
# Tabela de Contingência entre Inovação  e Target
crossstab_inovacao = pd.crosstab(df_segmento['inovacao'], df_segmento['segmento_de_cliente'], margins=True)[lista_segmentos].reset_index()

tabela_inovacao = ff.create_table(crossstab_inovacao)

# Mostrar a Crosstab
tabela_inovacao.show()

# %%
# Distribuição Idade da Empresa
px.histogram(df_segmento, x='idade')

# %%
# Distribuição Faturamento Mensal
px.histogram(df_segmento, x='faturamento_mensal')

# %%
# BoxPlot entre Idade e Segmento
px.box(df_segmento, x='segmento_de_cliente', y='idade', color='segmento_de_cliente', category_orders={'segmento_de_cliente': lista_segmentos})

# %%
# BoxPlot entre Faturamento Mensal e Segmento
px.box(df_segmento, x='segmento_de_cliente', y='faturamento_mensal', color='segmento_de_cliente', category_orders={'segmento_de_cliente': lista_segmentos})

# %%
# Teste de Qui-Quadrado de Pearson
# H0 - as variáveis são independentes
# H1 - as variáveis não são independentes
# Se p-value > 0.05, aceita a hipótese nula, caso contário rejeita
valor_esperado, valor_observado, estatisticas = pg.chi2_independence(df_segmento, 'segmento_de_cliente', 'inovacao')


# %%
# Valor Esperado
# É a frequência que seria esperada se não houvesse associação entre variáveis
# É calculado utlizando a distribuição assumida no teste qui-quadrado
valor_esperado


# %%
# Valor observado
# É a frequencia real dos dados coletados
valor_observado

# %%
# Estatísticas
estatisticas.round(5)

# %% [markdown]
# As variáveis localização e segmento de clientes são independentes. Qui-Quadrado (p-value = 0.817)
# As variáveis atividade econômica e segmento de clientes são independentes. Qui-Quadrado (p-value = 0.35292)
# As variáveis inovação e segmento não são indpendentes. Qui-Quadrado (p-value = 0.0)
# 
# 
# 

# %% [markdown]
# ### Treinamento do Modelo

# %%
# Separar X e y 
X = df_segmento.drop(columns=['segmento_de_cliente'])
y = df_segmento['segmento_de_cliente']

# %%
# Pipeline 
# OneHotEncode nas variáveis categóricas
# Treinamento do Modelo

# Lista de variáveis categóricas
categorical_features = ['atividade_economica', 'localizacao']

# Criar um tranformador de variáveis categóricas usando OneHotEncoder
categorical_tranformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_tranformer, categorical_features)
    ]
)

# Pipeline com Pre-Processor e o Modelo de Arvore de Decisão
dt_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier())])



# %% [markdown]
# ### Validação Cruzada

# %%
# Treinar o Modelo com Validação Cruzada, usando StratifiedKFold, dado que as classes estão desbalanceada

cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=51)
metrics_result = cross_validate(dt_model, X, y, cv=cv_folds, scoring=['accuracy'], return_estimator=True)

# %%
# Mostrar Retorno do Cross Validation
metrics_result

# %%
# Média da Acurácia, considerando os 3 splits
metrics_result['test_accuracy'].mean()

# %%
# Acurácia
# total de previsões corretas / total de previsões

# %% [markdown]
# ### Métricas

# %%
# Fazendo predições usando Cros Validation
y_pred = cross_val_predict(dt_model, X, y, cv=cv_folds)

# %%
# Avalidar o desemprenho do modelo
classification_report_str = classification_report(y, y_pred)

print(f'Relatório de classificação:\n{classification_report_str}')

# %%
# Mostrar Matriz de Confusão
confusion_matrix_modelo = confusion_matrix(y, y_pred, labels=lista_segmentos)
disp = ConfusionMatrixDisplay(confusion_matrix_modelo, display_labels=lista_segmentos)
disp.plot()

# %% [markdown]
# ### Tuning de Hiperparâmetros

# %%
## Ajustar hiperparâmentros do Modelo usando Optuna
# min_samples_leaf = Mínimo de instâncias requerido para formar uma folha (nó terminal) 
# max_depth = Profundidade máxima da árvore 

def decisiontree_optuna(trial):

    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_depth = trial.suggest_int('max_depth', 2, 8)

    dt_model.set_params(classifier__min_samples_leaf=min_samples_leaf)
    dt_model.set_params(classifier__max_depth=max_depth)

    scores = cross_val_score(dt_model, X, y, cv=cv_folds, scoring='accuracy')

    return scores.mean()

# %%
# Executar a automação de experimentos
estudo_decisiontree = optuna.create_study(direction='maximize')
estudo_decisiontree.optimize(decisiontree_optuna, n_trials=200)

# %%
# Mostrar melhor resultado e melhor conjunto de hiperparâmetros
print(f'Melhor acurácia: {estudo_decisiontree.best_value}')
print(f'Melhores parâmetros: {estudo_decisiontree.best_params}')


# %% [markdown]
# ### Visualizar Árvore

# %%
# Preparar o Conjunto de Dados para treinar e conseguir visualizar a árvore
X_train_tree = X.copy()
X_train_tree['localizacao_label'] = X_train_tree.localizacao.astype('category').cat.codes
X_train_tree['atividade_economica_label'] = X_train_tree.atividade_economica.astype('category').cat.codes
X_train_tree.drop(columns=['localizacao', 'atividade_economica'], axis=1, inplace=True)
X_train_tree.rename(columns={'localizacao_label': 'localizacao', 'atividade_economica_label': 'atividade_economica'}, inplace=True)
X_train_tree.head(10)



# %%
# Treinar o modelo com o conjunto de hiperparâmetros ideal

clf_decisiontree = DecisionTreeClassifier(min_samples_leaf=estudo_decisiontree.best_params['min_samples_leaf'], max_depth=estudo_decisiontree.best_params['max_depth'])

y_train_tree = y.copy()

clf_decisiontree.fit(X_train_tree, y_train_tree)

# %%
# Visualizar Árvore de Decisão com Plot Tree
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10), dpi=600)

plot_tree(clf_decisiontree,
          feature_names=X_train_tree.columns.to_numpy(),
          class_names=lista_segmentos,
          filled= True)

# %% [markdown]
# ### Salvar Modelo

# %%
import joblib

# Criar um pipeline "tunado"
dt_model_tunado = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier',
                             DecisionTreeClassifier(
                                 min_samples_leaf=estudo_decisiontree.best_params['min_samples_leaf'],
                                 max_depth=estudo_decisiontree.best_params['max_depth'])
)])

# Treinar Modelo Tunado
dt_model_tunado.fit(X, y)

# Salvar Modelo
joblib.dump(dt_model_tunado, 'modelo_classificacao_decision_tree.pkl')

# %% [markdown]
# ### Entregar modelo com App de Predição Batch (por arquivo)

# %%
import gradio as gr

modelo = joblib.load('./modelo_classificacao_decision_tree.pkl')

def predict(arquivo):
    df_empresas = pd.read_csv(arquivo.name)
    y_pred = modelo.predict(df_empresas)
    df_segmentos = pd.DataFrame(y_pred, columns=['segmento_de_cliente'])
    df_predicoes = pd.concat([df_empresas, df_segmentos], axis=1)
    df_predicoes.to_csv('./predicoes.csv',index=False)
    return './predicoes.csv'

demo = gr.Interface(
    predict,
    gr.File(file_types=[".csv"]),
    "file"
)

demo.launch()


