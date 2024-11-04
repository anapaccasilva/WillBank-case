"""
WillBank Case: EDA
Author: Ana Paula Pacca
"""

# Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import category_encoders as ce  # Certifique-se de que category_encoders está instalado

# Importar dataset
url = 'https://raw.githubusercontent.com/anapaccasilva/WillBank_case/main/Case_Bank_Customer.csv'
customers = pd.read_csv(url, sep=",", index_col=0)

# Eliminar outliers de idade
customers2 = customers[customers['age'] < 100]

# Função para criação do gráfico de barras
def barplot_count(dataframe, variavel, titulo):
    df = dataframe.groupby(variavel)[variavel].count().reset_index(name='count')
    total = df['count'].sum()
    df['percent'] = df['count'] / total * 100
    df = df.sort_values('count', ascending=True)

    x = df[variavel]
    y = df['count']
    plt.figure(figsize=(10, 8))
    cor = ['#FFD900', '#414141', '#CACACA', '#FFEA69', '#7B7B7B']
    bars = plt.bar(x, y, color=cor[:len(x)])

    plt.title(titulo)
    plt.xlabel(variavel)
    plt.ylabel('quantidade')
    plt.savefig(titulo + '.png', bbox_inches="tight")

    def addlabels(x, y):
        for i, j in zip(x, y):
            label = f'{j}  ({df.loc[df[variavel] == i, "percent"].iloc[0]:.2f}%)'
            plt.text(i, j, label, ha='center', va='bottom')

    addlabels(x, y)
    plt.show()

# Visualizar variável dependente (Churn)
barplot_count(customers2, 'churn', 'Quantidade de clientes por abandono (desbalanceado)')

# Balancear os dados de churn usando undersampling
df_majority = customers2[customers2.churn == 0]
df_minority = customers2[customers2.churn == 1]

df_majority_undersampled = resample(df_majority,
                                    replace=False,
                                    n_samples=len(df_minority),
                                    random_state=42)

customers_balanceado = pd.concat([df_majority_undersampled, df_minority])

barplot_count(customers_balanceado, 'churn', 'Quantidade de clientes por abandono (balanceado)')

# Separação entre treino e teste
X = customers_balanceado.drop(['churn'], axis=1)
y = customers_balanceado['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Transformar variáveis categóricas em numéricas
encoder = ce.OrdinalEncoder(cols=['country', 'gender'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Modelo Random Forest
modelo_rfc = RandomForestClassifier(n_estimators=10, random_state=123)
modelo_rfc.fit(X_train, y_train)
y_pred = modelo_rfc.predict(X_test)
acc_randomforest = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo usando 10 árvores: {acc_randomforest}')

# Matriz de confusão para Random Forest
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Matriz de confusão: Random Forest Model')
plt.show()

# Modelo Regressão Logística
modelo_logistic = LogisticRegression(solver='lbfgs', max_iter=1000)
modelo_logistic.fit(X_train, y_train)
predictions = modelo_logistic.predict(X_test)
acc_logistic = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo usando regressão logística: {acc_logistic}')

# Matriz de confusão para Regressão Logística
print(classification_report(y_test, predictions, digits=4))
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Matriz de confusão: Logistic Regression Model')
plt.show()

# Modelo Árvore de Decisão
model = DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=5, random_state=123)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
acc_arbol = accuracy_score(y_test, y_predict)
print(f'Acurácia do modelo usando o modelo árvore de decisão: {acc_arbol}')

# Matriz de confusão para Árvore de Decisão
print(classification_report(y_test, y_predict, digits=4))
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Matriz de confusão: Decision Tree Model')
plt.show()

# Seleção do melhor modelo
accuracy_dict = {'Random Forest': acc_randomforest, 'Regressão Logística': acc_logistic, 'Árvore de Decisão': acc_arbol}
max_value = max(accuracy_dict.values())
max_keys = ''.join([k for k, v in accuracy_dict.items() if v == max_value])
print(f'O modelo com maior poder de classificação é o modelo {max_keys} com {max_value} de acurácia')

# Fazendo previsões com o modelo Random Forest em customers3
customers3 = customers.copy()

# Remover a coluna 'churn' caso exista
if 'churn' in customers3.columns:
    customers3 = customers3.drop(columns=['churn'])

# Transformar variáveis categóricas em numéricas em customers3
customers3 = encoder.transform(customers3)

# Previsão das classes
classe = modelo_rfc.predict(customers3)

# Adicionando as previsões e a probabilidade ao dataframe
probs = modelo_rfc.predict_proba(customers3)  # Obtenha as probabilidades
customers3['prob_churn'] = probs[:, 1]  # Adicione a probabilidade de churn
customers3['classe_churn'] = classe  # Adicione a classe prevista

# Criar coluna de risco de churn
risk_func = lambda x: 'Muito Alto' if x >= 0.9 else 'Alto' if x >= 0.7 else 'Moderado' if x >= 0.5 else 'Baixo' if x >= 0.3 else 'Muito Baixo'
customers3['risco_churn'] = customers3['prob_churn'].apply(risk_func)
barplot_count(customers3, 'risco_churn', 'Distribuição de risco de churn')
