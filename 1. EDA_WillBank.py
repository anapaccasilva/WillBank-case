"""
WillBank Case: EDA
Author: Ana Paula Pacca
"""

# Importar bibliotecas necessárias
from pandas import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

# Importar o dataset
url = 'https://raw.githubusercontent.com/anapaccasilva/WillBank_case/main/Case_Bank_Customer.csv'
customers = pd.read_csv(url, sep=",", index_col=0)

# EDA - Análise Exploratória dos Dados

# Panorama geral
print(customers.shape)
print(customers.describe())
print(customers.info())
print(customers.isnull().sum())
print(customers.empty)
print(customers.isna().sum())

# Limpeza de dados e ajustes no dataset
customers = customers.replace(['German', 'Grmany'], 'Germany')
customers['gender'] = customers['gender'].fillna('Não Informado')
customers['active_member'] = customers['active_member'].replace({0: 'Inativo', 1: 'Ativo'})
customers['churn'] = customers['churn'].replace({0: 'Não', 1: 'Sim'})
customers['credit_card'] = customers['credit_card'].replace({0: 'Não', 1: 'Sim'})

# Função para criação de gráfico de barras com contagem
def barplot_count(dataframe, variavel, titulo):
    df = dataframe.groupby(variavel)[variavel].count().reset_index(name='count')
    total = df['count'].sum()
    df['percent'] = df['count'] / total * 100  # Calcula a quantidade relativa em porcentagem

    x = df[variavel]
    y = df['count']
    plt.figure(figsize=(10,8))

    # Construir gráfico de barras usando cores aleatórias a cada execução
    cor = ['#FFD900', '#414141', '#CACACA']
    bars = plt.bar(x, y, color=cor[:len(x)])

    # Adiciona título ao gráfico, aos eixos e salva a figura
    plt.title(titulo)
    plt.xlabel(variavel)
    plt.ylabel('quantidade')

    # Função para adicionar rótulos às colunas
    for i in range(len(x)):
        label = f'{y[i]}  ({df["percent"][i]:.2f}%)'
        plt.text(i, y[i], label, ha='center', va='bottom')

    plt.savefig(titulo + '.png', bbox_inches="tight")
    plt.show()

# Função para histograma simples
def histograma1(dataframe, variavel, titulo):
    plt.figure(figsize=(10,8))
    cor = ['#FFD900', '#414141', '#CACACA']
    colors = cor[random.randint(0,2)]  # Cor aleatória
    plt.hist(dataframe[variavel], edgecolor='white', color=colors)

    plt.title(titulo)
    plt.xlabel(variavel)
    plt.ylabel('Frequência')
    plt.savefig(titulo + '.png', bbox_inches="tight")
    plt.show()

# Função para histograma com contagem por categoria
def histograma2(dataframe, variavel, titulo):
    plt.figure(figsize=(10,8))
    df = dataframe.groupby(variavel)[variavel].count().reset_index(name='count')
    x = df[variavel]
    y = df['count']
    cor = ['#FFD900', '#414141', '#CACACA']
    colors = cor[random.randint(0,2)]  # Cor aleatória
    plt.bar(x, y, color=colors)

    plt.title(titulo)
    plt.xlabel(variavel)
    plt.ylabel('Contagem')
    plt.savefig(titulo + '.png', bbox_inches="tight")
    plt.show()

# Função para criar boxplot
def boxplot(dataframe, X, Y, titulo):
    sns.catplot(x=X, y=Y, data=dataframe, kind="box", aspect=1.5, palette=['#FFD900', '#414141'])
    sns.set(style="darkgrid")
    plt.title(titulo)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.savefig(titulo + '.png', bbox_inches="tight")
    plt.show()

# Função para gráfico de barras empilhadas
def stacked_bar(dataframe, X, Y, titulo):
    cross_tab = pd.crosstab(index=dataframe[Y], columns=dataframe[X])
    cross_tab_prop = pd.crosstab(index=dataframe[Y], columns=dataframe[X], normalize='index')

    cross_tab_prop.plot(kind='bar', stacked=True, color=['#FFD900', '#414141'], figsize=(10, 6))
    plt.legend(title=X, loc="lower left", ncol=2)
    plt.xlabel(Y)
    plt.ylabel("Proporção")
    plt.title(titulo)

    # Adicionar rótulos com as proporções nas barras
    for n, x in enumerate(cross_tab.index.values):
        for (proportion, y_loc) in zip(cross_tab_prop.loc[x], cross_tab_prop.cumsum(axis=1).loc[x]):
            plt.text(n, y_loc - proportion / 2, f'{proportion * 100:.1f}%', ha='center', va='center', color='white')

    plt.savefig(titulo + '.png', bbox_inches="tight")
    plt.show()

# Exemplo de uso das funções de visualização
barplot_count(customers, 'country', 'Quantidade de clientes por país')
barplot_count(customers, 'gender', 'Quantidade de clientes por gênero')
histograma1(customers, "credit_score", "Distribuição do Score de Crédito")
histograma2(customers, "age", "Distribuição de Idades")
boxplot(customers, "churn", "credit_score", "Boxplot Abandono x Score de Crédito")
stacked_bar(customers, 'credit_card', 'churn', 'Proporção de Churn por Possuir Cartão de Crédito')
