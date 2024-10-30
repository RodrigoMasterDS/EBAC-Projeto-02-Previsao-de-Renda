#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import os
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Projeto 02 | Previsão de Renda",
    page_icon="https://e3ba6e8732e83984.cdn.gocache.net/uploads/image/file/530714/regular_ebac-logo.png",
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.markdown('''
<div style="text-align:center">
<img src="https://github.com/RodrigoMasterDS/Rodrigo-EBAC-DS/blob/main/newebac_logo_black_half.png?raw=true" alt="ebac-logo">
</div>
                    
---
                    
# **Profissão: Cientista de Dados**
### **Projeto #02** | Previsão de Renda

Aluno [Rodrigo Costa Fonseca Pacheco]
[LinkedIn](https://www.linkedin.com/in/rodrigodatascience/)<br>
Data: 19 de outubro de 2023.
''', unsafe_allow_html=True)

with st.sidebar.expander("Bibliotecas/Pacotes", expanded=False):
    st.code('''
    import streamlit as st
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    ''', language='python')

# Verifica se o arquivo existe
filepath = "C:/Users/rodri/Documents/G I T  H U B/EBAC-Projeto-02-Previsao-de-Renda/input/previsao_de_renda.csv"
if not os.path.exists(filepath):
    st.error(f"Arquivo não encontrado: {filepath}. Verifique o caminho do arquivo.")
else:
    dfrenda = pd.read_csv(filepath_or_buffer=filepath)
    
    # Limpeza e preparação dos dados
    dfrenda.drop(columns=['Unnamed: 0', 'id_cliente'], inplace=True, errors='ignore')
    dfrenda.drop_duplicates(inplace=True, ignore_index=True)
    dfrenda.drop(columns='data_ref', inplace=True, errors='ignore')
    dfrenda.dropna(inplace=True)
    dfrenda_dummies = pd.get_dummies(data=dfrenda)

    # Separação das variáveis independentes (X) e dependentes (y)
    X = dfrenda_dummies.drop(columns='renda')
    y = dfrenda_dummies['renda']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Criação e treinamento do modelo
    reg_tree = DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_leaf=4)
    reg_tree.fit(X_train, y_train)

    st.header("Simulando a previsão de renda")
    with st.form("my_form"):
        st.subheader("Preencha os campos abaixo:")
        
        sexo = st.radio("Sexo", ('M', 'F'))
        veiculo = st.checkbox("Posse de veículo")
        imovel = st.checkbox("Posse de imóvel")
        filhos = st.number_input("Quantidade de filhos", 0, 15)
        tiporenda = st.selectbox("Tipo de renda", [
            'Sem renda', 'Empresário', 'Assalariado', 'Servidor público', 'Pensionista', 'Bolsista'])
        if tiporenda == 'Sem renda':
            tiporenda = None
        educacao = st.selectbox("Educação", [
            'Primário', 'Secundário', 'Superior incompleto', 'Superior completo', 'Pós graduação'])
        estadocivil = st.selectbox(
            "Estado civil", ['Solteiro', 'União', 'Casado', 'Separado', 'Viúvo'])
        residencia = st.selectbox("Tipo de residência", [
            'Casa', 'Governamental', 'Com os pais', 'Aluguel', 'Estúdio', 'Comunitário'])
        idade = st.slider("Idade", 18, 100)
        tempoemprego = st.slider("Tempo de emprego", 0, 50)
        qtdpessoasresidencia = st.number_input(
            "Quantidade de pessoas na residência", 1, 15)

        submitted = st.form_submit_button("Simular")
        if submitted:
            entrada = pd.DataFrame([{
                'sexo': sexo,
                'posse_de_veiculo': veiculo,
                'posse_de_imovel': imovel,
                'qtd_filhos': filhos,
                'tipo_renda': tiporenda,
                'educacao': educacao,
                'estado_civil': estadocivil,
                'tipo_residencia': residencia,
                'idade': idade,
                'tempo_emprego': tempoemprego,
                'qt_pessoas_residencia': qtdpessoasresidencia
            }])

            # Concatenar a entrada com as dummies
            entrada = pd.concat([entrada, pd.get_dummies(entrada)], axis=1)
            entrada = entrada.reindex(columns=X.columns, fill_value=0)  # Alinha as colunas com o modelo

            # Prever a renda
            renda_estimativa = reg_tree.predict(entrada)
            st.write(
                f"Renda estimada: R${np.round(renda_estimativa.item(), 2):,.2f}".replace('.', ','))

