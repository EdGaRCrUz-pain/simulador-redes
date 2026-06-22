pip install streamlit matplotlib numpy -q
npm install -g localtunnel -q

%%writefile app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("Simulador de Redes Neuronales (Prototipo)")

fila_superior_col1, fila_superior_col2 = st.columns(2)
fila_inferior_col1, fila_inferior_col2 = st.columns(2)

with fila_superior_col1:
    st.header("EDITOR")
    st.subheader("Configuración de la Red")
    capas = st.selectbox("Capas", [1, 2, 3, 4], index=1)
    neuronas = st.selectbox("Neuronas por capa", [2, 4, 8, 16, 32], index=2)
    optimizador = st.selectbox("Optimizadores", ["Momentum", "Gradiente D", "ADAM"])
    act_func = st.selectbox("Funciones de act.", ["tansig", "logsig", "relu"])
    operaciones = st.selectbox("Operaciones mat.", ["Suma matricial", "Producto punto"])
    funciones_mat = st.selectbox("Funciones mat.", ["Derivada", "Gradiente"])
    carga_datos = st.selectbox("Carga de datos", ["Conjunto XOR", "Regresión Lineal", "Subir CSV"])
    btn_simular = st.button("SIMULAR", use_container_width=True)

with fila_superior_col2:
    st.header("RED GRÁFICA")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis('off')
    entradas = 3
    for i in range(entradas):
        ax.scatter(0, i, s=300, c='white', edgecolors='black', zorder=3)
        ax.text(-0.5, i, f"x_{i+1}", va='center', ha='right')
    for j in range(neuronas):
        ax.scatter(1, j - (neuronas-entradas)/2, s=300, c='white', edgecolors='black', zorder=3)
        for i in range(entradas):
            ax.plot([0, 1], [i, j - (neuronas-entradas)/2], c='gray', lw=0.5, zorder=1)
    ax.scatter(2, 1, s=300, c='white', edgecolors='black', zorder=3)
    ax.text(2.5, 1, "y_1", va='center', ha='left')
    for j in range(neuronas):
        ax.plot([1, 2], [j - (neuronas-entradas)/2, 1], c='gray', lw=0.5, zorder=1)
    st.pyplot(fig)

with fila_inferior_col1:
    st.header("CÓDIGO")
    st.write("# Código de ejemplo generado")
    codigo_dinamico = f"""# Parámetros del simulador
alfa = 0.01;
optimizador = '{optimizador}';

for Epocas = 1:500
    Sum = 0;
    for = 1:Q
        a0 = rand(Q)
        a1 = {act_func}(W1 * P(:, q) + b1)
        a2 = {act_func}(W2 * a1 + b2)
    # Optimización usando {optimizador}
"""
    st.code(codigo_dinamico, language="matlab")

with fila_inferior_col2:
    st.header("ANÁLISIS")
    if btn_simular:
        st.success(f"Simulación completada usando {optimizador}")
        fig_analisis, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        puntos = np.random.rand(15)
        ax1.scatter(range(15), puntos, color='blue', s=10)
        ax1.set_title("Dispersión del Error")
        x = np.linspace(0, 10, 100)
        if optimizador == "ADAM":
            y = np.exp(-x) + np.random.normal(0, 0.02, 100)
        else:
            y = np.exp(-x*0.5) + np.random.normal(0, 0.05, 100)
        ax2.plot(x, y, color='green')
        ax2.set_title("Función de Pérdida")
        st.pyplot(fig_analisis)
    else:
        st.info("Presiona el botón 'SIMULAR' en el Editor para ver los resultados.")
