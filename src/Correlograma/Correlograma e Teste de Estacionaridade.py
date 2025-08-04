# ---------------------------------------
# 📦 IMPORTAÇÕES
# ---------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima_process import ArmaProcess

# ---------------------------------------
# 🔧 CONFIGURAÇÃO DE GRÁFICOS
# ---------------------------------------
rcParams['figure.figsize'] = 15, 6

# ---------------------------------------
# 🔹 RUÍDO BRANCO
# ---------------------------------------
x1 = np.random.normal(size=200)

plt.plot(x1)
plt.title('Ruído Branco')
plt.show()

plot_acf(x1, lags=30)
plt.title('Correlograma - Ruído Branco')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# Teste ADF
adf_result = adfuller(x1, autolag='AIC')
print('ADF Test: Ruído Branco')
print(f'Estatística do teste: {adf_result[0]:.4f}')
print(f'p-valor: {adf_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Estacionário." if adf_result[1] <= 0.05 else "Falha ao rejeitar H0: Não estacionário.")
print('\n')

# Teste KPSS
kpss_result = kpss(x1, regression='c')
print('KPSS Test: Ruído Branco')
print(f'Estatística do teste: {kpss_result[0]:.4f}')
print(f'p-valor: {kpss_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Não estacionário." if kpss_result[1] <= 0.05 else "Falha ao rejeitar H0: Estacionário.")
print('\n')

# ---------------------------------------
# 🔹 MODELO AR(1): +0.8
# ---------------------------------------
ar1 = np.array([1, -0.8])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
x2 = AR_object1.generate_sample(nsample=200)

plt.plot(x2)
plt.title('Modelo AR(1): X = 0.8Xt-1 + e')
plt.show()

plot_acf(x2, lags=30)
plt.title('Correlograma - AR(1): 0.8')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

adf_result = adfuller(x2, autolag='AIC')
print('ADF Test: AR(1): 0.8')
print(f'Estatística do teste: {adf_result[0]:.4f}')
print(f'p-valor: {adf_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Estacionário." if adf_result[1] <= 0.05 else "Falha ao rejeitar H0: Não estacionário.")
print('\n')

kpss_result = kpss(x2, regression='c')
print('KPSS Test: AR(1): 0.8')
print(f'Estatística do teste: {kpss_result[0]:.4f}')
print(f'p-valor: {kpss_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Não estacionário." if kpss_result[1] <= 0.05 else "Falha ao rejeitar H0: Estacionário.")
print('\n')

# ---------------------------------------
# 🔹 MODELO AR(1): -0.8
# ---------------------------------------
ar2 = np.array([1, 0.8])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
x3 = AR_object2.generate_sample(nsample=200)

plt.plot(x3)
plt.title('Modelo AR(1): X = -0.8Xt-1 + e')
plt.show()

plot_acf(x3, lags=30)
plt.title('Correlograma - AR(1): -0.8')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

adf_result = adfuller(x3, autolag='AIC')
print('ADF Test: AR(1): -0.8')
print(f'Estatística do teste: {adf_result[0]:.4f}')
print(f'p-valor: {adf_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Estacionário." if adf_result[1] <= 0.05 else "Falha ao rejeitar H0: Não estacionário.")
print('\n')

kpss_result = kpss(x3, regression='c')
print('KPSS Test: AR(1): -0.8')
print(f'Estatística do teste: {kpss_result[0]:.4f}')
print(f'p-valor: {kpss_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Não estacionário." if kpss_result[1] <= 0.05 else "Falha ao rejeitar H0: Estacionário.")
print('\n')

# ---------------------------------------
# 🔹 PASSEIO ALEATÓRIO (RANDOM WALK)
# ---------------------------------------
np.random.seed(0)
e = np.random.normal(size=200)
x4 = np.cumsum(e)

plt.plot(x4)
plt.title('Passeio Aleatório')
plt.show()

plot_acf(x4, lags=30)
plt.title('Correlograma - Passeio Aleatório')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

adf_result = adfuller(x4, autolag='AIC')
print('ADF Test: Passeio Aleatório')
print(f'Estatística do teste: {adf_result[0]:.4f}')
print(f'p-valor: {adf_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Estacionário." if adf_result[1] <= 0.05 else "Falha ao rejeitar H0: Não estacionário.")
print('\n')

kpss_result = kpss(x4, regression='c')
print('KPSS Test: Passeio Aleatório')
print(f'Estatística do teste: {kpss_result[0]:.4f}')
print(f'p-valor: {kpss_result[1]:.4f}')
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print(f'{chave}: {valor:.4f}')
print("Resultado:")
print("Rejeitar H0: Não estacionário." if kpss_result[1] <= 0.05 else "Falha ao rejeitar H0: Estacionário.")
print('\n')
