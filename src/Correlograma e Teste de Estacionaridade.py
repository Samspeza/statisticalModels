# Bibliotecas NumPy e Pandas
import numpy as np
import pandas as pd

# Visualização
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

# Testes e processos de séries temporais
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.arima_process import ArmaProcess

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6 
x1 = np.random.normal(size=200)
plt.plot(x1)
plt.title('Ruído Branco')
plt.show()
plot_acf(x1, lags=30)
plt.title('Correlograma - Ruído Branco')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
adf_result = adfuller(x1, autolag='AIC')
print('ADF Test: Ruído Branco')
print('Estatística do teste: {:.4f}'.format(adf_result[0]))
print('p-valor: {:.4f}'.format(adf_result[1]))
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if adf_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados não são estacionários.")
print('\n')
kpss_result = kpss(x1, regression='c')
print('KPSS Test: Ruído Branco')
print('Estatística do teste: {:.4f}'.format(kpss_result[0]))
print('p-valor: {:.4f}'.format(kpss_result[1]))
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if kpss_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados não são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados são estacionários.")
print('\n')
ar1 = np.array([1, -0.8])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
x2 = AR_object1.generate_sample(nsample=200)
plt.figure(figsize=(12, 8))
plt.subplot(211)
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
print('Estatística do teste: {:.4f}'.format(adf_result[0]))
print('p-valor: {:.4f}'.format(adf_result[1]))
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if adf_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados não são estacionários.")
print('\n')
kpss_result = kpss(x2, regression='c')
print('KPSS Test: AR(1): 0.8')
print('Estatística do teste: {:.4f}'.format(kpss_result[0]))
print('p-valor: {:.4f}'.format(kpss_result[1]))
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if kpss_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados não são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados são estacionários.")
print('\n')
ar2 = np.array([1, 0.8])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
x3 = AR_object2.generate_sample(nsample=200)
plt.figure(figsize=(12, 8))
plt.plot(x3)
plt.title('Modelo AR(1): X = -0.8Xt-1 + e')
plot_acf(x3, lags=30)
plt.title('Correlograma - AR(1): -0.8')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
adf_result = adfuller(x3, autolag='AIC')
print('ADF Test: AR(1): -0.8')
print('Estatística do teste: {:.4f}'.format(adf_result[0]))
print('p-valor: {:.4f}'.format(adf_result[1]))
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if adf_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados não são estacionários.")
print('\n')
kpss_result = kpss(x3, regression='c')
print('KPSS Test: AR(1): -0.8')
print('Estatística do teste: {:.4f}'.format(kpss_result[0]))
print('p-valor: {:.4f}'.format(kpss_result[1]))
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if kpss_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados não são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados são estacionários.")
print('\n')
np.random.seed(0)
e = np.random.normal(size=200)
x4 = np.cumsum(e)
plt.figure(figsize=(12, 8))
plt.plot(x4)
plt.title('Passeio Aleatório')
plt.show()
plot_acf(x4, lags=30)
plt.title('PAsseio Aleatorio')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()
adf_result = adfuller(x4, autolag='AIC')
print('ADF Test: Passeio Aleatório')
print('Estatística do teste: {:.4f}'.format(adf_result[0]))
print('p-valor: {:.4f}'.format(adf_result[1]))
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if adf_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados não têm raiz unitária e são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados têm raiz unitária e não são estacionários.")
print('\n')
kpss_result = kpss(x4, regression='c')
print('KPSS Test: Passeio Aleatório')
print('Estatística do teste: {:.4f}'.format(kpss_result[0]))
print('p-valor: {:.4f}'.format(kpss_result[1]))
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
if kpss_result[1] <= 0.05:
    print("Rejeitar a hipótese nula (H0): Os dados não são estacionários.")
else:
    print("Falha ao rejeitar a hipótese nula (H0): Os dados são estacionários.")
print('\n')
