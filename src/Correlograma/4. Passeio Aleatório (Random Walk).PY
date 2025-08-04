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

# Teste ADF
adf_result = adfuller(x4, autolag='AIC')
print('ADF Test: Passeio Aleatório')
print('Estatística do teste: {:.4f}'.format(adf_result[0]))
print('p-valor: {:.4f}'.format(adf_result[1]))
print('Valores Críticos:')
for chave, valor in adf_result[4].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
print("Rejeitar H0: Estacionário." if adf_result[1] <= 0.05 else "Falha ao rejeitar H0: Não estacionário.")
print('\n')

# Teste KPSS
kpss_result = kpss(x4, regression='c')
print('KPSS Test: Passeio Aleatório')
print('Estatística do teste: {:.4f}'.format(kpss_result[0]))
print('p-valor: {:.4f}'.format(kpss_result[1]))
print('Valores Críticos:')
for chave, valor in kpss_result[3].items():
    print('{}: {:.4f}'.format(chave, valor))
print('Resultado:')
print("Rejeitar H0: Não estacionário." if kpss_result[1] <= 0.05 else "Falha ao rejeitar H0: Estacionário.")
print('\n')
