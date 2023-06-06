import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


# K=[k32*, k34,    k35,    k36,    k43,    k53,    k63]
K = [2404, 0.0644, 0.0281, 0.0273, 0.0147, 0.1481, 0.1707]
m = 50000  # нг / сутки
s = sum(K[:4])
t = np.linspace(0, 180, 100 * 180 + 1)  # шаг 0.01
M3 = [m / s - m / s * math.exp(-s * time) for time in t]
M4 = [K[1] * m / s * time + K[1] * m / (s ** 2) * math.exp(-s * time) - K[1] * m / s ** 2 for time in t]
M5 = [K[2] * m / s * time + K[2] * m / (s ** 2) * math.exp(-s * time) - K[2] * m / s ** 2 for time in t]
M6 = [K[3] * m / s * time + K[3] * m / (s ** 2) * math.exp(-s * time) - K[3] * m / s ** 2 for time in t]


t_data = [0, 30, 60, 120, 180]
M3_data = [0, 18.5, 20.1, 19.1, 32.2]
M4_data = [0, 63.1, 84.7, 137.3, 166]
M5_data = [0, 63.7, 78.1, 96.1, 89]
M6_data = [0, 105, 102.1, 113, 90.7]
M7_data = [0, 61.8, 87.8, 267.8, 519.1]
M3err_data = [0, 5.5, 3.7, 9, 8.8]
M4err_data = [0, 5.3, 19.1, 41.2, 24]
M5err_data = [0, 19.8, 34.2, 38.7, 19.3]
M6err_data = [0, 38.3, 40.9, 33.4, 15.1]
M7err_data = [0, 33.2, 31.6, 66.7, 95.4]


matplotlib.rcParams['font.family'] = 'times new roman'
matplotlib.rcParams['figure.subplot.left'] = 0.05
matplotlib.rcParams['figure.subplot.bottom'] = 0.07
matplotlib.rcParams['figure.subplot.right'] = 0.99
matplotlib.rcParams['figure.subplot.top'] = 0.99
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.plot(t, M3, 'r-',      linewidth=2.0, label="кровь")
plt.plot(t, M4, 'orange',  linewidth=2.0, label="мозг")
plt.plot(t, M5, 'y-',      linewidth=2.0, label="легкие")
plt.plot(t, M6, 'g-',      linewidth=2.0, label="печень")
plt.scatter(t_data, M3_data, s=25, color='r',       marker='D')
plt.scatter(t_data, M4_data, s=25, color='orange',  marker='D')
plt.scatter(t_data, M5_data, s=25, color='y',       marker='D')
plt.scatter(t_data, M6_data, s=25, color='g',       marker='D')
plt.errorbar(t_data, M3_data, M3err_data, ls='', ecolor='r',        elinewidth=2.0, barsabove=True)
plt.errorbar(t_data, M4_data, M4err_data, ls='', ecolor='orange',   elinewidth=2.0, barsabove=True)
plt.errorbar(t_data, M5_data, M5err_data, ls='', ecolor='y',        elinewidth=2.0, barsabove=True)
plt.errorbar(t_data, M6_data, M6err_data, ls='', ecolor='g',        elinewidth=2.0, barsabove=True)
plt.xlabel("t, сутки", size=12)
plt.ylabel("M, нг", size=12)
plt.legend(fontsize=12)
plt.grid()
stroka = 'k32 = ' + str(round(K[0], 4)) + '\n' + 'k34 = ' + str(round(K[1], 4)) + '\n' + 'k35 = ' + \
         str(round(K[2], 4)) + '\n' + 'k36 = ' + str(round(K[3], 4)) + '\n' + 'k43 = ' + \
         str(round(K[4], 4)) + '\n' + 'k53 = ' + str(round(K[5], 4)) + '\n' + 'k63 = ' + \
         str(round(K[6], 4)) + '\n' + 's = ' + str(round(s, 4))
# plt.text(15, 248, stroka, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment='left', verticalalignment='top', size=12)
plt.show()
