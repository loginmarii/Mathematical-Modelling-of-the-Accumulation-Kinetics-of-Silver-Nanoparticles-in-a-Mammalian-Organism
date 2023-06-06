import numpy as np
from scipy.integrate import odeint
import scipy.optimize as spo
import matplotlib
import matplotlib.pyplot as plt


# M = [0, 0, 0, M3,  M4,  M5,  M6]
# M0= [0, 0, 0, M30, M40, M50, M60]
# K = [k32, k34, k35, k36]
# K = [0    1    2    3  ]


def model(M, t, K):
    dM3dt = -(K[0] + K[1] + K[2] + K[3]) * M[3] + mass
    dM4dt = K[1] * M[3]
    dM5dt = K[2] * M[3]
    dM6dt = K[3] * M[3]
    return [0, 0, 0, dM3dt, dM4dt, dM5dt, dM6dt]


def deviation(K):
    massiv = odeint(model, M0, t, args=(K,))
    dev_M3 = ((massiv[t_0][3] - 0) ** 2) + \
             ((massiv[t_30][3] - 18.5) ** 2) / (5.5 ** 2) + \
             ((massiv[t_60][3] - 20.1) ** 2) / (3.7 ** 2) + \
             ((massiv[t_120][3] - 19.1) ** 2) / (9 ** 2) + \
             ((massiv[t_180][3] - 32.2) ** 2) / (8.8 ** 2)
    dev_M4 = ((massiv[t_0][4] - 0) ** 2) + \
             ((massiv[t_30][4] - 63.1) ** 2) / (5.3 ** 2) + \
             ((massiv[t_60][4] - 84.7) ** 2) / (19.1 ** 2) + \
             ((massiv[t_120][4] - 137.3) ** 2) / (41.2 ** 2) + \
             ((massiv[t_180][4] - 166) ** 2) / (24 ** 2)
    dev_M5 = ((massiv[t_0][5] - 0) ** 2) + \
             ((massiv[t_30][5] - 63.7) ** 2) / (19.8 ** 2) + \
             ((massiv[t_60][5] - 78.1) ** 2) / (34.2 ** 2) + \
             ((massiv[t_120][5] - 96.1) ** 2) / (38.7 ** 2) + \
             ((massiv[t_180][5] - 89) ** 2) / (19.3 ** 2)
    dev_M6 = ((massiv[t_0][6] - 0) ** 2) + \
             ((massiv[t_30][6] - 105) ** 2) / (38.3 ** 2) + \
             ((massiv[t_60][6] - 102.1) ** 2) / (40.9 ** 2) + \
             ((massiv[t_120][6] - 113) ** 2) / (33.4 ** 2) + \
             ((massiv[t_180][6] - 90.7) ** 2) / (15.1 ** 2)
    return dev_M3 + dev_M4 + dev_M5 + dev_M6


time_limit = 180
mass = 50000  # нг / сутки
if time_limit >= 180:
    t = np.linspace(0, time_limit, 100 * time_limit + 1)  # шаг 0.01
else:
    t = np.linspace(0, 180, 100 * 180 + 1)  # шаг 0.01
t_0 = 100 * 0  # поскольку шаг по оси t в 100 раза меньше суток, то 30-е сутки будут 3000-ым элементом массива
t_30 = 100 * 30
t_60 = 100 * 60
t_120 = 100 * 120
t_180 = 100 * 180
M0 = [0, 0, 0, 0, 0, 0, 0]
K_start = [2500, 0.1268, 0.5875, 0.8209, 0.0147, 0.1481, 0.1707]


result = spo.minimize(deviation, K_start, options={'disp': True})
if result.success:
    print('Success!')
else:
    print('Sorry, could not find a minimum')
M = odeint(model, M0, t, args=(result.x,))


t_data = [0, 30, 60, 120, 180]
M3_data = [0, 18.5, 20.1, 19.1, 32.2]
M4_data = [0, 63.1, 84.7, 137.3, 166]
M5_data = [0, 63.7, 78.1, 96.1, 89]
M6_data = [0, 105, 102.1, 113, 90.7]
M3err_data = [0, 5.5, 3.7, 9, 8.8]
M4err_data = [0, 5.3, 19.1, 41.2, 24]
M5err_data = [0, 19.8, 34.2, 38.7, 19.3]
M6err_data = [0, 38.3, 40.9, 33.4, 15.1]


matplotlib.rcParams['font.family'] = 'times new roman'
matplotlib.rcParams['figure.subplot.left'] = 0.05
matplotlib.rcParams['figure.subplot.bottom'] = 0.07
matplotlib.rcParams['figure.subplot.right'] = 0.99
matplotlib.rcParams['figure.subplot.top'] = 0.99
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.plot(t, M[:, 3], 'r-',      linewidth=2.0, label="кровь")
plt.plot(t, M[:, 4], 'orange',  linewidth=2.0, label="мозг")
plt.plot(t, M[:, 5], 'y-',      linewidth=2.0, label="легкие")
plt.plot(t, M[:, 6], 'g-',      linewidth=2.0, label="печень")
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
stroka = 'k32 = ' + str(round(result.x[0], 4)) + '\n' + 'k34 = ' + str(round(result.x[1], 4)) + '\n' + 'k35 = ' + \
         str(round(result.x[2], 4)) + '\n' + 'k36 = ' + str(round(result.x[3], 4)) + '\n' + 'k43 = ' + \
         str(round(result.x[4], 4)) + '\n' + 'k53 = ' + str(round(result.x[5], 4)) + '\n' + 'k63 = ' + \
         str(round(result.x[6], 4)) + '\n' + 'otkl = ' + str(round(result.fun, 4)) + '\n' + 's = ' + str(round(sum(result.x[:4]), 4))
# plt.text(15, 248, stroka, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment='left', verticalalignment='top', size=12)
plt.show()


"""
# разность аналитического и численного решения
M3 = [m / s - m / s * math.exp(-s * time) for time in t]
M4 = [K[1] * m / s * time + K[1] * m / (s ** 2) * math.exp(-s * time) - K[1] * m / s ** 2 for time in t]
M5 = [K[2] * m / s * time + K[2] * m / (s ** 2) * math.exp(-s * time) - K[2] * m / s ** 2 for time in t]
M6 = [K[3] * m / s * time + K[3] * m / (s ** 2) * math.exp(-s * time) - K[3] * m / s ** 2 for time in t]
rM3 = [M3[time] - M[time][3] for time in range(len(t))]
rM4 = [M4[time] - M[time][4] for time in range(len(t))]
rM5 = [M5[time] - M[time][5] for time in range(len(t))]
rM6 = [M6[time] - M[time][6] for time in range(len(t))]


plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.plot(t, rM3, 'r-',      linewidth=1.0, label="кровь")
plt.plot(t, rM4, 'orange',  linewidth=1.0, label="мозг")
plt.plot(t, rM5, 'y-',      linewidth=1.0, label="легкие")
plt.plot(t, rM6, 'g-',      linewidth=1.0, label="печень")
plt.xlabel("t, сутки", size=18)
plt.ylabel("M, нг", size=18)
plt.legend(fontsize=12)
plt.grid()
plt.show()
"""


# bnds = ((40, 50), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))  # границы для возможных значений k
# cons = ({'type': 'eq', 'fun': lambda xy: (2 * xy[0] + xy[1] - 100)})  # дополнительные условия, выражение в скобках должно быть равно нулю
# cons = ({'type': 'ineq', 'fun': lambda xy: (2 * xy[0] + xy[1] - 100)})  # когда нужно условие: выражение в скобках >= 0
# result = spo.minimize(f, xy_start, options={'disp': True}, constraints=cons, bounds=bnds)


# print(M[t_0][3], M[t_30][3], M[t_60][3], M[t_120][3], M[t_180][3])
# print(M[t_0][4], M[t_30][4], M[t_60][4], M[t_120][4], M[t_180][4])
# print(M[t_0][5], M[t_30][5], M[t_60][5], M[t_120][5], M[t_180][5])
# print(M[t_0][6], M[t_30][6], M[t_60][6], M[t_120][6], M[t_180][6])


# print('K =', result.x, 'min_deviation =', result.fun)
# print(stroka)
# print('s =', sum(result.x[:4]))
