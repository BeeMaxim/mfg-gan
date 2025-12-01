import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
import matplotlib.pyplot as plt

def rho_i_func(x):
    x_i = 0.5
    sigma_i = 0.5
    a_i = np.exp(-(1-x_i)**2 / (2*sigma_i**2)) * (1-x_i) / (2*sigma_i**3 * np.sqrt(2*np.pi))
    b_i = np.exp(-x_i**2 / (2*sigma_i**2)) * x_i / (2*sigma_i**3 * np.sqrt(2*np.pi))

    return np.exp(-(x-x_i)**2 / (2*sigma_i**2)) / (sigma_i * np.sqrt(2*np.pi)) + a_i*x**2 + b_i*(1-x**2)

def rho0(i=0):
    S0 = 2769113
    E0 = 462
    I0 = 2520
    R0 = 6193
    H0 = 1845
    C0 = 26
    D0 = 129
    N = 2798170

    cam_vec = [S0, E0, I0, R0, H0, C0, D0]
    cam_dict = {
        0: 'S0',
        1: 'E0',
        2: 'I0',
        3: 'R0',
        4: 'H0',
        5: 'C0',
        6: 'D0'
    }


    for i, CAM_i in enumerate(cam_vec):
        A_i = CAM_i / N
        B_i = quad(rho_i_func, 0, 1)[0]

        print(f'Camera {cam_dict[i]}:')
        print(f'A_{cam_dict[i]} = {A_i}')
        print(f'B_{cam_dict[i]} = {B_i}')
        print('--------------')

        x = np.linspace(0, 1, 1000)
        y = A_i/B_i * rho_i_func(x)

        # Строим график
        plt.figure(figsize=(8, 6))  # Размер графика
        plt.plot(x, y, label=f'$rho_{cam_dict[i]}$', color='blue', linewidth=2)  # Линия графика
        plt.xlabel('x')  # Подпись оси X
        plt.ylabel(f'$rho_{cam_dict[i]}(x)$')  # Подпись оси Y
        plt.title(f'График функции $rho_{cam_dict[i]}(x)$ на отрезке [0, 1]')  # Заголовок
        plt.grid(True)  # Включение сетки
        plt.legend()  # Отображение легенды
        plt.show()  # Показать график

        F = cumulative_trapezoid(y, x, initial=0)
        F /= F[-1]  # Нормируем, чтобы F(1) = 1
        # Интерполируем обратную функцию F^{-1}(u)
        from scipy.interpolate import interp1d
        inv_cdf = interp1d(F, x, fill_value="extrapolate")

        # Генерируем 50 семплов
        u = np.random.rand(50)
        samples = inv_cdf(u)

        print(samples)
        print(np.mean(samples))
        print(np.std(samples))




rho0()