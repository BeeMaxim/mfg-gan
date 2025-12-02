import math
import numpy as np
import pandas as pd
import torch
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import interp1d
from utils.utils import DISC_STRING, GEN_STRING, sqeuc


# ==========================================
#           SEIR-HCD Environment
# ==========================================
class SEIRHCD_Env(object):
    """
    SEIR-HCD environment.
    """
    def __init__(self, device):
        self.dim = 7
        self.x_c = 0.5
        self.sigma_c = 0.2
        self.d1 = 1e-5
        self.d2 = 10

        self.init_count = [2769113, 462, 2520, 6193, 1845, 26, 129]

        # ---------------------- #
        self.nu = 0 # 0.1
        self.TT = 1
        self.ham_scale = 5
        self.psi_scale = 1
        self.lam_obstacle = 5
        self.lam_congestion = 1
        self.device = device
        self.name = "SEIRHCD_Env"  # Environment name

        # HARDCODED COEFS, FIX!
        self.a = 1
        self.alpha_i = 0.06
        self.alpha_e = 0.11
        self.w_inc = 0.21
        self.w_inf = 0.15
        self.w_imm = 0.006
        self.w_hosp = 0.33
        self.w_crit = 0.16
        self.beta = 0.22
        self.eps_hc = 0.01
        self.mu = 0.34

        # Options for plotting
        self.plot_window_size = 3
        self.plot_dim = 2

        data_path  = 'C:\\Users\\User\\Desktop\MFG_GAN_model\\total_data_SEIR_HCD_Novosibirsk.xlsx'

        params = pd.read_excel(data_path)

        self.a = params['a'].values
        self.alpha_i = params['alpha_I'].values
        self.alpha_e = params['alpha_E'].values
        self.w_inc = params['w_inc'].values
        self.w_inf = params['w_inf'].values
        self.w_imm = params['w_imm'].values
        self.w_hosp = params['w_hosp'].values
        self.w_crit = params['w_crit'].values
        self.beta = params['beta'].values
        self.eps_hc = params['eps_HC'].values
        self.mu = params['mu'].values

        # Параметры модели SEIR-HCD
        self.info_dict = {'env_name': self.name, 'dim': self.dim, 'nu': self.nu,
                          'ham_scale': self.ham_scale, 'psi_scale': self.psi_scale,
                          'lam_obstacle': self.lam_obstacle, 'lam_congestion': self.lam_congestion}
        

    def _rho_i_func(self, x):
        x_i = 0.5
        sigma_i = 0.5
        a_i = np.exp(-(1-x_i)**2 / (2*sigma_i**2)) * (1-x_i) / (2*sigma_i**3 * np.sqrt(2*np.pi))
        b_i = np.exp(-x_i**2 / (2*sigma_i**2)) * x_i / (2*sigma_i**3 * np.sqrt(2*np.pi))

        return np.exp(-(x-x_i)**2 / (2*sigma_i**2)) / (sigma_i * np.sqrt(2*np.pi)) + a_i*x**2 + b_i*(1-x**2)


    # The initial distribution rho_0 of the agents
    def sample_rho0(self, num_samples):
        """
        The initial distribution rho_0 of the agents.
        """
        
        B_i = quad(self._rho_i_func, 0, 1)[0]

        x = np.linspace(0, 1, 1000)
        y = B_i * self._rho_i_func(x)

        F = cumulative_trapezoid(y, x, initial=0)
        F /= F[-1]  # Нормируем, чтобы F(1) = 1
        # Интерполируем обратную функцию F^{-1}(u)
        
        inv_cdf = interp1d(F, x, fill_value="extrapolate")

        u = torch.rand((num_samples, self.dim))
        samples = torch.FloatTensor(inv_cdf(u))

        groups = torch.zeros((samples.shape), dtype=torch.float32)
        for i in range(7):
            groups[:, i] = self.init_count[i] / sum(self.init_count)

        return samples, groups
    

    # The final-time cost function.
    def psi_func(self, xx_inp, generator):
        """
        The final-time cost function.
        """
        t = torch.zeros((xx_inp.size(0), 1)) + self.TT
        rho_est = self._estimate_rho(generator, t, xx_inp)

        out = torch.zeros_like(xx_inp).to(xx_inp.device)
        out[:, 2] = 2 * self.d2 * rho_est[:, 2] # I group

        return out

    # The Hamiltonian
    # ham = env.ham(tt_samples, rhott_samples, (-1) * phi_grad_xx)
    def ham(self, generator, tt, xx, pp, phi_vals):
        """
        The Hamiltonian.
        """
        out = torch.zeros_like(xx).to(xx.device)
        out -= pp**2 / 2

        a = torch.zeros(tt.size(0))
        alpha_i = torch.zeros(tt.size(0))
        alpha_e = torch.zeros(tt.size(0))
        w_inc = torch.zeros(tt.size(0))
        w_inf = torch.zeros(tt.size(0))
        w_imm = torch.zeros(tt.size(0))
        w_hosp = torch.zeros(tt.size(0))
        w_crit = torch.zeros(tt.size(0))
        beta = torch.zeros(tt.size(0))
        eps_hc = torch.zeros(tt.size(0))
        mu = torch.zeros(tt.size(0))

        for i in range(xx.size(0)):
            index = int(tt.detach()[i] * len(self.a))
            a[i] = self.a[i]
            alpha_i[i] = self.alpha_i[index]
            alpha_e[i] = self.alpha_e[index]
            w_inc[i] = self.w_inc[index]
            w_inf[i] = self.w_inf[index]
            w_imm[i] = self.w_imm[index]
            w_hosp[i] = self.w_hosp[index]
            w_crit[i] = self.w_crit[index]
            beta[i] = self.beta[index]
            eps_hc[i] = self.eps_hc[index]
            mu[i] = self.mu[index]

        rho_est = self._estimate_rho(generator, tt, xx)

        out[:, 0] = (5 - a) / 5 * (phi_vals[:, 0] - phi_vals[:, 1]) * (alpha_i * rho_est[:, 2] + alpha_e * rho_est[:, 1])
        out[:, 1] = (5 - a) / 5 * (phi_vals[:, 0] - phi_vals[:, 1]) * alpha_e * rho_est[:, 0] + w_inc * (phi_vals[:, 2] - phi_vals[:, 3])
        out[:, 2] = (5 - a) / 5 * alpha_i * rho_est[:, 0] * (phi_vals[:, 0] - phi_vals[:, 1]) + w_inf * (phi_vals[:, 2] - phi_vals[:, 4]) + \
        + beta * w_inf * (phi_vals[:, 4] - phi_vals[:, 3])
        
        out[:, 3] = w_imm * (phi_vals[:, 3] - phi_vals[:, 0])
        out[:, 4] = w_hosp * (phi_vals[:, 4] - phi_vals[:, 3]) + eps_hc * w_hosp * (phi_vals[:, 3] - phi_vals[:, 5])
        out[:, 5] = w_crit * (phi_vals[:, 5] - phi_vals[:, 4]) + mu * w_crit * (phi_vals[:, 4] - phi_vals[:, 6])

        return out


    # Вычисляем Лаплассиан (вторая производная grad=phi_grad_xx по xx=rhott_samples)
    # get_hjb_loss --> env.get_trace(phi_grad_xx, rhott_samples, batch_size, env.dim, grad_outputs_vec)
    def get_trace(self, grad, xx, batch_size, dim, grad_outputs_vec):
        """
        Computation of the second-order term in the HJB equation.
        """
        # Вычисляем 2-ю производную phi (7*50,1) по rhott (7*50,7). grad_outputs_vec на диагоналях имеет единицы, остальные нули.
        # Это позволяет распараллелить вычисление второй производно по всей видимости.
        # Затем через stack всё собираем и снова делаем размерность (50, 7), где 50 это batch_size
        hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                                           grad_outputs=grad_outputs_vec,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Для каждого измерения i=0,..,6 извлекает только диагональные элементы гессиана (вторые производные по одному и тому же измерению)
        pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i + 1) * batch_size, i]
                                     for i in range(0, dim)], dim=1)

        # Вычисление лапласиана: Суммирует диагональные элементы гессиана по всем измерениям.
        # Результат: (50,) - лапласиан для каждого из 50 элементов
        laplacian = torch.sum(pre_laplacian, dim=1)
        laplacian_sum_repeat = laplacian.repeat((1, dim))

        return laplacian_sum_repeat.T


    # Вычисляем ограничивающую функцию f(x,t)
    def FF_func(self, td, tt, rhott_samples, disc_or_gen):
        """
        Computes the forcing term (i.e. obstacle or interaction between agents) of the
        Hamilton-Jacobi equation.
        """
        out = torch.zeros_like(rhott_samples).to(rhott_samples.device)
        rho_est = rho_est = self._estimate_rho(td['generator'], tt, rhott_samples)

        out[:, 1] = 2 * self.d1 * rho_est[:, 1]
        out[:, 2] = 2 * self.d1 * rho_est[:, 2]
        out[:, 3] = -2 * self.d1 * (1 - rho_est[:, 3])
        out[:, 6] = -2 * self.d1 * (1 - rho_est[:, 6])

        return out

    
    def _estimate_rho(self, generator, t, x, use_samples=50): # x - first half of output of generator B x dim
        """
        x - tensor of B x dim
        t - tensor of B x 1
        """
        h = 0.1
        out = torch.zeros_like(x).to(x.device)

        rho00, init_groups = self.sample_rho0(use_samples)
        rho00 = rho00.to(x.device)

        for b in range(x.size(0)):
            rhott_samples, groups = generator(torch.repeat_interleave(t[b], use_samples).unsqueeze(1), rho00, init_groups)
            rhott_samples = rhott_samples.detach().requires_grad_(True)
            groups = groups.detach().requires_grad_(True)

            diffs = (x[b] - rhott_samples) / h
            kernel_vals = torch.exp(-0.5 * diffs**2) / (2 * torch.pi)**0.5
            out[b, :] = torch.mean(kernel_vals, dim=0) / h

            out[b, :] *= groups[0] # different density across groups

        return out
