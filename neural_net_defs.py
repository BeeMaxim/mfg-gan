import torch

bias_bool = True

# ===================
# Network definitions
# ===================

class DiscNet(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, psi_func, device, TT):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim+1, ns, bias=bias_bool) # 3x100  ---->  время t + rho_0
        self.lin2 = torch.nn.Linear(ns, ns, bias=bias_bool) # 100x100
        self.lin3 = torch.nn.Linear(ns, ns, bias=bias_bool) # 100x100
        self.linlast = torch.nn.Linear(int(ns), 1) # 100x1
        self.act_func = act_func # tanh
        self.psi_func = psi_func

        self.dim = dim
        self.hh = hh
        self.TT = TT
        self.device = device

    def forward(self, t, inp):
        # Центрирование: вместо [0; 1] будет [-0.5; 0.5]
        t_normalized = t - self.TT/2

        out = torch.cat((t_normalized, inp), dim=1)

        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        ctt = t.view(inp.size(0), 1)
        c1 = (self.TT - ctt) / self.TT  # convex weight 1
        c2 = ctt / self.TT  # convex weight 2

        # В момент времени T нейросеть должна возвращать self.psi_func(inp = rho_T ?)
        return c1 * out + c2 * self.psi_func(inp).view(inp.size(0), 1)


class GenNet(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, device, mu, std, TT):
        super().__init__()
        self.mu = mu
        self.std = std

        self.lin1 = torch.nn.Linear(dim + 1, ns) # 3x100  ---->  время t + rho_0
        self.lin2 = torch.nn.Linear(ns, ns) # 100x100
        self.lin3 = torch.nn.Linear(ns, ns) # 100x100
        self.linlast = torch.nn.Linear(int(ns), dim)  # 100x2 ----> rho_t
        self.act_func = act_func # relu

        self.dim = dim
        self.hh = hh
        self.TT = TT
        self.device = device

    def forward(self, t, inp):
        # Центрирование: вместо [0; 1] будет [-0.5; 0.5]
        t_normalized = t - self.TT/2
        # Стандартизация: вместо inp = rho0 будет (inp - mean) / std
        inp_normalized = (inp - self.mu.expand(inp.size())) * (1 / self.std.expand(inp.size()))

        out = torch.cat((t_normalized, inp_normalized), dim=1)

        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        ctt = t.view(inp.size(0), 1) # меняем размерность t на [[t1],
                                     #                          [t2],
                                     #                          [t3]]
        c1 = ctt / self.TT  # convex weight 1
        c2 = (self.TT - ctt) / self.TT  # convex weight 2

        # G(z,t) = t*N_theta(x, t) + (1-t)*z; N_theta(x, t) = out, z=inp
        return c1 * out + c2 * inp
