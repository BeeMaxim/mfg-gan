import torch
import torch.nn.functional as F

bias_bool = True

# ===================
# Network definitions
# ===================

class DiscNet(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, psi_func, device, TT):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim + 1, ns, bias=bias_bool) # 3x100  ---->  время t + rho_0
        self.lin2 = torch.nn.Linear(ns, ns, bias=bias_bool) # 100x100
        self.lin3 = torch.nn.Linear(ns, ns, bias=bias_bool) # 100x100
        self.linlast = torch.nn.Linear(int(ns), dim) # 100x1
        self.act_func = act_func # tanh
        self.psi_func = psi_func

        self.dim = dim
        self.hh = hh
        self.TT = TT
        self.device = device

    def forward(self, t, inp, generator):
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
        return c1 * out + c2 * self.psi_func(inp, generator)


class GenNet(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, device, mu, std, TT):
        super().__init__()
        self.mu = mu
        self.std = std

        self.lin1 = torch.nn.Linear(dim + 1, ns) # 3x100  ---->  время t + rho_0
        self.lin2 = torch.nn.Linear(ns, ns) # 100x100
        self.lin3 = torch.nn.Linear(ns, ns) # 100x100
        self.linlast = torch.nn.Linear(int(ns), dim)  # 100x2 ----> rho_t

        # group part
        self.gr_lin1 = torch.nn.Linear(1, ns) # 3x100  ---->  время t + rho_0
        self.gr_lin2 = torch.nn.Linear(ns, ns) # 100x100
        self.gr_lin3 = torch.nn.Linear(ns, ns) # 100x100
        self.gr_linlast = torch.nn.Linear(int(ns), dim)  # 100x2 ----> rho_t

        self.act_func = act_func # relu

        self.dim = dim
        self.hh = hh
        self.TT = TT
        self.device = device

    def forward(self, t, inp, init_groups):
        # inp - 14 values, first 7 - spatial vars, last 7 - groups (sum equal to 1)
        t_normalized = t - self.TT / 2
        # inp_normalized = (inp - self.mu.expand(inp.size())) * (1 / self.std.expand(inp.size()))
        inp_normalized = inp # Mb fix?

        out = torch.cat((t_normalized, inp), dim=1)

        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)


        out_groups = self.act_func(self.gr_lin1(t_normalized))
        out_groups = self.act_func(out_groups + self.hh * self.gr_lin2(out_groups))
        out_groups = self.act_func(out_groups + self.hh * self.gr_lin3(out_groups))
        out_groups = self.gr_linlast(out_groups)

        ctt = t.view(inp.size(0), 1) # меняем размерность t на [[t1],
                                     #                          [t2],
                                     #                          [t3]]
        c1 = ctt / self.TT
        c2 = (self.TT - ctt) / self.TT

        out = F.sigmoid(out)
        out_groups = F.softmax(out_groups, dim=-1)

        return c1 * out + c2 * inp, c1 * out_groups + c2 * init_groups
