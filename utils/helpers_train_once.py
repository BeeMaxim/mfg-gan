import torch
from utils.utils import uniform_time_sampler, DISC_STRING, GEN_STRING


# ================
# Helper functions
# ================


def estimate_rho(td, t, x, use_samples=100): # x - first half of output of generator B x dim
    """
    x - tensor of B x dim
    t - tensor of B x 1
    """
    h = 0.1
    out = torch.zeros_like(x).to(td['device'])

    rho00 = td['env'].sample_rho0(use_samples).to(td['device'])

    for b in range(x.size(0)):
        rhott_samples = td['generator'](t[b], rho00).detach().requires_grad_(True) # mb crush (mismatch)
        diffs = (x[b] - rhott_samples) / h
        kernel_vals = torch.exp(-0.5 * diffs**2) / torch.sqrt(2 * torch.pi)
        out = torch.mean(kernel_vals) / h

    return out



def set_requires_grad(discriminator, generator, disc_or_gen):
    """
    Turn on requires_grad for the one we're training, and turn off for the one we aren't. For speed.
    """
    if disc_or_gen == DISC_STRING:
        # Включаем градиенты только для дискриминатора
        for param in discriminator.parameters():
            param.requires_grad_(True)
        # Отключаем градиенты генератора
        for param in generator.parameters():
            param.requires_grad_(False)
    elif disc_or_gen == GEN_STRING:
        # Отключаем градиенты дискриминатора
        for param in discriminator.parameters():
            param.requires_grad_(False)
        # Включаем градиенты только для генератора
        for param in generator.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


# Зануляем градиенты оптимизатора генератора/дискриминатора перед его обучением заданном батче
def set_zero_grad(disc_optimizer, gen_optimizer, disc_or_gen):
    """
    Zero the gradients for the one we're training.
    """
    if disc_or_gen == DISC_STRING:
        disc_optimizer.zero_grad()
    elif disc_or_gen == GEN_STRING:
        gen_optimizer.zero_grad()
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


# Генерируем генератором batch_size сэмплов rho_t, где t генерируется из равномерного распределения на отрезке от 0 до T
# Также возвращаем batch_size сэмплов rho0 и tt
def get_generator_samples(td, disc_or_gen):
    """
    Get generator samples.
    """
    # генерируем rho0 batch_size штук. На выходе получим вектор [batch_size x dim=2]
    quantiles, rho00, init_groups = td['env'].sample_rho0(td['batch_size'])
    rho00 = rho00.to(td['device'])
    init_groups = init_groups.to(td['device'])

    # генерируем вектор времени из равномерного распределения от 0 до T размером
    # Если td['TT'] уже тензор на GPU, .item() автоматически перенесет значение на CPU
    tt_samples = (td['TT'][0].item() * uniform_time_sampler(td['batch_size'])).to(td['device'])

    if disc_or_gen == DISC_STRING:
        pdf_params, groups = td['generator'](tt_samples, rho00, init_groups)
        rhott_samples = td['env'].sample_rhot(quantiles, pdf_params.cpu().detach().numpy()[..., 0], pdf_params.cpu().detach().numpy()[..., 1])
        rhott_samples = rhott_samples.detach().requires_grad_(True).to(groups.device)
        groups = groups.detach().requires_grad_(True)
    elif disc_or_gen == GEN_STRING:
        pdf_params, groups = td['generator'](tt_samples, rho00, init_groups)
        rhott_samples = td['env'].sample_rhot(quantiles, pdf_params.cpu().detach().numpy()[..., 0], pdf_params.cpu().detach().numpy()[..., 1])
        rhott_samples = rhott_samples.detach().requires_grad_(True).to(groups.device)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')

    return rho00, tt_samples, pdf_params, rhott_samples, groups


def get_kfp_loss(td, tt_samples, rhott_samples, rho_estimation, phi_grad_xx, ones_of_size_rho_out):
    """
    Compute the KFP error
    """
    env = td['env']
    out = torch.zeros_like(rhott_samples)

    for i in range(env.dim):
        phi_grad_tt = torch.autograd.grad(outputs=rho_estimation[:, i],
                                        inputs=tt_samples,
                                        grad_outputs=ones_of_size_rho_out[:, i],
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        
        rho_grad_xx = torch.autograd.grad(outputs=rho_estimation * 0, # * -phi_grad_xx,
                                      inputs=rhott_samples,
                                      grad_outputs=torch.ones_like(rho_estimation).to(td['device']).requires_grad_(True),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
        
        out[:, i] = phi_grad_tt[:, 0] + rho_grad_xx[:, 0]

    out += env.kfp_term(tt_samples, rho_estimation)

    return out
    


# Вычисление loss_t без усреднения по bach_size (т.е. все векторы, которые надо потом просуммировать по B и разделить на B
def get_hjb_loss(td, tt_samples, rhott_samples, groups, batch_size, ones_of_size_phi_out, grad_outputs_vec):
    """
    Compute the HJB error.
    """
    env = td['env']
    # Integral for the Hamilton-Jacobi part
    if env.nu > 0:  # Repeate to parallelize computing the Laplacian/trace for each sample of the batch.
        rhott_samples = rhott_samples.repeat(repeats=(env.dim, 1))
        groups = groups.repeat(repeats=(env.dim, 1))
        tt_samples = tt_samples.repeat(repeats=(env.dim, 1))
        # Например, после repeat(3, 1) при dim=3:
        #
        # [[x1, y1, z1],  # Оригинал 1
        #  [x2, y2, z2],  # Оригинал 2
        #  [x1, y1, z1],  # Копия 1
        #  [x2, y2, z2],  # Копия 2
        #  [x1, y1, z1],  # Копия 3
        #  [x2, y2, z2]]  # Копия 4

    # Включаем вычисление градиентов для tt_samples
    tt_samples.requires_grad_(True)  # WARNING: Keep this after generator evaluation, or else you chain rule generator's time variable

    # делаем оценку функции phi(t, rho(t))
    phi_out = td['discriminator'](tt_samples, rhott_samples[:, :td['env'].dim], groups, td['generator'])

    # Вычисляем производную phi по времени t: ∂(phi_out)/∂(tt_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out,
                                      inputs=tt_samples,
                                      grad_outputs=torch.repeat_interleave(ones_of_size_phi_out, td['env'].dim, dim=1),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]

    # Вычисляем производную phi по x: ∂(phi_out)/∂(rhott_samples)
    phi_grad_xx = torch.autograd.grad(outputs=phi_out,
                                      inputs=rhott_samples,
                                      grad_outputs=torch.repeat_interleave(ones_of_size_phi_out, td['env'].dim, dim=1),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]

    # Если ν > 0 (коэффициент диффузии), то надо вычислить оператор Лапласа (сумму вторых производных)
    if env.nu > 0:
        phi_trace_xx = env.get_trace(phi_grad_xx, rhott_samples, batch_size, env.dim, grad_outputs_vec)
    else:
        phi_trace_xx = torch.zeros(phi_grad_tt.size()).to(td['device'])

    # Гамильтониан
    ham = env.ham(td['generator'], tt_samples, rhott_samples, phi_grad_xx, phi_out)

    out = (phi_grad_tt + env.nu * phi_trace_xx + ham) * td['TT'][0].item()
    #print(out.shape)

    # Compute some info
    info = {'phi_trace_xx': phi_trace_xx.mean(dim=0).item() * td['TT'][0].item()}

    return out, info

# Вычисление loss_0
def get_disc_00_loss(td, discriminator):
    """
    Integral of phi_0 * rho_0.
    """

    # генерируем batch_size семплов rho0
    quantiles, rho0_samples, groups = td['env'].sample_rho0(td['batch_size'])
    rho0_samples = rho0_samples.to(td['device'])
    groups = groups.to(td['device'])

    # вычисляем phi в нулевой момент времени zero: нулевой вектор размерности (batch_size, 1)
    # и далее устредняем по размерности batch_size, чтобы получить loss_0
    disc_00_loss = discriminator(td['zero'], rho0_samples, groups, td['generator']).mean(dim=0)

    return disc_00_loss

# Вычисление f(xb,tb) - вынуждающие условия и препятствия
def get_FF_loss(td, tt_samples, rhott_samples, disc_or_gen):
    """
    Compute the forcing terms (interaction terms and obstacles).
    """
    FF_total_tensor = td['env'].FF_func(td, tt_samples, rhott_samples, disc_or_gen)

    return FF_total_tensor


def do_grad_clip(discriminator, generator, clip_value, disc_or_gen):
    """
    Clips the gradient of the discriminator and/or generator.
    """
    if disc_or_gen == DISC_STRING:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
    elif disc_or_gen == GEN_STRING:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')

def optimizer_step(disc_optimizer, gen_optimizer, disc_or_gen):
    """
    Take a step of the discriminator or generator optimizers.
    """
    if disc_or_gen == DISC_STRING:
        disc_optimizer.step()
    elif disc_or_gen == GEN_STRING:
        gen_optimizer.step()
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')