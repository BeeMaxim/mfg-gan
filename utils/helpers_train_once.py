import torch
from utils.utils import uniform_time_sampler, DISC_STRING, GEN_STRING


# ================
# Helper functions
# ================

# Включаем/отключаем градиенты генератора/дискриминатора
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
    rho00 = td['env'].sample_rho0(td['batch_size']).to(td['device'])
    # генерируем вектор времени из равномерного распределения от 0 до T размером
    # Если td['TT'] уже тензор на GPU, .item() автоматически перенесет значение на CPU
    tt_samples = (td['TT'][0].item() * uniform_time_sampler(td['batch_size'])).to(td['device'])

    if disc_or_gen == DISC_STRING:
        # .detach(): Отсоединяет тензор от вычислительного графа, Чтобы градиенты не распространялись в генератор
        #            при обучении дискриминатора. Т.е. замораживаем параметры генератора
        # .requires_grad_(True): Включает вычисление градиентов для самого выхода (rhott_samples),
        #                        чтобы дискриминатор мог вычислять градиенты по этим данным
        rhott_samples = td['generator'](tt_samples, rho00).detach().requires_grad_(True)
    elif disc_or_gen == GEN_STRING:
        rhott_samples = td['generator'](tt_samples, rho00)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')

    return rho00, tt_samples, rhott_samples

# Вычисление loss_t без усреднения по bach_size (т.е. все векторы, которые надо потом просуммировать по B и разделить на B
def get_hjb_loss(td, tt_samples, rhott_samples, batch_size, ones_of_size_phi_out, grad_outputs_vec):
    """
    Compute the HJB error.
    """
    env = td['env']
    # Integral for the Hamilton-Jacobi part
    if env.nu > 0:  # Repeate to parallelize computing the Laplacian/trace for each sample of the batch.
        rhott_samples = rhott_samples.repeat(repeats=(env.dim, 1))
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
    phi_out = td['discriminator'](tt_samples, rhott_samples)

    # Вычисляем производную phi по времени t: ∂(phi_out)/∂(tt_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out,
                                      inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out, # Вектор весов для градиентов (обычно тензор
                                                                         # единиц). Позволяет взвешивать вклад каждого
                                                                         # примера. Единицы означают "стандартный"
                                                                         # градиент
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]

    # Вычисляем производную phi по x: ∂(phi_out)/∂(rhott_samples)
    phi_grad_xx = torch.autograd.grad(outputs=phi_out,
                                      inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]

    # Если ν > 0 (коэффициент диффузии), то надо вычислить оператор Лапласа (сумму вторых производных)
    if env.nu > 0:
        phi_trace_xx = env.get_trace(phi_grad_xx, rhott_samples, batch_size, env.dim, grad_outputs_vec)
    else:
        phi_trace_xx = torch.zeros(phi_grad_tt.size()).to(td['device'])

    # Гамильтониан
    ham = env.ham(tt_samples, rhott_samples, (-1) * phi_grad_xx)

    out = (phi_grad_tt + env.nu * phi_trace_xx - ham) * td['TT'][0].item()

    # Compute some info
    info = {'phi_trace_xx': phi_trace_xx.mean(dim=0).item() * td['TT'][0].item()}

    return out, info

# Вычисление loss_0
def get_disc_00_loss(td, discriminator):
    """
    Integral of phi_0 * rho_0.
    """

    # генерируем batch_size семплов rho0
    rho0_samples = td['env'].sample_rho0(td['batch_size']).to(td['device'])

    # вычисляем phi в нулевой момент времени zero: нулевой вектор размерности (batch_size, 1)
    # и далее устредняем по размерности batch_size, чтобы получить loss_0
    disc_00_loss = discriminator(td['zero'], rho0_samples).mean(dim=0)

    return disc_00_loss

# Вычисление f(xb,tb) - вынуждающие условия и препятствия
def get_FF_loss(td, tt_samples, rhott_samples, disc_or_gen):
    """
    Compute the forcing terms (interaction terms and obstacles).
    """
    FF_total_tensor, FF_info = td['env'].FF_func(td, tt_samples, rhott_samples, disc_or_gen)

    return FF_total_tensor, FF_info


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