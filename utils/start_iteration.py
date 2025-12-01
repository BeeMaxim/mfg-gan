from utils.utils import act_funcs, DISC_STRING, GEN_STRING
from neural_net_defs import *
from utils.train_once import train_once
from utils.utils import Plotter, Logger
from torchviz import make_dot
from PIL import Image
from tqdm import tqdm
import os
from utils.init_utils import setup_saving_and_logging
from logger.cometml import CometMLWriter



def start_train(a):
    """
    a: A dictionary containing the training arguments
    """
    # env = SEIRHCD_Env(device=device)
    env = a['env']
    the_logger = Logger(a)
    the_plotter = Plotter(a, the_logger)
    writer = CometMLWriter(
        project_name='mfg_gan',
        run_name='testing',
        mode='online',
        loss_names=['disc_t0_loss', 'disc_t1_loss', 'disc_hjb_loss', 'disc_total_loss'],
        log_checkpoints=False,
        id_length=32
    )

    # =============================================
    #           Precompute some variables
    # =============================================
    # Precompute ones tensor of size phi out for gradient computation
    ones_of_size_phi_out = torch.ones(a['batch_size'] * env.dim, 1).to(a['device']) if env.nu > 0 \
                           else torch.ones(a['batch_size'], 1).to(a['device'])

    # ones_of_size_phi_out = torch.ones(batch_size * 2, 1)
    # [[1.],
    #  [1.],
    #  [1.],
    #  [1.],
    #  [1.],
    #  [1.]]

    # Precompute grad outputs vec for laplacian for Hessian computation
    list_1 = []
    for i in range(env.dim):
        vec = torch.zeros(size=(a['batch_size'], env.dim), dtype=torch.float).to(a['device'])
        vec[:, i] = torch.ones(size=(a['batch_size'],)).to(a['device'])
        list_1.append(vec)
    grad_outputs_vec = torch.cat(list_1, dim=0)

    # grad_outputs_vec = (batch_size * env.dim, env.dim) = (50 * 2, 2)
    #
    # [[1., 0.],  # Вектор для i=0 (первый пример)

    #  [1., 0.],  # Вектор для i=0 (второй пример)
    #  [1., 0.],  # Вектор для i=0 (третий пример)
    #  [0., 1.],  # Вектор для i=1 (первый пример)
    #  [0., 1.],  # Вектор для i=1 (второй пример)
    #  [0., 1.]]  # Вектор для i=1 (третий пример)


    # ======================================
    #           Setup the learning
    # ======================================
    # Compute the mean and variance of rho0 (assuming rho0 is a simple Gaussian)

    # FIX???
    '''
    temp_sample = env.sample_rho0(int(1e4)).to(a['device'])
    mu = temp_sample.mean(axis=0)
    std = torch.sqrt(temp_sample.var(axis=0))
    if 0 in std:
        raise ValueError("std of sample_rho0 has a zero!")'''
    
    mu, std = 0, 1

    # Make the networks
    # TODO: в forward(self, t, inp) добавить генератор generator, чтобы посчитать rho(T) = generator(T, rho0)
    discriminator = DiscNet(dim=env.dim, # dim=2
                            ns=a['ns'], # ns = 100 (Network size)
                            act_func=act_funcs[a['act_func_disc']], # act_func_disc = tanh
                            hh=a['hh'], # ResNet step-size = 0.5
                            device=a['device'],
                            # TODO: в psi_func нужно добавить генератор, чтобы посчитать rho(T)
                            psi_func=env.psi_func, # The final-time cost function
                            TT=env.TT).to(a['device']) # TT = 1

    generator = GenNet(dim=env.dim, # dim=2
                       ns=a['ns'], # ns = 100 (Network size)
                       act_func=act_funcs[a['act_func_gen']], # act_func_gen = relu
                       hh=a['hh'], # ResNet step-size = 0.5
                       device=a['device'],
                       mu=mu, # среднее начального распределения rho0
                       std=std, # стандартное отклонение распределения rho0
                       TT=env.TT).to(a['device']) # TT = 1

    disc_optimizer = torch.optim.Adam(discriminator.parameters(),
                                      lr=a['disc_lr'], # 0.0004
                                      weight_decay=a['weight_decay'], # 0.0001
                                      betas=a['betas']) # (0.5, 0.9)
    gen_optimizer = torch.optim.Adam(generator.parameters(),
                                     lr=a['gen_lr'],  # 0.0001
                                     weight_decay=a['weight_decay'], # 0.0001
                                     betas=a['betas']) # (0.5, 0.9)

    # ===================================
    #           Start iteration
    # ===================================
    # Define initial time and final time constants
    zero = torch.tensor([0], dtype=torch.float).expand((a['batch_size'], 1)).to(a['device'])
    TT = torch.tensor([env.TT], dtype=torch.float).expand((a['batch_size'], 1)).to(a['device'])

    # Start the iteration
    for epoch in tqdm(range(a['max_epochs'] + 1)):
        # =============================
        #           Info dump
        # =============================
        if epoch % a['print_rate'] == 0: # print_rate = 1000
            print()
            print('-' * 10)
            print(f'epoch: {epoch}\n')

            if epoch != 0:
                # Saving neural network and saving to csv
                the_logger.save_nets({'epoch': epoch,
                                      'discriminator': discriminator,
                                      'discriminator_optimizer': disc_optimizer,
                                      'generator': generator,
                                      'generator_optimizer': gen_optimizer})
                the_logger.write_training_csv(epoch)

        # ===========================================
        #           Setup training dictionary
        # ===========================================
        train_dict = a.copy()
        train_dict.update({'discriminator': discriminator,
                           'generator': generator,
                           'disc_optimizer': disc_optimizer,
                           'gen_optimizer': gen_optimizer,
                           'ham_func': env.ham,
                           'epoch': epoch,
                           'zero': zero, # нулевой вектор времени (batch_size, 1)
                           'TT': TT, # Конечный вектор TT=1 времени (batch_size, 1)
                           'ones_of_size_phi_out': ones_of_size_phi_out, # вектор (batch_size*2, 1) из единиц
                           'grad_outputs_vec': grad_outputs_vec, # единичная диаганально-блоковая матрица размерности
                                                                 # (batch_size*2, 2) из единиц и нулей в шахматном порядке
                           'the_logger': the_logger})

        # ===========================================
        #           Train phi/discriminator
        # ===========================================
        train_info = train_once(train_dict, DISC_STRING)

        the_logger.log_training(train_info, DISC_STRING)
        if epoch % a['print_rate'] == 0:
            pass
            #the_logger.print_to_console(train_info, DISC_STRING)

        # ======================================
        #           Train rho/generator
        # ======================================
        if epoch % a['gen_every_disc'] == 0:  # How many times to update discriminator per one update of generator.
            train_info.update(train_once(train_dict, GEN_STRING))

        the_logger.log_training(train_info, GEN_STRING)
        if epoch % a['print_rate'] == 0:
            pass
            #the_logger.print_to_console(train_info, GEN_STRING)

        # =======================================
        #           Plot images and etc.
        # =======================================
        if epoch % a['print_rate'] == 0:
            pass
            #path = the_plotter.make_plots(epoch, generator, the_logger)
            #print(path)
            #writer.add_image('image', path)

        if epoch % a['print_rate'] == 0:
            writer.set_step(epoch)
            writer.add_scalar('epoch', epoch)
            for k, v in train_info.items():
                writer.add_scalar(k, v)

    return the_logger
