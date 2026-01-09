import torch
from utils.utils import DISC_STRING, GEN_STRING
from utils.helpers_train_once import set_requires_grad, set_zero_grad, get_generator_samples, get_hjb_loss, \
                                     get_FF_loss, get_disc_00_loss, optimizer_step, get_kfp_loss


# ====================================
#           The main trainer
# ====================================
def train_once(td, disc_or_gen):
    """
    Trains the discriminator and generator.
    """
    error_msg = f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}'
    assert (disc_or_gen == DISC_STRING or disc_or_gen == GEN_STRING), error_msg

    # Activate computing computational graph of discriminator/generator
    set_requires_grad(td['discriminator'], td['generator'], disc_or_gen)

    # Zero the gradients
    set_zero_grad(td['disc_optimizer'], td['gen_optimizer'], disc_or_gen)

    # Integral for the Hamilton-Jacobi part
    rho00, tt_samples, pdf_params, rhott_samples, groups = get_generator_samples(td, disc_or_gen)

    # Вычисляем loss_t ...
    hjb_loss_tensor, hjb_loss_info = get_hjb_loss(td, tt_samples, rhott_samples, groups, td['batch_size'],
                                                  td['ones_of_size_phi_out'], td['grad_outputs_vec'])
    hjb_loss_tensor = hjb_loss_tensor[:td['batch_size']]

    # ... loss_t = hjb_loss_tensor.mean(dim=0)
    hjb_loss = (hjb_loss_tensor).mean(dim=0)

    # Interaction terms
    # Вычисляем f(xb,tb)
    FF_total_tensor = get_FF_loss(td, tt_samples, rhott_samples, disc_or_gen) #* groups

    rho_estimation = td['env']._estimate_rho(td['generator'], tt_samples, rhott_samples)
    rho_grad_tt = torch.autograd.grad(outputs=rho_estimation,
                                      inputs=tt_samples,
                                      grad_outputs=torch.ones_like(rho_estimation).to(td['device']).requires_grad_(True),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
    
    phi_out = td['discriminator'](tt_samples, rhott_samples, groups, td['generator'])
    ones_of_size_phi_out = torch.ones_like(phi_out).to(phi_out.device).requires_grad_(True)
    
    phi_grad_xx = torch.autograd.grad(outputs=phi_out,
                                      inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
    
    rho_grad_xx = torch.autograd.grad(outputs=rho_estimation * -phi_grad_xx,
                                      inputs=rhott_samples,
                                      grad_outputs=torch.ones_like(rho_estimation).to(td['device']).requires_grad_(True),
                                      create_graph=True,
                                      retain_graph=True,
                                      only_inputs=True)[0]
    
    ham = td['env'].ham(td['generator'], tt_samples, rhott_samples, phi_grad_xx, phi_out)

    # Finish computing the total loss
    if disc_or_gen == DISC_STRING:
        # Integral of phi_0 * rho_0
        disc_00_loss = get_disc_00_loss(td, td['discriminator']) #* groups
        # L2 Hamiltonian residual
        # Вычисляем loss_HJB = mean(norm(loss_t + f))
        disc_hjb_error = torch.norm(hjb_loss_tensor - FF_total_tensor, dim=1).mean(dim=0) / td['TT'][0].item()
        # Total loss

        total_loss = (-1) * (disc_00_loss + hjb_loss) + td['lam_hjb_error'] * disc_hjb_error
        total_loss = td['lam_hjb_error'] * disc_hjb_error
    else:  # disc_or_gen == GEN_STRING:
        # Total loss
        FF_total_loss = FF_total_tensor.mean(dim=0)

        kfp_loss = get_kfp_loss(td, tt_samples, rhott_samples, rho_estimation, phi_grad_xx, torch.ones_like(rho_estimation).to(td['device']).requires_grad_(True))
        # hjb_loss
        total_loss = torch.norm(rho_grad_tt + ham, dim=1).mean(dim=0)
        total_loss = torch.norm(kfp_loss, dim=1).mean(0)
        '''
        print('T:', tt_samples)
        print('PARAMS:', pdf_params)
        print('-' * 20)'''
        #total_loss = torch.norm(rho_grad_tt + rho_grad_xx + ham, dim=1).mean(dim=0) - FF_total_loss # + or - ???
        #total_loss = hjb_loss + FF_total_loss + kfp_loss
    # Backprop and optimize
    kfp_loss = get_kfp_loss(td, tt_samples, rhott_samples, rho_estimation, phi_grad_xx, torch.ones_like(rho_estimation).to(td['device']).requires_grad_(True))
    total_loss.sum().backward()
    optimizer_step(td['disc_optimizer'], td['gen_optimizer'], disc_or_gen)

    

    # Get info about the training
    with torch.no_grad():
        hjb_error = torch.norm(hjb_loss_tensor - FF_total_tensor, dim=1).mean(dim=0) / td['TT'][0].item()
        training_info = {}

        prefix = 'disc' if disc_or_gen == DISC_STRING else 'gen'
        if disc_or_gen == DISC_STRING:
            training_info[f'{prefix}_t0_loss'] = disc_00_loss.sum().item()
            training_info[f'{prefix}_t1_loss'] = hjb_loss.sum().item()
            training_info[f'{prefix}_hjb_loss'] = hjb_error.sum().item()
            training_info[f'{prefix}_total_loss'] = total_loss.sum().item()
            training_info[f'{prefix}_rho_grad_t'] = rho_grad_tt.sum().item()
            training_info[f'{prefix}_rho_grad_x'] = rho_grad_xx.sum().item()
            gr = ['S', 'E', 'I', 'R', 'H', 'C', 'D']
            for i, let in enumerate(gr):
                #training_info[f'{prefix}_kfp_equation_{let}'] = (rho_grad_tt + rho_grad_xx + ham)[:, i].sum().item()
                training_info[f'{prefix}_kfp_equation_{let}'] = kfp_loss[:, i].sum().item()

        else:
            training_info[f'{prefix}_t1_loss'] = hjb_loss.sum().item()
            gr = ['S', 'E', 'I', 'R', 'H', 'C', 'D']
            for i, let in enumerate(gr):
                training_info[f'{prefix}_ff_loss_{let}'] = FF_total_loss[i].item()
            training_info[f'{prefix}_total_loss'] = total_loss.sum().item()

    return training_info
