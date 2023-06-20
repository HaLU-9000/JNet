import sys
sys.path.append('/home/haruhiko/Documents/JNet')
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import matplotlib.pyplot as plt

def divide(x, partial):
    if partial is not None:
        x = x[:, :, partial[0]:partial[1]]
    return x

def calc_loss(pred, label, loss_function, partial):
    return loss_function(divide(pred , partial), 
                         divide(label, partial),)

def branch_calc_loss(out, rec, image, label, loss_function, midloss_function,
                     partial, reconstruct, check_middle):
    if reconstruct:
        loss = calc_loss(rec, image, loss_function, partial)
        if check_middle:
            midloss = calc_loss(out, label, midloss_function, partial)
        else:
            midloss = torch.tensor(0)
    else:
        loss = calc_loss(out, label, loss_function, partial)
        midloss = torch.tensor(0)
    return loss, midloss

def train_loop(n_epochs, optimizer, model, loss_fn, param_loss_fn, train_loader, val_loader,
               device, path, savefig_path, model_name, param_normalize, partial=None,
               scheduler=None, es_patience=10,
               tau_init=1.0, tau_lb=0.1, tau_sche=0.9999,
               reconstruct=False, check_middle=False, midloss_fn=None, 
               is_randomblur=False):
    earlystopping = EarlyStopping(name     = model_name ,
                                  path     = path       ,
                                  patience = es_patience,
                                  verbose  = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    loss_list, midloss_list, vloss_list, vmidloss_list = [], [], [], []
    for epoch in range(1, n_epochs + 1):
        loss_sum, midloss_sum, vloss_sum, vmidloss_sum, vparam_loss_sum = 0., 0., 0., 0., 0.
        model.train()
        for train_data in train_loader:
            if is_randomblur:
                label  = train_data[0].to(device = device)
                params = train_data[1]
                image  = model.image.sample_from_params(label, params).float()
                target_params = torch.Tensor(
                    list(param_normalize(params).values())[:-2]).to(device)

                model.set_upsample_rate(params["scale"])
            else:
                image    = train_data[0].to(device = device)
                label    = train_data[1].to(device = device)
            outdict = model(image)
            out   = outdict["enhanced_image"]
            rec   = outdict["reconstruction"]
            qloss = outdict["quantized_loss"]
            est_params = outdict["blur_parameter"]
            loss, midloss = branch_calc_loss(out, rec, image, label,
                                             loss_fn, midloss_fn, partial,
                                             reconstruct, check_middle)
            paramloss = param_loss_fn(est_params, target_params)
            #print("est", est_params, "target", target_params, "paramloss", paramloss, "loss", loss)
            if qloss is not None:
                loss += qloss
            loss += paramloss / 100  ## which means paramloss = 0
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_sum += loss.detach().item()
            if check_middle:
                midloss_sum += midloss.detach().item()
                
        model.eval()
        with torch.no_grad():
            for image, label in val_loader:
                if is_randomblur:
                    label  = train_data[0].to(device = device)
                    params = train_data[1]
                    image  = model.image.sample_from_params(label, params).float()
                    target_params = torch.Tensor(
                    list(param_normalize(params).values())[:-2]).to(device)
                    model.set_upsample_rate(params["scale"])
                else:
                    image    = train_data[0].to(device = device)
                    label    = train_data[1].to(device = device)               
                outdict = model(image)
                out   = outdict["enhanced_image"]
                rec   = outdict["reconstruction"]
                qloss = outdict["quantized_loss"]
                est_params = outdict["blur_parameter"]
                vloss, vmid_loss = branch_calc_loss(out, rec, image, label,
                                                    loss_fn,midloss_fn,partial,
                                                    reconstruct, check_middle)
                vloss_sum += vloss.detach().item()
                if qloss is not None:
                    vloss_sum += qloss.detach().item()
                if check_middle:
                    vmidloss_sum += vmid_loss.detach().item()
                vparam_loss_sum += param_loss_fn(target_params, est_params).detach().item()
        num  = len(train_loader)
        vnum = len(val_loader)
        ez0, bet_z, bet_xy, alpha = [i for i in model.parameters()][-4:]
        loss_list.append(loss_sum / num)
        midloss_list.append(midloss_sum / num) if check_middle else 0
        vloss_list.append(vloss_sum / vnum)
        vmidloss_list.append(vmidloss_sum / vnum) if check_middle else 0
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('train middle loss', midloss_sum / num, epoch) if check_middle else 0
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        writer.add_scalar('val param loss', vparam_loss_sum / vnum, epoch)
        writer.add_scalar('val middle loss', vmidloss_sum / vnum, epoch) if check_middle else 0
        writer.add_scalar('bet_xy', bet_xy.item(), epoch)
        writer.add_scalar('bet_z' , bet_z.item() , epoch)
        writer.add_scalar('alpha' , alpha.item() , epoch)
        writer.add_scalar('ez0'   , ez0.item()   , epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {vloss_list[-1]}')
            #torch.save(model.state_dict(), f'{path}/{model_name}_e{epoch}.pt')
        if scheduler is not None:
            scheduler.step(vloss_sum / vnum)
        earlystopping((vloss_sum / vnum), model, condition=True)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    if check_middle:
        plt.plot(midloss_list, label='train loss (middle)')
        plt.plot(vmidloss_list , label='validation loss (middle)')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)