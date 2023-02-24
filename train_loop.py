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

def calc_loss(pred, label, predparams, params,
              loss_function, paramloss_function, partial):
    loss  = loss_function(divide(pred , partial), 
                          divide(label, partial),)
    loss += paramloss_function(predparams, params)
    return loss

def train_loop(n_epochs, optimizer, model, loss_fn, ploss_fn, train_loader, val_loader,
               device, path, savefig_path, model_name, partial=None,
               scheduler=None, es_patience=10,
               tau_init=1.0, tau_lb=0.1, tau_sche=0.9999,):
    earlystopping = EarlyStopping(name     = model_name ,
                                  path     = path       ,
                                  patience = es_patience,
                                  verbose  = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    loss_list, vloss_list = [], []
    tau = tau_init
    for epoch in range(1, n_epochs + 1):
        loss_sum, midloss_sum, vloss_sum, vmidloss_sum = 0., 0., 0., 0.
        model.train()
        for image, label, p in train_loader:
            model.set_tau(tau)
            image    = image.to(device = device)
            label    = label.to(device = device)
            out, pp  = model(image)
            loss,    = calc_loss(out, label, pp, p,
                                 loss_fn, ploss_fn, partial,)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            tau = max(tau_lb, tau * tau_sche)
            loss_sum += loss.detach().item()
                
        model.eval()
        model.set_tau(tau_lb)
        with torch.no_grad():
            for image, label in val_loader:
                image       = image.to(device = device)
                label       = label.to(device = device)
                out, rec    = model(image)
                vloss, = calc_loss(out, rec, image, label,
                                   loss_fn, ploss_fn, partial,)
                vloss_sum += vloss.detach().item()
        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('tau', tau, epoch)
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {vloss_list[-1]}')
        if scheduler is not None:
            scheduler.step(vloss_sum / vnum)
        earlystopping((vloss_sum / vnum), model, tau == tau_lb)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list, label='train loss')
    plt.plot(vloss_list , label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)