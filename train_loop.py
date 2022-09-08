import sys
sys.path.append('/home/haruhiko/Documents/JNet')
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import matplotlib.pyplot as plt

def train_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, device,
               path, savefig_path, model_name, partial=None, scheduler=None, es_patience=10, 
               tau_init=1.0, tau_lb=0.1, tau_sche=0.9999):
    earlystopping = EarlyStopping(name     = model_name ,
                                  path     = path       ,
                                  patience = es_patience,
                                  verbose  = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    loss_list = []
    val_list  = []
    tau = tau_init
    for epoch in range(1, n_epochs + 1):
        writer.add_scalar('tau', 
                    tau,
                    epoch,)
        loss_sum     = 0.0
        valloss_sum  = 0.0
        model.train()
        for image, label in train_loader:
            model.set_tau(tau)
            image    = image.to(device = device)
            label    = label.to(device = device)
            out, rec = model(image)
            if partial is not None:
                loss = loss_fn(out[:,   :, partial[0]:partial[1]],
                               label[:, :, partial[0]:partial[1]])
            else:
                loss = loss_fn(out, label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True) ###
            optimizer.step()
            loss_sum += loss.detach()
        tau = max(tau_lb, tau * tau_sche) # JNet_96_x1~
        loss_list.append(loss_sum.item() / len(train_loader))
        writer.add_scalar('train loss', 
                          loss_sum.item() / len(train_loader),
                          epoch,)

        model.eval()
        model.set_tau(tau_lb)
        with torch.no_grad():
            for image, label in val_loader:
                image       = image.to(device = device)
                label       = label.to(device = device)
                out, rec    = model(image) #, 0
                if partial is not None:
                    val_loss = loss_fn(out[:,   :, partial[0]:partial[1]],
                                       label[:, :, partial[0]:partial[1]])
                else:
                    val_loss = loss_fn(out, label)
                valloss_sum += val_loss.detach()
            val_list.append(valloss_sum.item() / len(val_loader))
            writer.add_scalar('val loss',
                              valloss_sum.item() / len(val_loader),
                              epoch)
        if epoch == 1 or epoch % 100 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {val_list[-1]}')
        if scheduler is not None and tau == tau_lb:
            scheduler.step(valloss_sum.item() / len(val_loader))
        earlystopping((valloss_sum.item() / len(val_loader)), model, tau == tau_lb)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list, label='train loss')
    plt.plot(val_list , label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)