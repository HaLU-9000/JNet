from subprocess import check_output
import sys
sys.path.append('/home/haruhiko/Documents/JNet')
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import matplotlib.pyplot as plt

def train_loop(n_epochs,
               optimizer,
               model,
               discriminator,
               loss_fn,
               train_loader,
               val_loader,
               device,
               path,
               savefig_path,
               model_name,
               partial=None,
               scheduler=None,
               es_patience=10, 
               tau_init=1.0,
               tau_lb=0.1,
               tau_sche=0.9999,
               reconstruct=False,
               dloss_fn=None):

    earlystopping = EarlyStopping(name     = model_name ,
                                  path     = path       ,
                                  patience = es_patience,
                                  verbose  = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    loss_list    = []
    dloss_list = []
    val_list     = []
    valmid_list  = []
    tau = tau_init
    for epoch in range(1, n_epochs + 1):
        writer.add_scalar('tau', 
                    tau,
                    epoch,)
        loss_sum     = 0.0
        dloss_sum  = 0.0
        valloss_sum  = 0.0
        valmid_sum   = 0.0
        model.train()
        model.set_hard(True) ######
        for image, label in train_loader:
            model.set_tau(tau)
            image    = image.to(device = device)
            label    = label.to(device = device)
            out, rec = model(image)
            if reconstruct:
                if partial is not None:
                    rloss = loss_fn(rec[  :, :, partial[0]:partial[1]],
                                   image[:, :, partial[0]:partial[1]])
                else:
                    rloss = loss_fn(rec, image)
            else:
                if partial is not None:
                    rloss = loss_fn(out[  :, :, partial[0]:partial[1]],
                                   label[:, :, partial[0]:partial[1]])
                else:
                    rloss = loss_fn(out, label)

            if partial is not None:
                dloss = discriminator(out[  :, :, partial[0]:partial[1]],
                                        label[:, :, partial[0]:partial[1]])
            else:
                dloss = discriminator(out, label)

            dloss_sum += dloss.detach()
            loss_sum += rloss.detach()
            tau = max(tau_lb, tau * tau_sche)
            optimizer.zero_grad()
            rloss.backward(retain_graph=True) ###
            optimizer.step()
        loss_list.append(loss_sum.item() / len(train_loader))
        writer.add_scalar('train loss', 
                          loss_sum.item() / len(train_loader),
                          epoch,)
        dloss_list.append(dloss_sum.item() / len(train_loader))
        writer.add_scalar('train middle loss', 
                        dloss_sum.item() / len(train_loader),
                        epoch,)
                          
        model.eval()
        model.set_hard(True)
        model.set_tau(tau_lb)
        with torch.no_grad():
            for image, label in val_loader:
                image       = image.to(device = device)
                label       = label.to(device = device)
                out, rec    = model(image)
                if reconstruct:
                    if partial is not None:
                        rval_loss = loss_fn(rec[  :, :, partial[0]:partial[1]],
                                            image[:, :, partial[0]:partial[1]])
                    else:
                        rval_loss = loss_fn(rec, image)

                else:
                    if partial is not None:
                        rval_loss = loss_fn(out[  :, :, partial[0]:partial[1]],
                                           label[:, :, partial[0]:partial[1]])
                    else:
                        rval_loss = loss_fn(out, label)
                
                dval_loss = discriminator(out[  :, :, partial[0]:partial[1]],
                                          label[:, :, partial[0]:partial[1]]).detach()
                dval_loss = discriminator(out, label).detach()
                dval_loss_sum  += dval_loss
                rval_loss_sum += rval_loss.detach()
            val_list.append(valloss_sum.item() / len(val_loader))
            writer.add_scalar('val loss',
                              valloss_sum.item() / len(val_loader),
                              epoch)
            valmid_list.append(valmid_sum.item() / len(val_loader))
            writer.add_scalar('val middle loss',
                                valmid_sum / len(val_loader),
                                epoch)
            writer.add_scalar('mu_z',
                            model.state_dict()['blur.mu_z'].item(),
                            epoch)
            writer.add_scalar('sig_z',
                            model.state_dict()['blur.sig_z'].item(),
                            epoch)
            writer.add_scalar('bet_xy',
                            model.state_dict()['blur.bet_xy'].item(),
                            epoch)
            writer.add_scalar('bet_z',
                            model.state_dict()['blur.bet_z'].item(),
                            epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {val_list[-1]}')
            torch.save(model.state_dict(), f'{path}/{model_name}_e{epoch}.pt')
        if scheduler is not None:
            scheduler.step(valloss_sum.item() / len(val_loader))
        earlystopping((valloss_sum.item() / len(val_loader)), model, tau == tau_lb)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list, label='train loss')
    plt.plot(val_list , label='validation loss')
    plt.plot(dloss_list, label='train disc loss')
    plt.plot(valmid_list , label='validation disc loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)