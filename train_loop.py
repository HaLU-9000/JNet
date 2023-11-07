import sys
sys.path.append('/home/haruhiko/Documents/JNet')
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from dataset import Vibrate, Mask


def divide(x, partial):
    if partial is not None:
        x = x[:, :, partial[0]:partial[1]]
    return x

def calc_loss(pred, label, loss_function, partial):
    return loss_function(divide(pred , partial), 
                         divide(label, partial),)

def branch_calc_loss(out, rec, image, label, loss_function,
                     partial, reconstruct,):
    if reconstruct:
        loss = calc_loss(rec, image, loss_function, partial)
    else:
        loss = calc_loss(out, label, loss_function, partial)
    return loss

vibrate = Vibrate()

def train_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader,
               device, path, savefig_path, model_name, ewc, params, train_dataset_params, val_dataset_params, partial=None,
               scheduler=None, es_patience=10,
               reconstruct=False,
               is_instantblur=False, is_vibrate=False,
               loss_weight=1, qloss_weight = 1/100, ploss_weight = 1/100, tau_init=1, tau_last=0.1, tau_sche=0.9999):
    earlystopping = EarlyStopping(name        = model_name ,
                                  path        = path       ,
                                  patience    = es_patience,
                                  window_size = 10         ,
                                  metric      = "median"   ,
                                  verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss", "validatation loss"] )
    loss_list, midloss_list, vloss_list, vmidloss_list = [], [], [], []
    vibrate = Vibrate()
    mask = Mask()
    #tau = tau_init
    #model.tau = tau
    for epoch in range(1, n_epochs + 1):
        loss_sum, midloss_sum, vloss_sum, vqloss_sum, vmidloss_sum, \
        vparam_loss_sum = 0., 0., 0., 0., 0., 0.
        model.train()
        for train_data in train_loader:
            if is_instantblur:
                label = train_data[1].to(device = device)
                with torch.no_grad():
                    image = model.image.emission.sample(label, params)
                    image = model.image.blur(image)
                    image = model.image.noise(image)
                    image = model.image.preprocess(image)
                image = mask.apply_mask(train_dataset_params["mask"]      ,
                                        image                             ,
                                        train_dataset_params["mask_size"] ,
                                        train_dataset_params["mask_num"]  ,)
            else:
                image    = train_data[0].to(device = device)
                label    = train_data[1].to(device = device)
            if is_vibrate:
                vimage = vibrate(image)
            else:
                vimage = image
            
            outdict = model(vimage)
            out   = outdict["enhanced_image"]
            rec   = outdict["reconstruction"]
            qloss = outdict["quantized_loss"]
            ploss = outdict["psf_loss"]
            loss  = branch_calc_loss(out, rec, image, label,
                                     loss_fn, partial, reconstruct)
            loss *= loss_weight
            if ewc is not None:
                loss += ewc.calc_ewc_loss(100000)
            if qloss is not None:
                loss += qloss * qloss_weight
            if ploss is not None:
                loss += ploss * ploss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
        #tau = model.tau
        model.eval()
        #model.tau = tau_last
        with torch.no_grad():
            for val_data in val_loader:
                if is_instantblur:
                    label = val_data[1].to(device = device)
                    image = model.image.emission.sample(label, params)
                    image = model.image.blur(image)
                    image = model.image.noise(image)
                    image = model.image.preprocess(image)
                else:
                    image    = val_data[0].to(device = device)
                    label    = val_data[1].to(device = device)
                
                #image = image + image * torch.randn_like(image)
                #image = torch.clamp(image, torch.tensor(0., device=device), torch.tensor(1., device=device))
                if is_vibrate:
                    vimage = vibrate(image)
                else:
                    vimage = image
                outdict = model(vimage)
                out   = outdict["enhanced_image"]
                rec   = outdict["reconstruction"]
                qloss = outdict["quantized_loss"]
                ploss = outdict["psf_loss"]
                vloss = branch_calc_loss(out, rec, image, label,
                                                    loss_fn, partial,
                                                    reconstruct)
                vloss_sum += vloss.detach().item() * loss_weight
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss_sum += ploss
#        model.tau = max(tau_last, tau * tau_sche)

        num  = len(train_loader)
        vnum = len(val_loader)
        
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        writer.add_scalar('val param loss', vparam_loss_sum / vnum, epoch)
        writer.add_scalar('val vq loss', vqloss_sum / num, epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = train_curve.append(row, ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv", index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {vloss_list[-1]}')
            #torch.save(model.state_dict(), f'{path}/{model_name}_e{epoch}.pt')
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        earlystopping((vloss_sum / vnum), model, condition = True)#tau == tau_last)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)


class ElasticWeightConsolidation():
    """
    modified from https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    """
    def __init__(self, model, prev_dataloader, loss_fn,
                 init_num_batch, is_vibrate, device, skip_register=True):
        self.model = model
        self.device = device
        self.is_vibrate = is_vibrate
        self.loss_fn = loss_fn
        self.prev_dataloader = prev_dataloader
        num_fisher = 0
        num_params = len([name for name, _ in self.model.named_parameters()])
        for name, _ in self.model.named_buffers():
            if '_estimated_fisher' in name:
                num_fisher += 1
        if num_fisher != 0 and skip_register:
            print("(ewc) ewc params found. Registeration is skipped...")
        else:
            print("(ewc) registering ewc params...")
            self.register_ewc_params(prev_dataloader,
                                     num_batch=init_num_batch)

    def _update_mean_params(self):
        """
        save parameters of previous task in buffer.
        """
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean',
                                       param.data.clone())

    def _update_fisher_params(self, dataloader, num_batch=100):
        """
        calculate diagonal components of fisher information matrix on the task
        and save them in buffer.
        """
        grad_log_likelihood_data = []
        for i, (image, label) in enumerate(dataloader):
            #self.optimizer.zero_grad()
            image = image.to(self.device)
            label = label.to(self.device)
            if i > num_batch:
                break
            if self.is_vibrate:
                vimage = vibrate(image)
            else:
                vimage = image
            outdict = self.model(vimage)
            out   = outdict["enhanced_image"]
            label = label.to(self.device)
            log_likelihood = torch.log(self.loss_fn(out, label))
            grad_log_likelihood = torch.autograd.grad(log_likelihood,
                                                      self.model.parameters(),
                                                      retain_graph = False,
                                                      allow_unused=True)
            grad_log_likelihood = list(grad_log_likelihood)
            #print(len(grad_log_likelihood))
            for n, param in enumerate(grad_log_likelihood):
                if param is None:
                        param_data_clone = None
                else:
                    param_data_clone = param.data.clone() / len(dataloader)
                if len(grad_log_likelihood_data) != len(grad_log_likelihood):
                    #print(len(grad_log_likelihood_data))
                    grad_log_likelihood_data.append(param_data_clone)
                elif param is not None:
                    #print(len(grad_log_likelihood_data))
                    grad_log_likelihood_data[n] += param_data_clone
        grad_log_likelihood_data = tuple(grad_log_likelihood_data)
        _buff_param_names = [param[0].replace('.', '__')
                             for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names,
                                           grad_log_likelihood_data):
            if param is None:
                param_data_clone = None
            else:
                param_data_clone = param.data.clone() ** 2 / len(dataloader)
            self.model.register_buffer(_buff_param_name+"_estimated_fisher",
                                       param_data_clone)

    def register_ewc_params(self, dataloader, num_batch):
        """
        update mean and fisher for initialization
        """
        self._update_mean_params()
        self._update_fisher_params(dataloader, num_batch)

    def calc_ewc_loss(self, lambda_):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            mean = getattr(self.model, f'{_buff_param_name}_estimated_mean')
            fisher = getattr(self.model, f'{_buff_param_name}_estimated_fisher')
            if fisher is not None and mean is not None and param is not None:
                losses.append((fisher * (param - mean) ** 2).sum())
        return (lambda_ / 2) * sum(losses)