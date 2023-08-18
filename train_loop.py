import sys
sys.path.append('/home/haruhiko/Documents/JNet')
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import matplotlib.pyplot as plt
from dataset import Vibrate


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

def preprocess():
    pass

def train_loop(n_epochs, optimizer, model, loss_fn, param_loss_fn, train_loader, val_loader,
               device, path, savefig_path, model_name, param_normalize, augment, val_augment, partial=None,
               scheduler=None, es_patience=10,
               reconstruct=False, check_middle=False, midloss_fn=None, 
               is_randomblur=False, is_vibrate=False,
               loss_weight=1, qloss_weight = 1/100, paramloss_weight = 1/10,
               verbose=False):
    earlystopping = EarlyStopping(name     = model_name ,
                                  path     = path       ,
                                  patience = es_patience,
                                  verbose  = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    loss_list, midloss_list, vloss_list, vmidloss_list = [], [], [], []
    vibrate = Vibrate()
    for epoch in range(1, n_epochs + 1):
        loss_sum, midloss_sum, vloss_sum, vqloss_sum, vmidloss_sum, \
        vparam_loss_sum = 0., 0., 0., 0., 0., 0.
        model.train()
        for train_data in train_loader:
            if is_randomblur:
                label  = train_data[0].to(device = device)
                params = train_data[1]
                image  = model.image.sample_from_params(label, params).float()
                b = image.shape[0]
                plist = list(param_normalize(params).values())
                target_params = torch.empty(b, len(params) - 2)
                for i, p in enumerate(plist[:len(params) - 2]):
                    target_params[:, i] = p
                target_params = target_params.to(device)
                model.set_upsample_rate(int(params["scale"][0]))
                image, label = augment.crop(image, label)
                image = augment(image)
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
            loss, midloss = branch_calc_loss(out, rec, image, label,
                                             loss_fn, midloss_fn, partial,
                                             reconstruct, check_middle)
            loss *= loss_weight
            if qloss is not None:
                loss += qloss * qloss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
            if check_middle:
                midloss_sum += midloss.detach().item()
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                if is_randomblur:
                    label  = val_data[0].to(device = device)
                    params = val_data[1]
                    image  = model.image.sample_from_params(label, params).float()
                    target_params = torch.empty(b, len(params) - 2)
                    for i, p in enumerate(plist[:len(params) - 2]):
                        target_params[:, i] = p
                    target_params = target_params.to(device)
                    model.set_upsample_rate(params["scale"][0])
                    image, label = val_augment.crop(image, label)
                    image = val_augment(image)
                else:
                    image    = val_data[0].to(device = device)
                    label    = val_data[1].to(device = device)
                if is_vibrate:
                    vimage = vibrate(image)
                else:
                    vimage = image
                outdict = model(vimage)
                out   = outdict["enhanced_image"]
                rec   = outdict["reconstruction"]
                qloss = outdict["quantized_loss"]
                vloss, vmid_loss = branch_calc_loss(out, rec, image, label,
                                                    loss_fn,midloss_fn,partial,
                                                    reconstruct, check_middle)
                vloss_sum += vloss.detach().item() * loss_weight
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if check_middle:
                    vmidloss_sum += vmid_loss.detach().item()
        num  = len(train_loader)
        vnum = len(val_loader)
        ez0, bet_z, bet_xy  = [i for i in model.parameters()][-3:]
        if verbose:
            print([i for i in model.state_dict()][-3:])
            print(ez0, bet_z, bet_xy)
        loss_list.append(loss_sum / num)
        midloss_list.append(midloss_sum / num) if check_middle else 0
        vloss_list.append(vloss_sum / vnum)
        vmidloss_list.append(vmidloss_sum / vnum) if check_middle else 0
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('train middle loss', midloss_sum / num, epoch) if check_middle else 0
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        writer.add_scalar('val param loss', vparam_loss_sum / vnum, epoch)
        writer.add_scalar('val middle loss', vmidloss_sum / vnum, epoch) if check_middle else 0
        writer.add_scalar('val vq loss', vqloss_sum / num, epoch)
        writer.add_scalar('bet_xy', bet_xy.item(), epoch)
        writer.add_scalar('bet_z' , bet_z.item() , epoch)
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

def train_loop_v2(n_epochs, optimizer, model, loss_fn,
                  train_loader0,train_loader1, train_loader2,
                  val_loader0, val_loader1, val_loader2,
               device, path, savefig_path, model_name, partial=None,
               scheduler=None, es_patience=10,
               reconstruct=False, check_middle=False, midloss_fn=None,
               loss_weight=1, qloss_weight = 1/100, verbose=False):
    earlystopping = EarlyStopping(name     = model_name ,
                                  path     = path       ,
                                  patience = es_patience,
                                  verbose  = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    loss_list, midloss_list, vloss_list, vmidloss_list = [], [], [], []
    for epoch in range(1, n_epochs + 1):
        loss_sum, midloss_sum, vloss_sum, vqloss_sum, vmidloss_sum, \
        vparam_loss_sum = 0., 0., 0., 0., 0., 0.
        model.train()
        for train_data0, train_data1, train_data2 in zip(train_loader0, train_loader1, train_loader2):
            trains = [train_data0, train_data1, train_data2]
            for train_data in trains:
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
                loss *= loss_weight
                if qloss is not None:
                    loss += qloss * qloss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
            if check_middle:
                midloss_sum += midloss.detach().item()
        model.eval()
        with torch.no_grad():
            for val_data0, val_data1, val_data2 in zip(val_loader0, val_loader1, val_loader2):
                vals = [val_data0, val_data1, val_data2]
                for val_data in vals:
                    image    = val_data[0].to(device = device)
                    label    = val_data[1].to(device = device)
                    outdict = model(image)
                    out   = outdict["enhanced_image"]
                    rec   = outdict["reconstruction"]
                    qloss = outdict["quantized_loss"]
                    est_params = outdict["blur_parameter"]
                    vloss, vmid_loss = branch_calc_loss(out, rec, image, label,
                                                        loss_fn,midloss_fn,partial,
                                                        reconstruct, check_middle)
                    vloss_sum += vloss.detach().item() * loss_weight
                    if qloss is not None:
                        qloss = qloss.detach().item() * qloss_weight
                        vloss_sum += qloss
                        vqloss_sum += qloss
                    if check_middle:
                        vmidloss_sum += vmid_loss.detach().item()
        num  = len(train_loader0)
        vnum = len(val_loader0)
        ez0, bet_z, bet_xy, alpha = [i for i in model.parameters()][-4:]
        if verbose:
            print([i for i in model.state_dict()][-4:])
            print(bet_z, bet_xy, alpha, ez0)
        loss_list.append(loss_sum / num)
        midloss_list.append(midloss_sum / num) if check_middle else 0
        vloss_list.append(vloss_sum / vnum)
        vmidloss_list.append(vmidloss_sum / vnum) if check_middle else 0
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('train middle loss', midloss_sum / num, epoch) if check_middle else 0
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        writer.add_scalar('val param loss', vparam_loss_sum / vnum, epoch)
        writer.add_scalar('val middle loss', vmidloss_sum / vnum, epoch) if check_middle else 0
        writer.add_scalar('val vq loss', vqloss_sum / num, epoch)
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


class ElasticWeightConsolidation():
    """
    modified from https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    """
    def __init__(self, model, prev_dataloader, loss_fn,
                 init_num_batch, is_vibrate, device):
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
        if num_params == num_fisher:
            pass
        else:
            print("(ewc) registering ewc params...")
            self.register_ewc_params(prev_dataloader,
                                     num_batch=init_num_batch)
        self.vibrate = Vibrate()

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
        log_likelihood = []
        for i, (image, label) in enumerate(dataloader):
            if i > num_batch:
                break
            if self.is_vibrate:
                vimage = self.vibrate(image)
            else:
                vimage = image
            outdict = self.model(vimage)
            out   = outdict["enhanced_image"]
            label = label.to(self.device)
            log_likelihood.append(torch.log(self.loss_fn(out, label)))
        log_likelihood = sum(log_likelihood) / len(log_likelihood)
        grad_log_likelihood = torch.autograd.grad(log_likelihood,
                                                  self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__')
                             for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names,
                                           grad_log_likelihood):
            self.model.register_buffer(_buff_param_name+"_estimated_fisher",
                                       param.data.clone() ** 2)

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
            losses.append((fisher * (param - mean) ** 2).sum())
        return (lambda_ / 2) * sum(losses)