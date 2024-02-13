import torch
import torchrl
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from dataset import Vibrate, Mask

vibrate = Vibrate()

def imagen_instantblur(model, label, device, params):
    #image  = model.image.emission.sample(label, params)
    out    = model.image.blur(label)
    image  = out["out"]
    image  = model.image.noise(image)
    image  = model.image.preprocess.sample(image)
    return image

def _loss_fnz(_input, mask, target):
    if mask is not None:
        label_loss = F.gaussian_nll_loss(
            input  = _input*mask            , 
            target = target                 ,
            var    = torch.ones_like(_input) )
    else:
        label_loss = 0.
    trnorm   = torchrl.modules.TruncatedNormal(
        loc     = torch.zeros_like(_input),
        scale   = torch.ones_like(_input) ,
        upscale = torch.zeros_like(_input) )
    log_prob = trnorm.log_prob(torch.log(_input))
    nll_loss = - torch.mean(log_prob) * 1/100
    return label_loss + nll_loss

def pretrain_loop(n_epochs             ,
                  optimizer            ,
                  model                ,
                  loss_fnx             ,
                  loss_fnz             ,
                  train_loader         ,
                  val_loader           ,
                  device               ,
                  path                 ,
                  savefig_path         ,
                  model_name           ,
                  params               ,
                  train_dataset_params ,
                  scheduler   = None   ,
                  es_patience = 10     ,
                  is_vibrate  = False  ,
                  wx          = 1.     ,
                  wz          = 1.     ,
                  ):

    earlystopping = EarlyStopping(name        = model_name ,
                                  path        = path       ,
                                  patience    = es_patience,
                                  window_size = 5          ,
                                  metric      = "mean"     ,
                                  verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss", "validatation loss"])
    loss_list, vloss_list = [], []
    vibrate = Vibrate()
    mask = Mask()
    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        model.train()
        for train_data in train_loader:
            labelx = train_data["labelx"].to(device = device)
            labelz = train_data["labelz"].to(device = device)
            with torch.no_grad():
                image = imagen_instantblur(model  = model ,
                                           label  = labelz,
                                           device = device,
                                           params = params,)
            image = mask.apply_mask(train_dataset_params["mask"]      ,
                                    image                             ,
                                    train_dataset_params["mask_size"] ,
                                    train_dataset_params["mask_num"]  ,)
            vimage = vibrate(image) if is_vibrate else image
            outdict = model(vimage)
            out = outdict["enhanced_image" ]
            lum = outdict["estim_luminance"]
            lossx  = loss_fnx(out, labelx)
            lossz  = _loss_fnz(lum, labelx, labelz)
            loss = wx * lossx + wz * lossz
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
        
        vloss_sum = 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                labelx = val_data["labelx"].to(device = device)
                labelz = val_data["labelz"].to(device = device)
                image = imagen_instantblur(model  = model ,
                                           label  = labelz,
                                           device = device,
                                           params = params,)
                vimage = vibrate(image) if is_vibrate else image
                outdict = model(vimage)
                out   = outdict["enhanced_image"]
                lum   = outdict["estim_luminance"]
                lossx = loss_fnx(out, labelx)
                lossz = _loss_fnz(lum, labelx, labelz)
                vloss = wx * lossx + wz * lossz
                vloss_sum += vloss.detach().item()
        
        num  = len(train_loader)
        vnum = len(val_loader)
        loss_list.append(loss_sum / num)
        vloss_list.append(vloss_sum / vnum)
        writer.add_scalar('train loss', loss_sum / num, epoch)
        writer.add_scalar('val loss', vloss_sum / vnum, epoch)
        row = pd.DataFrame([[loss_list[-1], vloss_list[-1]]],
                           columns = train_curve.columns)
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {vloss_list[-1]}')
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        earlystopping((vloss_sum / vnum), model, condition = True)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)

def luminance_adjustment(rec, image):
    e = 1e-7
    cov = torch.mean((rec - torch.mean(rec)) * (image - torch.mean(image)))
    var = (torch.mean((rec - torch.mean(rec)) ** 2))
    beta  = (cov + e) / (var + e)
    beta  = torch.clip(beta, min=0., max=30.)           
    alpha = torch.mean(image) - beta * torch.mean(rec)  
    return alpha + beta * rec

def finetuning_loop(n_epochs               ,
                    optimizer              ,
                    model                  ,
                    loss_fn                ,
                    train_loader           ,
                    val_loader             ,
                    device                 ,
                    path                   ,
                    savefig_path           ,
                    model_name             ,
                    ewc                    ,
                    train_dataset_params   ,
                    adjust_luminance       ,
                    scheduler    = None    ,
                    es_patience  = 10      ,
                    is_vibrate   = False   ,
                    loss_weight  = 1.      ,
                    ewc_weight   = 100000 ,
                    qloss_weight = 1/100   ,
                    ploss_weight = 1/100   ,
                    ):
    
    earlystopping = EarlyStopping(name        = model_name ,
                                  path        = path       ,
                                  patience    = es_patience,
                                  window_size = 5          ,
                                  metric      = "mean"     ,
                                  verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss"    ,
                                        "validatation loss"] )
    loss_list, vloss_list= [], []
    vibrate = Vibrate()
    mask = Mask()
    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        model.train()
        for train_data in train_loader:
            # should write code for finetuning!
            image  = train_data["image"].to(device = device)
            _image = mask.apply_mask(train_dataset_params["mask"]      ,
                                     image                             ,
                                     train_dataset_params["mask_size"] ,
                                     train_dataset_params["mask_num"]  ,)
            vimage = vibrate(_image) if is_vibrate else image
            outdict = model(vimage)
            rec     = outdict["reconstruction"]
            qloss   = outdict["quantized_loss"]
            ploss   = outdict["psf_loss"]
            if adjust_luminance:
                rec = luminance_adjustment(rec, image)
            loss  = loss_fn(rec, image) * loss_weight
            if ewc is not None:
                loss += ewc.calc_ewc_loss(ewc_weight)
            if qloss is not None:
                loss += qloss * qloss_weight
            if ploss is not None:
                loss += ploss * ploss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
        vloss_sum, vqloss_sum, vparam_loss_sum = 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                image   = val_data["image"].to(device = device)
                vimage  = vibrate(image) if is_vibrate else image
                outdict = model(vimage)
                rec     = outdict["reconstruction"]
                qloss   = outdict["quantized_loss"]
                ploss   = outdict["psf_loss"]
                if adjust_luminance:
                    rec = luminance_adjustment(rec, image)
                vloss   = loss_fn(rec, image) * loss_weight
                vloss_sum += vloss.detach().item() * loss_weight
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss_sum += ploss

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
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {vloss_list[-1]}')
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        earlystopping((vloss_sum / vnum), model, condition = True)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png',
                format='png', dpi=500)

def finetuning_with_simulation_loop(
                    n_epochs               ,
                    optimizer              ,
                    model                  ,
                    loss_fn                ,
                    train_loader           ,
                    val_loader             ,
                    device                 ,
                    path                   ,
                    savefig_path           ,
                    model_name             ,
                    params                 ,
                    ewc                    ,
                    train_dataset_params   ,
                    adjust_luminance       ,
                    scheduler    = None    ,
                    es_patience  = 10      ,
                    is_vibrate   = False   ,
                    loss_weight  = 1.      ,
                    ewc_weight   = 1000000 ,
                    qloss_weight = 1/100   ,
                    ploss_weight = 1/100   ,
                    ):
    
    earlystopping = EarlyStopping(name        = model_name ,
                                  path        = path       ,
                                  patience    = es_patience,
                                  window_size = 5          ,
                                  metric      = "mean"     ,
                                  verbose     = True       ,)
    writer = SummaryWriter(f'runs/{model_name}')
    train_curve = pd.DataFrame(columns=["training loss"    ,
                                        "validatation loss"] )
    loss_list, vloss_list= [], []
    vibrate = Vibrate()
    mask = Mask()
    torch.save(model.image.state_dict(),
               f"{path}/{model_name}_pre.pt")
    for epoch in range(1, n_epochs + 1):
        loss_sum = 0.
        model.train()
        for train_data in train_loader:
            torch.save(model.image.state_dict(),
                       f"{path}/{model_name}_tmp.pt")
            model.image.load_state_dict(
                torch.load(f"{path}/{model_name}_pre.pt"))
            labelz = train_data["labelz"].to(device = device)
            with torch.no_grad():
                image = imagen_instantblur(model  = model ,
                                           label  = labelz,
                                           device = device,
                                           params = params,)
            _image = mask.apply_mask(train_dataset_params["mask"]     ,
                                    image                             ,
                                    train_dataset_params["mask_size"] ,
                                    train_dataset_params["mask_num"]  ,)
            model.image.load_state_dict(
                torch.load(f"{path}/{model_name}_tmp.pt"))
            vimage = vibrate(_image) if is_vibrate else image
            outdict = model(vimage)
            rec     = outdict["reconstruction"]
            lum     = outdict["estim_luminance"]
            qloss   = outdict["quantized_loss"]
            ploss   = outdict["psf_loss"]
            if adjust_luminance:
                rec = luminance_adjustment(rec, image)
            loss = loss_fn(rec, image) * loss_weight
            loss_z = _loss_fnz(
                _input =lum ,
                mask   =None,
                target =None )
            loss = loss + loss_z
            if ewc is not None:
                loss += ewc.calc_ewc_loss(ewc_weight)
            if qloss is not None:
                loss += qloss * qloss_weight
            if ploss is not None:
                loss += ploss * ploss_weight
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            loss_sum += loss.detach().item()
        vloss_sum, vqloss_sum, vparam_loss_sum = 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                torch.save(model.image.state_dict(),
                           f"{path}/{model_name}_tmp.pt")
                model.image.load_state_dict(
                    torch.load(f"{path}/{model_name}_pre.pt"))
                labelz = val_data["labelz"].to(device = device)
                image = imagen_instantblur(model  = model ,
                                           label  = labelz,
                                           device = device,
                                           params = params,)
                model.image.load_state_dict(
                    torch.load(f"{path}/{model_name}_tmp.pt"))
                vimage  = vibrate(image) if is_vibrate else image
                outdict = model(vimage)
                rec     = outdict["reconstruction"]
                lum     = outdict["estim_luminance"]
                qloss   = outdict["quantized_loss"]
                ploss   = outdict["psf_loss"]
                if adjust_luminance:
                    rec = luminance_adjustment(rec, image)
                vloss   = loss_fn(rec, image) * loss_weight
                loss_z = _loss_fnz(
                    _input =lum ,
                    mask   =None,
                    target =None )
                loss = loss + loss_z
                vloss_sum += vloss.detach().item() * loss_weight
                if ewc is not None:
                    vloss_sum += ewc.calc_ewc_loss(ewc_weight).detach().item()
                if qloss is not None:
                    qloss = qloss.detach().item() * qloss_weight
                    vloss_sum += qloss
                    vqloss_sum += qloss
                if ploss is not None:
                    ploss = ploss.detach().item() * ploss_weight
                    vloss_sum += ploss

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
        train_curve = pd.concat([train_curve, row], ignore_index=True)
        train_curve.to_csv(f"./experiments/traincurves/{model_name}.csv",
                           index=False)
        
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}, Train {loss_list[-1]}, Val {vloss_list[-1]}')
        if scheduler is not None:
            scheduler.step(epoch, vloss_list[-1])
        earlystopping((vloss_sum / vnum), model, condition = True)
        if earlystopping.early_stop:
            break
    plt.plot(loss_list , label='train loss')
    plt.plot(vloss_list, label='validation loss')
    plt.legend()
    plt.savefig(f'{savefig_path}/{model_name}_train.png', format='png', dpi=500)

class ElasticWeightConsolidation():
    """
    modified from
    https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
    """
    def __init__(self, model, params, prev_dataloader, loss_fnx, loss_fnz,wx, wz,
                 init_num_batch, ewc_dataset_params, is_vibrate, device):
        self.model = model
        self.device = device
        self.is_vibrate = is_vibrate
        self.loss_fnx = loss_fnx
        self.loss_fnz = loss_fnz
        self.wx = wx
        self.wz = wz
        self.params = params
        self.prev_dataloader = prev_dataloader
        self.ewc_dataset_params = ewc_dataset_params
        num_fisher = 0

        for name, _ in self.model.named_buffers():
            if '_estimated_fisher' in name:
                num_fisher += 1        
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
        mask = Mask()
        grad_log_likelihood_data = []
        for i, val_data in enumerate(dataloader):
            if i > num_batch:
                break
            labelx = val_data["labelx"].to(device = self.device)
            labelz = val_data["labelz"].to(device = self.device)
            with torch.no_grad():
                image = imagen_instantblur(model  = self.model ,
                                           label  = labelz     ,
                                           device = self.device,
                                           params = self.params,)
            image = mask.apply_mask(self.ewc_dataset_params["mask"]      ,
                                    image                                ,
                                    self.ewc_dataset_params["mask_size"] ,
                                    self.ewc_dataset_params["mask_num"]  ,)
            vimage = vibrate(image) if self.is_vibrate else image
            outdict = self.model(vimage)
            out   = outdict["enhanced_image"]
            lum   = outdict["estim_luminance"]
            lossx = self.loss_fnx(out, labelx)
            lossz = _loss_fnz(lum,labelx,labelz)
            vloss = self.wx * lossx + self.wz * lossz
            log_likelihood = torch.log(vloss)
            grad_log_likelihood = torch.autograd.grad(log_likelihood,
                                                      self.model.parameters(),
                                                      retain_graph=False,
                                                      allow_unused=True)
            grad_log_likelihood = list(grad_log_likelihood)
            for n, param in enumerate(grad_log_likelihood):
                if param is None:
                        param_data_clone = None
                else:
                    param_data_clone = param.data.clone() / len(dataloader)
                if len(grad_log_likelihood_data) != len(grad_log_likelihood):
                    grad_log_likelihood_data.append(param_data_clone)
                elif param is not None:
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