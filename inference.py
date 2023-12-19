import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import RandomCutDataset
import model_new as model
import old_model
from dataset import Vibrate
from utils import array_to_tif, load_anything

vibrate = Vibrate()

class PretrainingInference():
    def __init__(self, model_name, pretrain=True):
        self.device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs         = json.load(config)
        self.model_name      = self.configs["pretrained_model"] if pretrain else model_name
        self.params          = self.configs["params"]
        val_dataset_params   = self.configs["pretrain_val_dataset"]

        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        if os.path.isfile(f'model/{self.model_name}.pt'):
            self.JNet.load_state_dict(torch.load(f'model/{self.model_name}.pt'),
                                      strict=False)
        self.JNet.eval()

        val_dataset   = RandomCutDataset(
            folderpath    = val_dataset_params["folderpath"]   ,
            labelname     = val_dataset_params["labelname"]    ,
            size          = val_dataset_params["size"]         ,
            cropsize      = val_dataset_params["cropsize"]     , 
            I             = val_dataset_params["I"]            ,
            low           = val_dataset_params["low"]          ,
            high          = val_dataset_params["high"]         ,
            scale         = val_dataset_params["scale"]        ,  ## scale
            mask          = val_dataset_params["mask"]         ,
            mask_size     = val_dataset_params["mask_size"]    ,
            mask_num      = val_dataset_params["mask_num"]     ,  #( 1% of image)
            surround      = val_dataset_params["surround"]     ,
            surround_size = val_dataset_params["surround_size"],
            seed          = val_dataset_params["seed"]         ,
                                        )
        self.val_loader  = DataLoader(
            val_dataset                   ,
            batch_size  = 1               ,
            shuffle     = False           ,
            pin_memory  = False           ,
            num_workers = os.cpu_count()  ,
                         )
    
    def get_result(self, num_results)->list:
        #with torch.no_grad():
        results = []
        for n, val_data in enumerate(self.val_loader):
            if n >= num_results:
                break
            if self.configs["pretrain_loop"]["is_instantblur"]:
                label = val_data[1].to(device = self.device)
                image = self.JNet.image.emission.sample(label, self.params)
                image = self.JNet.image.blur(image)
                image = self.JNet.image.noise(image)
                image = self.JNet.image.preprocess(image)
            else:
                image    = val_data[0].to(device = self.device)
                label    = val_data[1].to(device = self.device)
            if self.configs["pretrain_loop"]["is_vibrate"]:
                image   = vibrate(image).detach().clone()
            outdict = self.JNet(image)
            output  = F.sigmoid(outdict["enhanced_image"])
            qloss   = outdict["quantized_loss"]
            qloss = qloss.item() if qloss is not None else 0
            image   = image[0].detach().cpu().numpy()
            label   = label[0].detach().cpu().numpy()
            output  = output[0].detach().cpu().numpy()
            array_to_tif(f"_result_cbias/{self.model_name}_image_{n}.tif", image)
            array_to_tif(f"_result_cbias/{self.model_name}_out_{n}.tif", output)
            array_to_tif(f"_result_cbias/{self.model_name}_label_{n}.tif", label)
            results.append([image, output, label, qloss])
        return results
        
    def evaluate(self, results)->list:
        mses = []
        bces = []
        for n, [image, output, label, qloss] in enumerate(results):
            mse = np.mean(((label - output) ** 2).flatten())
            bce = np.mean(-(label*np.log(output) + (1. - label)*np.log(1. - output)).flatten())
            mses.append(mse)
            bces.append(bce)
        return {"MSE": mses,
                "BCE": bces }
    
    def threshold_argmax_f1score(self, results):
        best_ths = []
        for [_, output, label, _] in results:
            scores = []
            for threshold in range(100):
                t = threshold / 100
                score = self.calc_f1_score(output, label, t)
                scores.append(score)
            best_th = np.argmax(scores)
            best_ths.append(best_th)
        return np.mean(best_ths) / 100
    
    def threshold_argmax_jaccardscore(self, results):
        best_ths = []
        for [_, output, label, _] in results:
            scores = []
            for threshold in range(100):
                t = threshold / 100
                score = self.calc_jaccard_score(output, label, t)
                scores.append(score)
            best_th = np.argmax(scores)
            best_ths.append(best_th)
        return np.mean(best_ths) / 100

    def calc_f1_score(self, pred, label, threshold):
        pred = pred >= threshold
        tp = np.sum(pred * label)
        fp = np.sum(pred * (1 - label))
        fn = np.sum((1 - pred) * label)
        score = tp / (tp + 1/2 * (fp + fn))
        return score
    
    def calc_jaccard_score(self, pred, label, threshold):
        pred = pred >= threshold
        tp = np.sum(pred * label)
        fp = np.sum(pred * (1 - label))
        fn = np.sum((1 - pred) * label)
        score = tp / (tp + (fp + fn))
        return score
        
    def visualize(self, results):
        for n, [image, output, label, qloss] in enumerate(results):
            path = self.configs["visualization"]["path"] 
            j   = self.configs["visualization"]["z_stack"]
            j_s = j // self.params["scale"]
            i   = self.configs["visualization"]["x_slice"]
            mip = self.configs["visualization"]["mip"]
            mip_s = mip * self.params["scale"]

            image_xy  = np.max(image [0, j_s:j_s+mip_s, :, :], axis=0)
            output_xy = np.max(output[0, j  :j+mip, :, :]    , axis=0)
            label_xy  = np.max(label [0, j  :j+mip, :, :]    , axis=0)
            image_z   = np.max(image [0, :  , i:i+mip, :]    , axis=1)
            output_z  = np.max(output[0, :  , i:i+mip, :]    , axis=1)
            label_z   = np.max(label [0, :  , i:i+mip, :]    , axis=1)
            plt.clf()
            plt.close()
            plt.axis("off")
            plt.imshow(image_xy, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{n}_original_plane.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()
            plt.axis("off")
            plt.imshow(output_xy, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{n}_output_plane.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()
            plt.axis("off")
            plt.imshow(label_xy, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{n}_label_plane.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()
            plt.axis("off")
            plt.imshow(image_z, cmap='gray', vmin=0.0, aspect=self.params["scale"])
            plt.savefig(path + f'/{self.model_name}_{n}_original_depth.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()
            plt.axis("off")
            plt.imshow(output_z, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{n}_output_depth.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()
            plt.axis("off")
            plt.imshow(label_z, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{n}_label_depth.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

    def visualize_oldversion(self, results, path='result', verbose=False, threshold=-1):
        j   = self.configs["visualization"]["z_stack"]
        j_s = j // self.params["scale"]
        i   = self.configs["visualization"]["x_slice"]
        for n, [image, output, label, qloss] in enumerate(results):
            if threshold != -1:
                output = output >= threshold
            fig = plt.figure(figsize=(25, 15))
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax4 = fig.add_subplot(234)
            ax5 = fig.add_subplot(235)
            ax6 = fig.add_subplot(236)
            ax1.set_axis_off()
            ax2.set_axis_off()
            ax3.set_axis_off()
            ax4.set_axis_off()
            ax5.set_axis_off()
            ax6.set_axis_off()
            plt.subplots_adjust(hspace=-0.0)
            ax1.imshow(image[0, j_s, :, :],
                cmap='gray', vmin=0.0, aspect=1)
            ax2.imshow(output[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax3.imshow(label[0, j, :, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax4.imshow(image[0, :, i, :],
                cmap='gray', vmin=0.0, aspect= self.params["scale"])
            ax5.imshow(output[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            ax6.imshow(label[0, :, i, :],
                cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_result_{n}.png',
                format='png', dpi=250)
            if verbose:
                plt.show()
    
    def del_model(self):
        self.JNet = self.JNet.cpu()
        del self.JNet
            
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import RealDensityDataset, sequentialflip

import model_new as model

#device = (torch.device('cuda') if torch.cuda.is_available()
#          else torch.device('cpu'))
#print(f"Inference on device {device}.")
#
#parser = argparse.ArgumentParser(description='inference with bead images.')
#parser.add_argument('model_name')
#args   = parser.parse_args()
#
#configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
#configs              = json.load(configs)
#params               = configs["params"]
#
#JNet = model.JNet(params)
#JNet = JNet.to(device = device)
#j   = 12
#j_s = j // params["scale"]
#i = 64
#
#JNet.load_state_dict(torch.load(f'model/{args.model_name}.pt'), strict=False)
#JNet.eval()
#dirpath = "_beads_roi_extracted_stackreg"
#images = [os.path.join(dirpath, f) for f in sorted(os.listdir(dirpath))]
#print(images)
#outputs = []
#loss_fn = nn.MSELoss()
#for image_name in images[:-1]:
#    image_ = torch.load(image_name, map_location="cuda").to(torch.float32)
#    image = image_#(torch.clip(image_, min=0.1, max=1.) - 0.1) / (1.0 - 0.1)
#    outdict = JNet(image.to("cuda").unsqueeze(0))
#    output  = outdict["enhanced_image"]
#    output  = output.detach().cpu()
#    reconst = outdict["reconstruction"]
#    #loss    = loss_fn(reconst, image).item()
#    qloss   = outdict["quantized_loss"]
#    print("output ", torch.sum(output).item() * (0.05 * 0.05 * 0.05))
#    outputs.append(torch.sum(output).item() * (0.05 * 0.05 * 0.05))
#    reconst = reconst.squeeze(0).detach().cpu().numpy()
#    #losses.append(loss)
#    fig = plt.figure(figsize=(10, 10))
#    ax1 = fig.add_subplot(121)
#    ax2 = fig.add_subplot(122)
#    ax1.set_axis_off()
#    ax2.set_axis_off()
#    ax1.set_title('original image')
#    ax2.set_title(f'reconstruct image\n{args.model_name}')
#    plt.subplots_adjust(hspace=-0.0)
#    ax1.imshow(image_[0, :, i, :].to(device='cpu'),
#            cmap='gray', vmin=0.0, vmax=1.0, aspect=params["scale"])
#    ax2.imshow(output[0, 0, :, i, :],
#            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
#    plt.savefig(f'result/{args.model_name}_new_{image_name[30:-3]}.png', format='png', dpi=250)
#print(args.model_name)
#print(np.mean(np.array(outputs)))


class BeadsInference():
    def __init__(self, model_name, pretrain=True, threshold=-1):
        self.device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs = json.load(config)
        self.params  = self.configs["params"]
        self.params["reconstruct"]     = True
        self.params["apply_vq"]        = True
        #if pretrain == False:
        #    self.params["use_x_quantized"] = True
        if threshold != -1:
            self.params["threshold"] = threshold
        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        self.psf_pretrain = self.JNet.image.blur.show_psf_3d()
        self.model_name = self.configs["pretrained_model"] if pretrain else model_name
        if os.path.isfile(f'model/{self.model_name}.pt'):
            self.JNet.load_state_dict(torch.load(f'model/{self.model_name}.pt'),
                                      strict=False)
        self.psf_post = self.JNet.image.blur.show_psf_3d()
        self.JNet.eval()
        #self.JNet.tau = 0.1
    
    def get_result(self, datapath="_20231208_tsuji_beads_roi_stackreged"):
        self.images  = [os.path.join(datapath, f) for f in sorted(os.listdir(datapath))]
        self.datapath = datapath
        results  = []
        for image_name in self.images:
            if image_name[-3:] == ".pt":
                image   = torch.load(image_name,
                                 map_location=self.device).to(torch.float32)
            else:
                image = load_anything(image_name).to(self.device).to(torch.float32)
            outdict = self.JNet(image.to(self.device).unsqueeze(0))
            output  = outdict["enhanced_image"]
            output  = output.squeeze(0).detach().cpu().numpy()
            qloss   = outdict["quantized_loss"]
            qloss   = qloss.item() if qloss is not None else 0
            reconst = outdict["reconstruction"]
            reconst = reconst.squeeze(0).detach().cpu().numpy()
            image   = image.detach().cpu().numpy()
            results.append([image, output, reconst, qloss])
        return results
    
    def evaluate(self, results):
        volumes = []
        mses    = []
        qlosses = []
        for [image, output, rec, qloss] in results:
            volume = np.sum(output).item() * \
                (self.params["res_lateral"] ** 3)
            e = 1e-7
            cov = np.mean((rec - np.mean(rec)) * (image - np.mean(image)))
            var = (np.mean((rec - np.mean(rec)) ** 2))
            beta  = (cov + e) / (var + e)
            alpha = np.mean(image) - beta * np.mean(rec)
            mse = np.mean(((image - (alpha + beta * rec)) ** 2).flatten())
            volumes.append(volume)
            mses.append(mse)
            qlosses.append(qloss)
        return {"volume": volumes,
                "MSE"   : mses   ,
                "qloss" : qlosses }
            
    def visualize(self, results):
        for n, [image, output, reconst, qloss] in enumerate(results):
            rec = reconst
            e = 1e-7
            cov = np.mean((rec - np.mean(rec)) * (image - np.mean(image)))
            var = (np.mean((rec - np.mean(rec)) ** 2))
            beta  = (cov + e) / (var + e)
            alpha = np.mean(image) - beta * np.mean(rec)
            reconst = alpha + beta * rec
            path = self.configs["visualization"]["path"] 
            j    = self.configs["visualization"]["z_stack"]
            i    = self.configs["visualization"]["x_slice"]
            mip  = self.configs["visualization"]["mip"]
            image_z   = np.max(image [0,  : , :, i:i+mip], axis=2)
            output_z  = np.max(output[0,  : , :, i:i+mip], axis=2)
            reconst_z = np.max(reconst[0, : , :, i:i+mip], axis=2)
            plt.axis("off")
            plt.imshow(image_z, cmap='gray', vmin=0.0, aspect=self.params["scale"])
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_original_depth.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

            plt.axis("off")
            plt.imshow(output_z, cmap='gray', vmin=0.0, aspect=1)
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_output_depth.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

            plt.axis("off")
            plt.imshow(reconst_z, cmap='gray', vmin=0.0, aspect=self.params["scale"])
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_reconst_depth.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

            plt.axis("off")
            plt.imshow((reconst_z - image_z + 1) / 2,  vmin=0.0,  vmax=1.0, aspect=self.params["scale"], cmap='seismic')
            plt.savefig(path + f'/{self.model_name}_{self.images[n][len(self.datapath)+1:-3]}_heatmap_depth.png',
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
            plt.clf()
            plt.close()

    def psf_visualize(self):
        psfpre  = self.psf_pretrain.detach().cpu().numpy()
        psfpost = self.psf_post.detach().cpu().numpy()
        array_to_tif(f"_result_cbias/{self.model_name}_psfpre.tif", psfpre)
        array_to_tif(f"_result_cbias/{self.model_name}_psfpost.tif", psfpost)

        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpre[:, self.params["size_x"]//2, :], aspect=10)
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_pre.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
        plt.clf()
        plt.close()
        plt.axis("off")
        plt.imshow(psfpost[:, self.params["size_x"]//2, :], aspect=10)
        plt.savefig(f'./{self.configs["visualization"]["path"]}/{self.model_name}_psf_post.png', 
                        format='png',dpi=250,bbox_inches='tight',pad_inches=0)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference for simulation data')
    parser.add_argument('model_name')
    args   = parser.parse_args()
    inference = BeadsInference(args.model_name, pretrain=True)
    inference.psf_visualize()
    #inference = PretrainingInference(args.model_name)
    #results = inference.get_result(5)
    #print(inference.evaluate(results))
    #inference.visualize(results)
