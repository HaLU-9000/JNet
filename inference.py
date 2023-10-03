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

vibrate = Vibrate()

class PretrainingInference():
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
        

        config = open(os.path.join("experiments/configs", f"{model_name}.json"))
        self.configs         = json.load(config)
        self.params          = self.configs["params"]

        val_dataset_params   = self.configs["pretrain_val_dataset"]

        JNet = model.JNet(self.params)
        self.JNet = JNet.to(device = self.device)
        self.JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)

        val_dataset   = RandomCutDataset(
            folderpath    = val_dataset_params["folderpath"]   ,
            imagename     = val_dataset_params["imagename"]    , 
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
            image   = vibrate(image).detach().clone()
            outdict = self.JNet(image)
            output  = outdict["enhanced_image"]
            qloss   = outdict["quantized_loss"]
            qloss = qloss.item() if qloss is not None else 0
            image   = image[0].detach().cpu().numpy()
            label   = label[0].detach().cpu().numpy()
            output  = output[0].detach().cpu().numpy()
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
                "BCE": bces,}

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
            image_z   = np.max(image [0, :  , i:i+mip, :]    , axis=0)
            output_z  = np.max(output[0, :  , i:i+mip, :]    , axis=0)
            label_z   = np.max(label [0, :  , i:i+mip, :]    , axis=0)
            #fig = plt.figure()
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

    def visualize_oldversion(self, results, path='result'):
        j   = self.configs["visualization"]["z_stack"]
        j_s = j // self.params["scale"]
        i   = self.configs["visualization"]["x_slice"]
        for n, [image, output, label, qloss] in enumerate(results):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference for simulation data')
    parser.add_argument('model_name')
    args   = parser.parse_args()
    inference = PretrainingInference(args.model_name)
    results = inference.get_result(5)
    print(inference.evaluate(results))
    inference.visualize(results)
