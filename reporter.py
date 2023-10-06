import os
import codecs
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from mdutils.mdutils import MdUtils
import pandas as pd
import model_new as model
import inference

parser = argparse.ArgumentParser(description='generates report')
parser.add_argument('model_name')
args   = parser.parse_args() 
configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs = json.load(configs)
md = MdUtils(file_name=f'./experiments/reports/{args.model_name}.md')
###########
## Title ##
###########
md.new_header(level=1, title=f"{args.model_name} Report")
md.new_line(configs["explanation"])
md.new_line(f'pretrained model : {configs["pretrained_model"]}')
################
## Parameters ##
################
md.new_header(level=2, title="Model Parameters")
md.new_line()
params_list = ["Parameter", "Value", "Comment"]
n = 0
for param in configs["params"]:
    if "$" not in param:
        if ("$"+param) in configs["params"]:
            comment = configs["params"]["$"+param]
        else:
            comment = ""
        params_list.extend([param, configs["params"][param], comment])
        n += 1
md.new_table(columns=3, rows=n+1, text=params_list, text_align="left")
##################
## Architecture ##
##################
print(model.JNet(configs["params"]), file = codecs.open("experiments/tmp/"+args.model_name+".txt", "w", "utf-8"))
md.new_header(level=2, title="Architecture")
md.new_line()
md.new_line("```")
with open("experiments/tmp/"+args.model_name+".txt") as f:
    lines = f.readlines()
    for line in lines:
        md.new_line(line.rstrip("\n"))
    f.close()
md.new_line("```")
md.new_line()
##############
## Datasets ##
##############
md.new_header(level=2, title="Datasets and other training details")
for name in [
                "simulation_data_generation",
                "pretrain_dataset"          ,
                "pretrain_val_dataset"      ,
                "train_dataset"             , 
                "val_dataset"               , 
                "pretrain_loop"             ,
                "train_loop"                ,
              ]:
    default_list = ["Parameter", "Value"]
    for n, param in enumerate(configs[name]):
        default_list.extend([param, configs[name][param]])
    md.new_header(level=3, title=name)
    md.new_table(columns=2, rows=len(default_list)//2, text=default_list, text_align="left")
#####################
## Training Curves ##
#####################
md.new_header(level=2, title="Training Curves")
md.new_line()
md.new_header(level=3, title="Pretraining")
df = pd.read_csv(f'experiments/traincurves/{configs["pretrained_model"]}.csv')
plt.figure()
df.plot()
plt.xlabel("epoch")
plt.ylabel(configs["pretrain_loop"]["loss_fn"])
path = f'./experiments/tmp/{configs["pretrained_model"]}_train.png'
plt.savefig(path)
md.new_line(md.new_reference_image(text="pretrained_model", path=path[1:]))
md.new_header(level=3, title="Finetuning")
df = pd.read_csv(f'experiments/traincurves/{args.model_name}.csv')
plt.figure()
df.plot()
plt.xlabel("epoch")
loss_metrics = configs["train_loop"]["loss_fn"]+" + "\
    +"qloss "+"* "+str(configs["train_loop"]["qloss_weight"])
if configs["train_loop"]["ewc"] is not None:
    loss_metrics += "ewc"
plt.ylabel(loss_metrics)
path = f'./experiments/tmp/{args.model_name}_train.png'
plt.savefig(path)
md.new_line(md.new_reference_image(text="finetuned", path=path[1:]))
plt.clf()
plt.close()
#############
## Results ##
#############
md.new_header(level=2, title="Results")
# generate results
num_result = 5
infer = inference.PretrainingInference(args.model_name)
results = infer.get_result(num_result)
evals = infer.evaluate(results)
md.new_line(f'mean MSE: {np.mean(evals["MSE"])}, mean BCE: {np.mean(evals["BCE"])}')
infer.visualize(results)
slice_list = ["plane", "depth"]
type_list  = ["original", "output", "label"]
for n in range(num_result):
    md.new_header(level=3, title=f"{n}")
    for slice in slice_list:
        im_list = []    
        for tp in type_list:
            path = f'./experiments/imagetests/{args.model_name}_{n}_{tp}_{slice}.png'
            im_list.append(md.new_reference_image(text=f"{n}_{tp}_{slice}", path=path[1:]))
        md.new_table(columns=3, rows=2, text=[*type_list, *im_list],)
        md.new_line(f'MSE: {evals["MSE"][n]}, BCE: {evals["BCE"][n]}')
        md.new_line()

btype_list = ["original", "output", "reconst"]
binfer = inference.BeadsInference(args.model_name)
results = binfer.get_result()
bevals  = binfer.evaluate(results)
binfer.visualize(results)
for n in range(len(results)):
    image_name = binfer.images[n][len(binfer.datapath)+1:-3]
    md.new_header(level=3, title=image_name)
    im_list = []
    for tp in btype_list:
        path = f'./experiments/imagetests/{args.model_name}_{image_name}_{tp}_depth.png'
        im_list.append(md.new_reference_image(text=f"{image_name}_{tp}_{slice}", path=path[1:]))
    md.new_table(columns=3, rows=2, text=[*btype_list, *im_list],)
    md.new_line(f'volume: {bevals["volume"][n]}, MSE: {bevals["MSE"][n]}, quantized loss: {bevals["qloss"][n]}')
    md.new_line()



#########
## End ##
#########
md.new_line()
md.create_md_file()