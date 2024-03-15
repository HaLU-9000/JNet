import os
import argparse
import json

import utils

# get args
parser = argparse.ArgumentParser(description='')
parser.add_argument('image_name')
parser.add_argument('model_name')
parser.add_argument('-t', '--train_mode')
args = parser.parse_args()
configs = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs = json.load(configs)
params  = configs["params"]
shape = [80, 112, 112]

# image load
image_org  = utils.load_anything(args.image_name)
image      = utils.ImageProcessing(image_org)

# model load
model = utils.init_model(params, is_finetuning = True)
utils.load_model_weight(model, model_name = args.model_name)
utils.mount_model_to_device(model, configs = configs)
model.eval()

# batch process
image.process_image(model, params, shape, "enhanced_image")
print(image.processed_image.shape)

# save image
image.save_processed_image(file=f"_apply_test/{args.image_name[-14:-4]}",
                           format="tif",
                           bit=16)