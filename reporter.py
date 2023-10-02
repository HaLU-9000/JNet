import os
import argparse
import json
from mdutils.mdutils import MdUtils
import torch

parser = argparse.ArgumentParser(description='generates report')
parser.add_argument('model_name')
args   = parser.parse_args() 
configs = open(os.path.join("experiments/configs", f"{args.model_name}.json"))
configs = json.load(configs)

md = MdUtils(file_name=f'./experiments/reports/{args.model_name}.md')
md.new_header(level=1, title=args.model_name)
md.new_line(configs["explanation"])
md.new_line(f'pretrained model : {configs["pretrained_model"]}')
md.new_header(level=2, title="parameters")
for p in configs["params"]:
    md.new_line(f'{p}\t{configs["params"][p]}')

md.create_md_file()