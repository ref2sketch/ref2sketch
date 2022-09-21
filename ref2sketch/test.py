from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils2 import is_image_file, load_img, save_img, test_load_img

# Testing settings
parser = argparse.ArgumentParser(description='ref2sketch implementation')
parser.add_argument('--name_weight', required=True, help='ref2sketch_deep')
parser.add_argument('--name_data', required=True, help='examples')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "./checkpoints/{}_pretrained.pth".format(opt.name_weight)
net_g = torch.load(model_path,map_location='cuda:0').to(device)

if opt.direction == "a2b":
    image_dir = "datasets/{}/test/a/".format(opt.name_data)
else:
    image_dir = "datasets/{}/test/b/".format(opt.name_data)

style_test_image_dir ="datasets/{}/test/c/".format(opt.name_data)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
style_test_image_dir_name = [x for x in os.listdir(style_test_image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize([0.5], [0.5])]
                  

transform = transforms.Compose(transform_list)
for style_name in style_test_image_dir_name:
    style_img = load_img(style_test_image_dir+style_name)
    for image_name in image_filenames:
        img,w,h = test_load_img(image_dir + image_name)


        img = transform(img)
        style_img = transform(style_img)
        
        input = img.unsqueeze(0).to(device)
        style_input = style_img.unsqueeze(0).to(device)
        out,_= net_g(input,style_input)
        out_img = out.detach().squeeze(0).cpu()
        
        
        

        if not os.path.exists(os.path.join("result/{}/{}".format(opt.name_data,style_name), opt.name_data)):
            os.makedirs(os.path.join("result/{}/{}".format(opt.name_data,style_name), opt.name_data))
        print(image_name)
        save_img(out_img,w,h ,"result/{}/{}/{}".format(opt.name_data,style_name,image_name))
        
        style_img = load_img(style_test_image_dir+style_name)