#!/usr/bin/env python3
import os
import argparse

import torch
import torchvision.models as models

from reshape import reshape_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse_img_size(img_size_str):
    # รองรับ "96x48" หรือ "48,96" หรือ [48,96]
    if isinstance(img_size_str, (list, tuple)) and len(img_size_str) == 2:
        return int(img_size_str[0]), int(img_size_str[1])
    img_size_str = img_size_str.replace(",", "x").replace("X", "x")
    w, h = map(int, img_size_str.split("x"))
    return h, w   # torch ใช้ (C, H, W)

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth.tar', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")
parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")
parser.add_argument('--img-size', type=str, default='96x48', help="export image size, format WxH เช่น 96x48")
parser.add_argument('--no-activation', action='store_true', help="disable adding Softmax or Sigmoid layer to model (default is to add it)")

args = parser.parse_args()
print(args)

# format input model path
if args.model_dir:
    args.model_dir = os.path.expanduser(args.model_dir)
    args.input = os.path.join(args.model_dir, args.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('=> running on device ' + str(device))

# load the model checkpoint
print('=> loading checkpoint:  ' + args.input)
checkpoint = torch.load(args.input, map_location=device)
arch = checkpoint['arch']

# create the model architecture
print('=> using model:  ' + arch)
model = models.__dict__[arch](pretrained=True)

# reshape the model's output
model = reshape_model(model, arch, checkpoint['num_classes'])

# load the model weights
model.load_state_dict(checkpoint['state_dict'])

# add softmax layer
if not args.no_activation:
    if checkpoint.get('multi_label', False):
        print('=> adding nn.Sigmoid layer to multi-label model')
        model = torch.nn.Sequential(model, torch.nn.Sigmoid())
    else:
        print('=> adding nn.Softmax layer to model')
        model = torch.nn.Sequential(model, torch.nn.Softmax(1))

model.to(device)
model.eval()

print(model)

# ==== Main: กำหนด image size ====
if args.img_size:
    h, w = parse_img_size(args.img_size)
elif isinstance(checkpoint.get('resolution'), (tuple, list)):
    h, w = checkpoint['resolution']
else:
    h = w = int(checkpoint.get('resolution', 224))  # fallback

# create example image data
dummy_input = torch.ones((1, 3, h, w)).to(device)
print('=> input size:  {}x{}'.format(w, h))

# format output model path
if not args.output:
    args.output = arch + '.onnx'

if args.model_dir and args.output.find('/') == -1 and args.output.find('\\') == -1:
    args.output = os.path.join(args.model_dir, args.output)

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('=> exporting model to ONNX...')
torch.onnx.export(model, dummy_input, args.output, verbose=True, input_names=input_names, output_names=output_names)
print('=> model exported to:  {:s}'.format(args.output))