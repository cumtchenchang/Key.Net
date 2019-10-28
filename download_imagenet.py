import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cufflinks as cf
import torch
import torchvison

#imagenet_data = torchvision.datasets.ImageNet('/home/SENSETIME/chenchang1/data/ImageNet/')
imagenet_data = torchvision.datasets.ImageNet('/mnt/lustre/chenchang1/data/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
