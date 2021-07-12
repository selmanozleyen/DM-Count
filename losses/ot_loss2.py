from numpy import mod
import torch
from torch import nn
from torch.nn import Module
from .bregman_pytorch import sinkhorn
import cupy
import ot
import torchvision.transforms
from pytorch_lightning.loggers.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from  matplotlib.figure import Figure


from .stochastic import solve_semi_dual_entropic
class OT_Loss(Module):
    def __init__(self, c_size, downsample_ratio, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % downsample_ratio == 0
        self.downsample_ratio = downsample_ratio
        self.crop_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        cupy.cuda.Device(0).use()
        self.method = 'asgd'
        self.itr = 0    
        assert self.crop_size % self.downsample_ratio == 0
        self.output_size = self.crop_size//self.downsample_ratio 

        self.cood = torch.arange(0, self.crop_size, step=self.downsample_ratio,device=self.device) + self.downsample_ratio / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 # map to [-1, 1]


    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1],dtype=torch.float32).to('cuda')
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                src_prob = normed_density[idx].reshape([-1]).detach()
                target_prob = torch.ones(len(im_points))/len(im_points)
                
                x = im_points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = im_points[:, 1].unsqueeze_(1)
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood # [#gt, #cood]
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood


                
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1)) # size of [#gt, #cood * #cood]

                dis = cupy.asarray(dis)
                target_prob = cupy.asarray(target_prob)
                src_prob = cupy.asarray(src_prob)


                pi, log= solve_semi_dual_entropic(
                    target_prob, src_prob, dis, method= self.method,
                    log=True,lr=None,numItermax=self.num_of_iter_in_ot,reg=self.reg
                )
                alpha,beta = log['alpha'], log['beta']
                w = self.crop_size // self.downsample_ratio
                
                src = normed_density[idx]
                

                beta = torch.as_tensor(beta,device='cuda')


                source_density = unnormed_density[idx].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta # size of [#cood * #cood]
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8) # size of 1
                im_grad = im_grad_1 - im_grad_2
                
                im_grad = im_grad.detach().view([1, w, w])

                loss = loss + (unnormed_density[idx]*im_grad).sum()


        return loss
