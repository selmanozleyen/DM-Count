from numpy import mod
import torch
from torch import nn
from torch.nn import Module
from .smooth import *

class OT_Loss2(Module):
    def __init__(self, c_size, downsample_ratio, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss2, self).__init__()
        assert c_size % downsample_ratio == 0
        self.downsample_ratio = downsample_ratio
        self.crop_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.method = 'asgd'
        self.itr = 0    
        assert self.crop_size % self.downsample_ratio == 0
        self.output_size = self.crop_size//self.downsample_ratio 

        self.cood = torch.arange(0, self.crop_size, step=self.downsample_ratio,device=self.device) + self.downsample_ratio / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 # map to [-1, 1]
        self.cood_sqr = self.cood * self.cood

        self.emptyC = torch.empty(size=((c_size//self.downsample_ratio) * (c_size//self.downsample_ratio), 1)
        ,dtype=torch.float32,device='cuda')
        self.emptyC[0,:] = 1*c_size**2

    def forward(self, normed_density, unnormed_density, points, alpha_pred):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1],dtype=torch.float32,device='cuda')
        ot_obj_values = torch.zeros([1],dtype=torch.float32,device='cuda')
        wd = 0 # wasserstain distance
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1
                target_prob = torch.ones(len(im_points),device='cuda')/len(im_points)
                
                x = im_points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = im_points[:, 1].unsqueeze_(1)
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood_sqr # [#gt, #cood]
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood_sqr
                
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1)) # size of [#gt, #cood * #cood]

                dis = dis.T
            else:
                dis = self.emptyC
                target_prob = torch.ones([1],device='cuda')

            alpha_hat = alpha_pred[idx]
            src_prob = normed_density[idx].reshape([-1]).detach()


            # beta = averaged_sgd_entropic_transport(
            #     target_prob, src_prob, dis,
            #     lr=None,numItermax=self.num_of_iter_in_ot,reg=self.reg
            # )

            # src_prob, target_prob, dis, alpha_hat = torch.as_tensor(src_prob,device='cuda'), torch.as_tensor(target_prob,device='cuda'), torch.as_tensor(dis,device='cuda'), torch.as_tensor(alpha_hat,device='cuda')
            regul = SquaredL2(gamma=self.reg)
            obj, dalpha= semi_dual_obj_grad2(alpha=alpha_hat,a=src_prob,b=target_prob,C=dis,regul=regul)
            # pi, log = solve_semi_dual_entropic(a=target_prob,b=src_prob,M=dis,
            # reg=self.reg,numItermax=self.num_of_iter_in_ot,lr=None,log=True,method='asgd')
            #print(obj)
            w = self.crop_size // self.downsample_ratio
            
            
            alpha = alpha_pred[idx].detach()
            ot_obj_values += obj

            source_density = unnormed_density[idx].view([-1]).detach()
            source_count = source_density.sum()
            im_grad_1 = (source_count) / (source_count * source_count + 1e-16) * alpha # size of [#cood * #cood]
            im_grad_2 = (source_density * alpha).sum() / (source_count * source_count + 1e-16) # size of 1
            im_grad = im_grad_1 - im_grad_2
            
            im_grad = im_grad.detach().view([1, w, w])

            loss += torch.sum(unnormed_density[idx]*im_grad) + 0.1*(torch.sum(alpha_pred[idx]*(dalpha.detach())) - obj)
            # wd += cupy.sum(dis * pi).item()


        return loss, wd, ot_obj_values
