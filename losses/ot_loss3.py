import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn
import mlflow


class OT_Loss3(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0, wob = 1):
        super(OT_Loss3, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.wob = wob
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        mlflow.log_param('wob',wob)
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 # map to [-1, 1]
        self.output_size = self.cood.size(1)


    def forward(self, normed_density, unnormed_density, points, v_pred):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0 # wasserstain distance
        bd = 0
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                if self.norm_cood:
                    im_points = im_points / self.c_size * 2 - 1 # map to [-1, 1]
                x = im_points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = im_points[:, 1].unsqueeze_(1)
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood # [#gt, #cood]
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
                y_dis.unsqueeze_(2)
                x_dis.unsqueeze_(1)
                dis = y_dis + x_dis
                dis = dis.view((dis.size(0), -1)) # size of [#gt, #cood * #cood]

                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
                # use sinkhorn to solve OT, compute optimal beta.
                warm_start = {'v':v_pred[idx].detach()}
                P, log = sinkhorn(a=target_prob, b=source_prob, 
                C=dis, reg=self.reg, maxIter=self.num_of_iter_in_ot, log=True,
                warm_start=warm_start)
                _, beta = log['alpha'], log['beta'] # size is the same as source_prob: [#cood * #cood]
                u, v, K = log['u'], log['v'],log['K']
                Kv = torch.matmul(K, v_pred[idx])
                u_pred = torch.div(target_prob, Kv + 1e-16)
                ot_obj_values += torch.sum(normed_density[idx] * beta.view([1, self.output_size, self.output_size]))
                # compute the gradient of OT loss to predicted density (unnormed_density).
                # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta # size of [#cood * #cood]
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8) # size of 1
                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
                
                lossb = torch.mean(1-torch.cosine_similarity(v_pred[idx],v,dim=0))
                lossb += torch.mean(1-torch.cosine_similarity(u_pred,u,dim=0))
                # lossb = self.kl_loss(torch.exp(v_pred[idx]),v) + self.kl_loss(torch.exp(u_pred),u)
                
                # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
                loss += torch.sum(unnormed_density[idx] * im_grad) + lossb*self.wob
                bd += lossb.detach().cpu().item()
                wd += torch.sum(dis * P).item()

        mlflow.log_metric(
            'lossb',bd
        )
        return loss, wd, ot_obj_values


