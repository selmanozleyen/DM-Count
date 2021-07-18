import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn

M_EPS = 1e-16
class OT_Loss4(Module):
    def __init__(self, c_size, stride, norm_cood, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss4, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.kl = torch.nn.KLDivLoss(reduction='sum')
        self.cos = torch.nn.CosineSimilarity(dim=0)
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 # map to [-1, 1]
        self.output_size = self.cood.size(1)
        self.emptyC = torch.empty(size=(1, (c_size//stride) * (c_size//stride)),
        device=device,dtype=torch.float32)
        self.emptyC[0,:] = 2*c_size**2


    def forward(self, normed_density, unnormed_density, points, v_pred):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        lossv = torch.zeros([1]).to(self.device)
        lossu = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0 # wasserstain distance
        errs =0 
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
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
            else:
                dis = self.emptyC
                target_prob = (torch.ones([1])).to(self.device)
            source_prob = normed_density[idx][0].view([-1]).detach()
            
            v_init = v_pred[idx]
            K = torch.empty(dis.shape, dtype=dis.dtype,device=self.device)
            torch.div(dis, -self.reg, out=K)
            torch.exp(K, out=K)
            Kv = torch.matmul(K, v_init)
            u_init = torch.div(target_prob, Kv + M_EPS)
            # b_hat = torch.matmul(u_init, K) * v_init
            
            # use sinkhorn to solve OT, compute optimal beta.
            warm_start = {'v':v_init.detach(), 'u': u_init.detach(), 'K': K.detach()}

            P, log = sinkhorn(a=target_prob, b=source_prob,stopThr=1e-9,
            C=dis, reg=self.reg, maxIter=self.num_of_iter_in_ot, log=True,
            warm_start=warm_start)

            beta = log['beta'] # size is the same as source_prob: [#cood * #cood]
            u, v = log['u'], log['v']
            ot_obj_values += torch.sum(normed_density[idx] * beta.view([1, self.output_size, self.output_size]))
            # compute the gradient of OT loss to predicted density (unnormed_density).
            # im_grad = beta / source_count - < beta, source_density> / (source_count)^2
            source_density = unnormed_density[idx][0].view([-1]).detach()
            source_count = source_density.sum()
            im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta # size of [#cood * #cood]
            im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8) # size of 1
            im_grad = im_grad_1 - im_grad_2
            im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
            # lossw = torch.sum(dis * (u_init.reshape(-1, 1) * K * v_pred[idx].reshape(1, -1)))
            # lossw = lossw*self.wow
            # b_hat = torch.matmul(u_init, K) * v_pred[idx]

            errs+=log['err'][-1]
            # lossb = -torch.cosine_similarity(v_pred[idx],v,dim=0)
            # lossb += -torch.cosine_similarity(u_init,u,dim=0)
            # lossb = lossb*self.wob
            # lossb = self.kl_loss(torch.exp(v_pred[idx]),v) + self.kl_loss(torch.exp(u_pred),u)
            # lossb = torch.norm(v_pred[idx]-v,dim=0) + torch.norm(u_init-u,dim=0) 
            # lossv+= torch.sum(torch.square(source_prob-b_hat))
            v = v/(v.sum()+M_EPS)
            v_hat = v_pred[idx]
            lossv+= self.kl(torch.log(v_hat),v)
            lossu += -self.cos(u_init,u)
            # lossv += torch.sum(torch.abs(v_pred[idx]-v)) 
            # lossb = self.kl_loss(torch.exp(v_pred[idx]),v) + self.kl_loss(torch.exp(u_pred),u)
            # Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
            loss += torch.sum(unnormed_density[idx] * im_grad)
            wd += torch.sum(dis * P).item()
            # wd2 += lossw.item()
        return loss, lossv, lossu, wd, ot_obj_values, errs


