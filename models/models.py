import torch
from torch import nn
from models.pc_encoder import PointNetEncoder
from models.diffusor import DiffusionPoint, PointwiseNet, VarianceSchedule
from models.flow import build_latent_flow
from models.commonUtils import reparameterize_gaussian, gaussian_entropy, standard_normal_logprob, truncated_normal_, simulate_laser_scanner
from chamferdist import ChamferDistance
from evaluation.emd_module import emdModule

class DiffusionPointCloudModel(nn.Module):
    # Deterministic diffusion model encoder
    def __init__(self, args):
        super(DiffusionPointCloudModel, self).__init__()

        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim, bias=args.bias)
        self.flow = build_latent_flow(args)
        self.diffusion_reconstruction = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual, bias=args.bias),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T_reconstruction,
                mode=args.sched_mode
            )
        )

        self.diffusion_generation = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual, bias=args.bias),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T_generation,
                mode=args.sched_mode
            )
        )

        
        # Chamfer Distance and EMD metrics
        self.cd = chamfDist = ChamferDistance()
        self.emd = emdModule()

        self._initialize_weights()
    
    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        mean, sigma = self.encoder(x)
        return mean, sigma
    
    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion_reconstruction.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)
    
    def run_flow(self, x, mean, sigma, kl_weight):
        '''
            The goal of this function is to train a generative model for generating noisy output versions of a ground-truth shape. These outputs are 
            supposed to produce point clouds of objects that are the result of the process of scanning objects with a laser scanners.
            These point clouds could be used to train the classifoer model if necessary
            Args:
                x: original point cloud
                mean: fixed mean (i.e. no gradient flow from the flow model to update the encoder network so the code is always defined w.r.t the reconstruction component. From another perspective, maybe a more principled better way of implementing this would be to multiply the prior loss by a very small number like 0.01. Will try this later)
                sigma: variance for the encoded point cloud
                kl_weight: kl divergence weight for enforcing the degree to which the posterior distrubition is close to the prior
        '''
        batchSize = x.size(0)
        noisy_x = simulate_laser_scanner(x)

        z = reparameterize_gaussian(mean=mean, logvar=sigma)  # (B, F) # sample the posterior distribution
        
        # H[Q(z|X)]
        # entropy = gaussian_entropy(logvar=sigma)      # (B, )
        entropy = gaussian_entropy(logvar=sigma).view(x.size(0), 1)      # (B, 1) # my modification to match all of the dimension of all loss elements

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batchSize, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batchSize, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batchSize, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        neg_elbo = self.diffusion_generation.get_loss(noisy_x, z)
        neg_elbo = neg_elbo.mean(dim=1, keepdim=True) # my modification to match all of the dimension of all loss elements

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean() # maybe multiply this with 0.01?
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        return loss
    
    def sample(self, w, num_points, flexibility, truncate_std=None):
        batch_size = w.size(0)
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion_generation.sample(num_points, context=z, flexibility=flexibility)
        return samples

    def get_loss(self, x, kl_weight, test_time=False):
        mean, sigma = self.encode(x)
        recon_loss = self.diffusion_reconstruction.get_loss(x, mean)
        recon_loss = recon_loss.mean(dim=1, keepdim=True) # my modification to match all of the dimension of all loss elements
        generation_loss = self.run_flow(x, mean.clone().detach(), sigma, kl_weight)
        cd_cmd_prob = torch.rand(1).item()
        if cd_cmd_prob <= 0.02 and not test_time:
            reconstruction = self.decode(mean, x.size(1), flexibility=self.args.flexibility) # reconstructions of the input point clouds
            cd_emd = self.computeCD_EMD(x, reconstruction)
        # return recon_loss + generation_loss
        if not test_time:
            if cd_cmd_prob <= 0.02:
                return (recon_loss + generation_loss + cd_emd).mean()
            else:
                return (recon_loss + generation_loss).mean()
        else:
            return recon_loss.mean(), generation_loss.mean()
    
    def computeCD_EMD(self, x, reconstruction):
        cd = self.cd(reconstruction.cpu(), x.cpu(), bidirectional=True, batch_reduction=None, point_reduction=None)
        emd, _ = self.emd(reconstruction, x, 0.007, 150)
        return cd.mean(dim=1, keepdim=True).cuda() + emd.mean(dim=1, keepdim=True)
    
    
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        print('here2')
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        print('here1')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    print('here')
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)