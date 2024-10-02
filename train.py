import argparse, os, random, gc
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scheduler import AnnealingStepLR
from models.models import DiffusionPointCloudModel
from commonUtils import mkdirs, fileExist
from utils.dataset import ShapeNetCore
from utils.transform import RandomRotate
from utils.misc import str_list
from evaluation.evaluation import EMD_CD
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Diffusion Model')
    # Model parameters
    parser.add_argument('--latent_dim', type=int, help='dimension of encoded point cloud (default: 256)', default=64)
    parser.add_argument('--num_steps', type=int, help='number of steps for the diffusion process (default: 200)', default=200)
    parser.add_argument('--beta_1', type=float, help='minimum value for beta for variance scheduling (default: 1e-4)', default=1e-4)
    parser.add_argument('--beta_T_reconstruction', type=float, help='maximum value for beta for variance scheduling for reconstruction (default: 0.05)', default=0.05)
    parser.add_argument('--beta_T_generation', type=float, help='maximum value for beta for variance scheduling for generating noisy samples (default: 0.02)', default=0.02)
    parser.add_argument('--sched_mode', type=str, help='variance scheduling strategy (default: linear)', default='linear')
    parser.add_argument('--flexibility', type=float, help='sets variability in generated output point clouds (default: 0.0)', default=0.0)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--truncate_std', type=float, default=2.0)
    parser.add_argument('--latent_flow_depth', type=int, default=14)
    parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bias', type=eval, help='Set bias when constructing neural network layers (default: True)', choices=[True, False], default=False)

    # Training parameters
    parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch (default: 64)')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate (default: 2e-4)')
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--resume_training', type=eval, help='set to true to start training from scratch and ignore previously-saved models, if any. So make sure not to overwrite the saved models if you think you would still need those models', \
                        choices=[False, True], default=False)
    parser.add_argument('--seed', type=int, help='random seed (set to -1 to oes not set the random seed)', default=8713)

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ShapeNet', help='dataset (dafault: ShapeNet)')
    parser.add_argument('--train_data_dir', type=str, help='location of training set data', \
                        default="data/shapenet.hdf5")
    parser.add_argument('--test_data_dir', type=str, help='location of test set data', \
                        default="data/shapenet.hdf5")
    parser.add_argument('--rotate', type=eval, help='rotation of objects along an axis', default=False, choices=[True, False])
    parser.add_argument('--categories', type=str_list, default=['all'], choices=['all', 'airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'bottle', 'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap', 'car', 'chair', 'clock', \
                    'dishwasher', 'monitor', 'table', 'telephone', 'tin_can', 'tower', 'train', 'keyboard', 'earphone', 'faucet', 'file', 'guitar', \
                    'helmet', 'jar', 'guitar', 'helmet', 'jar', 'knife', 'lamp', 'laptop', 'speaker', 'mailbox', 'microphone', 'piano', 'pillow', \
                    'pistol', 'pot', 'printer', 'pistol', 'rifle', 'rocket', 'skateboard', 'sofa', 'stove', 'vessel', 'washer', 'cellphone', \
                    'birdhouse', 'bookshelf'])
    
    # Misc
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='logs')
    parser.add_argument('--log_dir', type=str, help='log directory (default: '')', default='')
    parser.add_argument('--log_dir_notes', type=str, help='additional notes to be concatenated to the end of log_dir', default='')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=1600)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=40000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--debug', type=eval, help='Set to true when debugging so that nothing gets saved on disk (i.e. no model and no tensorboard log file)', \
                        choices=[False, True], default=[False])
    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark_limit = 0

    '''
        Uncomment the followings for more deterministic behavior and debugging 
    '''
    # torch.use_deterministic_algorithms(mode=True)
    # print(torch.are_deterministic_algorithms_enabled)
    # torch.autograd.set_detect_anomaly(True)

    # Seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    log_dir += '/' + args.dataset
    log_dir += '-{0:s}'.format(args.log_dir_notes) if args.log_dir_notes != '' else ''
    if args.debug:
        print(log_dir)
        mkdirs(log_dir)
        mkdirs(os.path.join(log_dir, 'models'))
        mkdirs(os.path.join(log_dir, 'stats'))

        # TensorBoardX
        writer = SummaryWriter(log_dir=os.path.join(log_dir,'stats'), flush_secs=25)

    # Number of scenes over which each weight update is computed
    B = args.batch_size
    
    # Maximum number of training steps
    S_max = args.gradient_steps

    # Mesh visualization parameters
    point_size_config = {
    'material': {
        'cls': 'PointsMaterial',
        'size': 0.05
        }
    }

    # Define model
    lastModelStatePath = None
    startGradStepTrain = 0
    startGradStepTest = 0
    if args.resume_training:
        for t in range(S_max, -save_interval_num, -save_interval_num):
            if fileExist(log_dir + "/models/model_optim-{}.pt".format(t)):
                lastModelStatePath = log_dir + "/models/model_optim-{}.pt".format(t)
                startGradStepTrain = t-1
                startGradStepTest = ((startGradStepTrain+1) // log_interval_num-(((startGradStepTrain+1) // log_interval_num)%B))
                startGradStepTrain = (startGradStepTrain+1-((startGradStepTrain+1)%B))
                break
        checkPoint = torch.load(lastModelStatePath)
        args = checkPoint['args']
        
    model = DiffusionPointCloudModel(args=args).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print('number of model parameters are', num_params)
    # exit()
    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001, amsgrad=True)
    scheduler = AnnealingStepLR(optimizer, mu_i=args.lr, mu_f=1e-5, n=1.2e5)
    if lastModelStatePath is not None:
        model.load_state_dict(checkPoint['model_state_dict'])
        optimizer.load_state_dict(checkPoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkPoint['scheduler_state_dict'])
    checkPoint = None
    del checkPoint

    # Dataloader
    testBatchNumSqrt = 3
    transform = None
    if args.rotate:
        transform = RandomRotate(180, ['pointcloud'], axis=1)
        print('Transform: %s' % repr(transform))
    trainDataset = ShapeNetCore(path=args.train_data_dir, cates=args.categories, split='train', scale_mode='shape_unit', transform=transform)
    testDataset = ShapeNetCore(path=args.test_data_dir, cates=args.categories, split='val', scale_mode='shape_unit', transform=transform)
    kwargs = {'num_workers':num_workers, 'pin_memory': True, 'prefetch_factor': 8} if torch.cuda.is_available() else {}
    trainDataLoader = DataLoader(trainDataset, batch_size=B, shuffle=True, **kwargs)
    testDataLoader = DataLoader(testDataset, batch_size=testBatchNumSqrt**2, shuffle=True, **kwargs)

    train_iter = iter(trainDataLoader)
    test_iter = iter(testDataLoader)
    testDataBatch =  next(test_iter)
    '''
        Run tests on a fixed batch of test samples
    '''
    x_test =testDataBatch['pointcloud'].to(device)
    shift = testDataBatch['shift'].to(device)
    scale = testDataBatch['scale'].to(device)

    losses = []
    gradnorms = []
    loss_train_total = 0
    write_update_iters = 30
    for t in tqdm(range(S_max)):

        try:
            x = next(train_iter)['pointcloud']
        except StopIteration:
            train_iter = iter(trainDataLoader)
            x = next(train_iter)['pointcloud']
        x = x.to(device) # Data on the GPU

        # Perform forward/reverse diffusion for reconstruction and generate noisy point clouds
        loss = model.get_loss(x, args.kl_weight)

        # Store the loss for visualization
        losses.append(loss.item())
        # if t % 50 == 0:
        #     print(t, torch.Tensor(losses[:-50]).mean().item())
        loss_train_total += loss.item()
        
        if args.debug and t % write_update_iters == 0:
            writer.add_scalar('train_loss', loss_train_total/write_update_iters, t)
            loss_train_total = 0

        '''
            Test the model during training
        '''
        if args.debug and t % log_interval_num == 0:
            model.eval()
            with torch.no_grad():
                code_mean, sigma = model.encode(x_test)
                recon_loss, generation_loss = model.get_loss(x_test, args.kl_weight, test_time=True)
                reconstruction = model.decode(code_mean, x_test.size(1), flexibility=args.flexibility) # reconstructions of the input point clouds
                generated = model.sample(code_mean, args.sample_num_points, flexibility=args.flexibility) # supposedly noisy, laser scan-like, point clouds conditioned on an input point cloud (or just sample z from the prior to get a random noisy shape)
                cd, emd = EMD_CD(reconstruction, x_test, reduced=True)
                writer.add_scalar('test/recon', recon_loss.item(), t)
                writer.add_scalar('test/gen', generation_loss.item(), t)
                writer.add_scalar('test/cd', cd.item(), t)
                writer.add_scalar('test/emd', emd.item(), t)
                # print('cd', cd.item(), 'emd', emd.item())

                if t == 0:
                    writer.add_mesh('ground_truth', x_test, config_dict=point_size_config)

                writer.add_mesh('ground_truth_1recon', reconstruction, config_dict=point_size_config)
                writer.add_mesh('ground_truth_generated', generated, config_dict=point_size_config)
                writer.flush()

            model.train()
        
        if t >= 40000 and t % save_interval_num == 0:
            # torch.save(model.state_dict(), log_dir + "/models/model-{}.pt".format(t))
            with torch.no_grad():
                torch.save({'args': args, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'weight_decay': optimizer.defaults['weight_decay'], 'lr': optimizer.defaults['lr']}, log_dir + "/models/model_optim-{}.pt".format(t))

        
        # Backprop and update the params:
        loss.backward()
        if args.debug and t % write_update_iters == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('grad/' + name, param.grad, t)
        orignorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 150.0) # clip gradients to lower the chance of exploding gradients problem. TODO: I should try a value of ~10
        # print(orignorm.item())
        optimizer.step()
        optimizer.zero_grad()

        # Update optimizer state
        scheduler.step()

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(trainDataLoader):])/len(trainDataLoader)
    print(f'Finished epoch {S_max}. Average loss for this epoch: {avg_loss:05f}')