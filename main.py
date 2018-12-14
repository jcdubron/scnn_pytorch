import math
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

# import sys
# sys.path.append(os.getcwd())
from python.model import *
from python.transforms import *
from python.dataset import *
from python.train import *
from python.test import *
from python.prob2lines import *

# parser
parser = argparse.ArgumentParser(description='PyTorch SCNN Model')
parser.add_argument('--train_data_dir', metavar='DIR', default='/home/dwt/scnn_pytorch',
                    help='path to train dataset (default: /home/dwt/scnn_pytorch)')
parser.add_argument('--eval_data_dir', metavar='DIR', default='/home/dwt/scnn_pytorch',
                    help='path to eval dataset (default: /home/dwt/scnn_pytorch)')
parser.add_argument('--test_data_dir', metavar='DIR', default=None,
                    help='path to test dataset')
parser.add_argument('--train_list_file', metavar='DIR', default='train.txt',
                    help='train list file (default: train.txt)')
parser.add_argument('--eval_list_file', metavar='DIR', default='eval.txt',
                    help='eval list file (default: eval.txt)')
parser.add_argument('--test_list_file', metavar='DIR', default='test.txt',
                    help='eval list file (default: test.txt)')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size for training (default: 4)')
parser.add_argument('--epoches', type=int, default=10, metavar='N',
                    help='number of epoches to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
# parser.add_argument('--gpu', action='store_true', default=False,
#                     help='GPU training')
parser.add_argument('--gpu', metavar='N', type=int, nargs='+', default=-1,
                    help='GPU ids')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--checkpoint', metavar='DIR', default=None,
                    help='use pre-trained model')
parser.add_argument('--weights', metavar='DIR', default=None,
                    help='use finetuned model')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--snapshot_interval', type=int, default=1, metavar='N',
                    help='how many epoches to wait before saving snapshot (default: 2)')
parser.add_argument('--snapshot_prefix', type=str, default='./snapshot/model', metavar='PATH',
                    help='snapshot prefix (default: ./snapshot/model)')
parser.add_argument('--tensorboard', type=str, default='log', metavar='PATH',
                    help='tensorboard log path  (default: log)')
args = parser.parse_args()

# tensorboardX
writer = SummaryWriter(args.tensorboard)

# random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# cuda and seed
use_cuda = args.gpu[0]>=0 and torch.cuda.is_available()
device = torch.device('cuda:{0}'.format(args.gpu[0]) if use_cuda else 'cpu')
torch.manual_seed(args.seed)
if use_cuda:
    print('Use Device: GPU', args.gpu)
else:
    print('Use Device: CPU')

# model and scheduler
model = SCNN().to(device)
# if len(args.gpu) > 1:
#     model = torch.nn.DataParallel(model, device_ids=args.gpu)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: math.pow(1-epoch/90000, 0.9))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)  # ExponentialLR(optimizer, gamma=0.9)
epoch_start = 1

# continue training from checkpoint
if args.checkpoint is not None:
    assert os.path.isfile(args.checkpoint)
    print('Start loading checkpoint.')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    for k,v in checkpoint['model_state_dict'].items():
        print('load weights', k, v.shape)
    if 'epoch' in checkpoint:
        epoch_start = checkpoint['epoch']
    print('Loading checkpoint done.')
# finetune training with weights
elif args.weights is not None:
    assert os.path.isfile(args.weights)
    print('Start loading weights.')
    model_dict = model.state_dict()
    for k,v in model_dict.items():
        print('original model', k, v.shape)
    weights = torch.load(args.weights)
    weights = {k: v for k, v in weights.items() if k in model.state_dict()}
    for k,v in weights.items():
        print('load weights', k, v.shape)
    model_dict.update(weights)
    for k,v in model_dict.items():
        print('After load', k, v.shape)
    model.load_state_dict(model_dict)
    print('Loading weights done.')


# for sample in train_dataset:
#     print(sample)
# for idx, sample in enumerate(train_loader):
#     print(idx, sample['image'].size(), sample['probmap'].size())
#     input()

if args.test_data_dir is not None:
    print('Start dataset loading initialization.')
    # test_dataset = LaneDataset(img_dir=args.train_data_dir, prob_dir=args.train_data_dir+'_labelmap',
    #                            list_file=args.train_list_file, tag=True,
    #                            transform=transforms.Compose([SampleResize((800, 288)),
    #                                                          SampleToTensor(),
    #                                                          SampleNormalize(mean=[0.3598, 0.3653, 0.3662],
    #                                                                           std=[0.2573, 0.2663, 0.2756])]))
    test_dataset = TestLaneDataset(img_dir=args.test_data_dir, list_file=args.test_list_file,
                                   transform=transforms.Compose([TestSampleResize((800, 288)),
                                                                 TestSampleToTensor(),
                                                                 TestSampleNormalize(mean=[0.3598, 0.3653, 0.3662],
                                                                                     std=[0.2573, 0.2663, 0.2756])]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)
    print('Dataset loading initialization done.')
    test(model, device, test_loader, args)

else:
    # mean=[0.37042467, 0.36758537, 0.3584016]
    print('Start dataset loading initialization.')
    train_dataset = LaneDataset(img_dir=args.train_data_dir, prob_dir=args.train_data_dir+'_labelmap',
                                list_file=args.train_list_file, tag=False,
                                transform=transforms.Compose([SampleResize((800, 288)),
                                                              SampleToTensor(),
                                                              SampleNormalize(mean=[0.3598, 0.3653, 0.3662],
                                                                              std=[0.2573, 0.2663, 0.2756])]))
    eval_dataset = LaneDataset(img_dir=args.eval_data_dir, prob_dir=args.eval_data_dir+'_labelmap',
                               list_file=args.eval_list_file, tag=True,
                               transform=transforms.Compose([SampleResize((800, 288)),
                                                             SampleToTensor(),
                                                             SampleNormalize(mean=[0.3598, 0.3653, 0.3662],
                                                                             std=[0.2573, 0.2663, 0.2756])]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)
    print('Dataset loading initialization done.')

    train(model, writer, args, device, train_loader, eval_loader, scheduler, epoch_start, loss_weight=(1, 0.1))
