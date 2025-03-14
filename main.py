import argparse
import os
import torch.nn.parallel
import torch.optim
#from models.DTA_SNN import *
#from models.SEW_ResNet import * ### 32-7B-ResNet
#from models.MS_ResNet import * ### ResNet-18-32input
from data.augmentations import *
from data.loaders import build_cifar
from utils.utils import *
from spikingjelly.datasets import cifar10_dvs, dvs128_gesture
from spikingjelly.clock_driven import functional
import numpy as np
import torch.nn as nn

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Dual Temporal-channel-wise Attention for Spiking Neural Networks')


parser.add_argument('--DTA',
                    default=True,
                    type=bool,
                    help='using DTA')
parser.add_argument('--DS',
                    default='dvs_gesture',
                    type=str,
                    help='cifar10, cifar100, dvs_gesture, dvs_cifar10')
parser.add_argument('--model',
                    default='MSResNet',
                    type=str,
                    help='backbone network: SEWResNet, SpikingResNet, MSResNet')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number')

parser.add_argument('--batch_size',
                    default=16,  
                    type=int,
                    metavar='N')

parser.add_argument('--learning_rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')

parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training')

parser.add_argument('--time_step',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time steps (default: 6)')
parser.add_argument('--workers',
                    default=20,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dvs_aug',
                    default=False,
                    type=bool,
                    metavar='N',
                    help='augmentation for dvs data')
parser.add_argument('--weight_decay',
                    default=0.00015, 
                    type=float,
                    metavar='N',
                    help='weight_decay')
parser.add_argument('--beta', 
                    default=1.0, 
                    type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', 
                    default=0.5, 
                    type=float,
                    help='cutmix probability')
parser.add_argument('--mixup', 
                    type=float, 
                    default=0.5,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-prob', 
                    type=float, 
                    default=0.5,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', 
                    type=float, 
                    default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', 
                    type=str, 
                    default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', 
                    type=float, 
                    default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--lambda_cov', 
                    type=float, 
                    default=0.1, 
                    help='Coefficient for auxiliary loss (if any).')
args = parser.parse_args()



def train(model, device, train_loader, criterion, optimizer, epoch, dvs_aug, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    r = np.random.rand(1)

    for  i,(images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.DS == 'dvs_cifar10': 
            #print('dvs_cifar10')
            images, target = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            images = images.float()  
            N,T,C,H,W = images.shape
            ###resize###
            train_aug = get_train_aug(args)
            images = torch.stack([(train_aug(images[i])) for i in range(N)])
            ############
            
            # #밑에 s 붙이는거 기억###### aug 안하려면 주석 ##############
            # trival_aug = get_trival_aug() #Autoaug  
            # mixup_fn = get_mixup_fn(args, num_CLS) #mixup
            # images = torch.stack([(trival_aug(images[i])) for i in range(N)])
            # images, target = mixup_fn(images, target)
            # targets = target.argmax(dim=-1)
            # ############### aug 안하려면 주석 ##############

            # outputs = model(images)
            # loss = criterion(outputs, target) #aug 사용 시 target뒤에 s 추가
            
            #for moe v2
            outputs, load_balance_loss = model(images)  # MoE 구조이므로 보조 손실 반환
            loss = criterion(outputs, target) + args.lambda_cov * load_balance_loss

        elif args.DS == 'dvs_gesture': 
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            images = images.float()
            N, T, C, H, W = images.shape
            
            ###resize###
            train_aug = get_train_aug(args)
            images = torch.stack([(train_aug(images[i])) for i in range(N)])
            ############
            
            if dvs_aug is not None:
                images = dvs_aug(images)
            outputs = model(images)  # [batch_size, num_classes] = [16, 11]
            loss = criterion(outputs, targets)  # mean_out 제거

        running_loss += loss.item()
        loss.backward()  # loss.mean() 제거, 이미 스칼라값이므로 .mean() 불필요
        optimizer.step()
        functional.reset_net(model)

        total += float(targets.size(0))
        _, predicted = outputs.cpu().max(1)  # mean_out 대신 outputs 사용
        correct += float(predicted.eq(targets.cpu()).sum().item())

    return running_loss/M, 100 * correct / total

    #######    for origin ms_resnet    ######## 
    #     elif args.DS == 'dvs_gesture':
    #         images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
    #         images = images.float()  
    #         N,T,C,H,W = images.shape
    #         #train_aug = get_train_aug(args)
    #         #images = torch.stack([(train_aug(images[i])) for i in range(N)])
    #         outputs = model(images)
    #         mean_out = outputs.mean(1)
    #         loss = criterion(mean_out, targets)

    #     running_loss += loss.item()
    #     loss.mean().backward()
    #     optimizer.step()
    #     total += float(targets.size(0))
    #     _, predicted = mean_out.cpu().max(1)
    #     correct += float(predicted.eq(targets.cpu()).sum().item())

    # return running_loss/M, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if args.DS == 'dvs_cifar10': 
            #print('dvs_cifar10')
            inputs = inputs.to(device, non_blocking=True)
            target = targets.to(device, non_blocking=True)
            ###resize###
            N,T,C,H,W = inputs.shape
            test_aug = get_test_aug(args)
            inputs = torch.stack([(test_aug(inputs[i])) for i in range(N)])
            ############
            inputs = inputs.float()
            #outputs = model(inputs)
            
            #for moe v2
            outputs, _ = model(inputs)
            
            #mean_out = outputs.mean(1)
            # _, predicted = mean_out.cpu().max(1)
            # total += float(target.size(0))
            # correct += float(predicted.eq(target.cpu()).sum().item())

        elif args.DS == 'dvs_gesture': 
            # print('dvs_gesture')
            inputs = inputs.to(device, non_blocking=True)
            target = targets.to(device, non_blocking=True)
            ###resize###
            N,T,C,H,W = inputs.shape
            test_aug = get_test_aug(args) 
            inputs = torch.stack([(test_aug(inputs[i])) for i in range(N)])
            ############
            inputs = inputs.float()
            outputs = model(inputs)  # [batch_size, num_classes] = [N, 11]

        functional.reset_net(model)
        _, predicted = outputs.cpu().max(1)  # mean_out 제거
        total += float(target.size(0))
        correct += float(predicted.eq(target.cpu()).sum().item())

    final_acc = 100 * correct / total
    return final_acc

    # #######    for origin ms_resnet    ######## 
    #     elif args.DS == 'dvs_gesture':
    #         #print('dvs_gesture')
    #         inputs = inputs.to(device, non_blocking=True)
    #         target = targets.to(device, non_blocking=True)
    #         # N,T,C,H,W = inputs.shape
    #         #test_aug = get_test_aug(args)
    #         #inputs = torch.stack([(test_aug(inputs[i])) for i in range(N)])
    #         inputs = inputs.float()
    #         outputs = model(inputs)
    #         mean_out = outputs.mean(1)
    #         _, predicted = mean_out.cpu().max(1)
    #         total += float(target.size(0))
    #         correct += float(predicted.eq(target.cpu()).sum().item())
    # final_acc = 100 * correct / total
    # return final_acc

if __name__ == '__main__':

    seed_all(args.seed)

    T = args.time_step

    if args.DS == 'dvs_gesture':
        print('dvs_gesture')
        num_CLS = 11
        save_ds_name = 'dvs_gesture'
        #trm = DVSTransform(5, 30, 16, 0.3)
        train_dataset = dvs128_gesture.DVS128Gesture(root='./dataset/DVS_Gesture', train=True, data_type='frame', frames_number=args.time_step,
                                                     split_by='number')#, transform=trm)
        val_dataset = dvs128_gesture.DVS128Gesture(root='./dataset/DVS_Gesture', train=False, data_type='frame', frames_number=args.time_step,
                                                    split_by='number')#, transform=trm)


    elif args.DS == 'dvs_cifar10':
        print('dvs_cifar10')
        num_CLS = 10
        save_ds_name = 'DVSCIFAR10'
        origin_set = cifar10_dvs.CIFAR10DVS(root="./dataset/DVS_CIFAR10", data_type='frame', frames_number=args.time_step, split_by='number')
        train_dataset, val_dataset = split_to_train_test_set(0.9, origin_set, 10)    
    
    # 32-7B-ResNet
    if args.DS == 'dvs_gesture':
        print('dvs_gesture')
        if args.model == 'MSResNet':
            #from models.gesture import *
            from models.MS_ResNet import *
            DP_model = dta_msresnet(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True)
        elif args.model == 'SEWResNet':
            #from models.gesture import *
            #from models.MS_ResNet import *
            from models.SEW_ResNet_moe import *
            DP_model = GA_sewresnet_moe(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True)
        elif args.model == 'SpikingResNet':
            from models.SEW_ResNet import *
            DP_model = SpikingResNet()
    
    # # ResNet-18-32input
    # if args.DS == 'dvs_gesture':
    #     print('dvs_gesture')
    #     if args.model == 'MSResNet':
    #         DP_model = dta_msresnet18(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True) 
    #     elif args.model == 'SEWResNet':
    #         DP_model = dta_sewresnet18(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True) 
    #     elif args.model == 'SpikingResNet':
    #         DP_model = SpikingResNet()

    elif args.DS == 'dvs_cifar10':
        from models.SEW_ResNet_moe import *
        from models.DTA_SNN import *
        print('dvs_cifar10')
        print(num_CLS)
        #DP_model = sewresnet_cifar(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True) 
        #DP_model = sewresnet_ga_cifar(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True)
        DP_model = sewresnet_ga_moe_cifar_v2(num_classes=num_CLS, time_step=args.time_step, DTA_ON=args.DTA, dvs=True)
        
        
    DP_model = torch.nn.DataParallel(DP_model).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.SGD(DP_model.parameters(),lr=args.learning_rate,momentum=0.9,weight_decay=args.weight_decay) #큰batch, 긴layer작업에서 선호
    optimizer = torch.optim.Adam(DP_model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay) #빠른수렴
    #optimizer = torch.optim.AdamW(DP_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #빠른수렴 + L2정규화
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1) #step_size는 일반적으로 총 epoch의 1/3 또는 1/2

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    
    logger = get_logger(f'{save_ds_name}-{args.model}-resize64x64_GA_MOE_fix1_tdfc_layer678_dropout1_15_2_25_experts_shared_fc_new_moe_v2_top1_noscale_noaug-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}.log')
    logger.info('start training!')

    # if args.dvs_aug == True:
    #     dvs_aug = Cutout(n_holes=1, length=16)
    # else:
    #     dvs_aug = None
    dvs_aug = None

    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        loss, acc = train(DP_model, device, train_loader, criterion, optimizer, epoch, dvs_aug, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch +1, args.epochs, loss, acc ))
        scheduler.step()
        facc = test(DP_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch+1, args.epochs, facc ))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(DP_model.module.state_dict(), f'{save_ds_name}-{args.model}-resize64x64_GA_MOE_fix1_tdfc_layer678_dropout1_15_2_25_experts_shared_fc_new_moe_v2_top1_noscale_noaug-S{args.seed}-B{args.batch_size}-T{args.time_step}-E{args.epochs}-LR{args.learning_rate}.pth.tar')

        logger.info('Epoch:[{}/{}]\t Best Test acc={:.3f}'.format(best_epoch, args.epochs, best_acc ))
        logger.info('\n')

