from __future__ import print_function

import argparse
import os
from datetime import datetime
from os import path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from data_loader import data_loader
from model import multires_model



from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='16', help='number of channels for 1st resolution')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=100, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('-alr', default=0.01, type=float, help=' alpha learning rate')
parser.add_argument('-am', default=0.9, type=float, help='alpha momentum')
parser.add_argument('-awd', default=0, type=float, help='weight decay')
parser.add_argument('--alpha_train_start', '-ats', type=int, default=0, help='epochs to start training alpha')
parser.add_argument('--max_scales', '-mx', type=int, default=4, help='number of scales to use')
parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
parser.add_argument('--number_train', '-nb', type=int, default=0, help='number of examles to traing on')
parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')

args = parser.parse_args()


def main():
    startTime = datetime.now()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    print('Test ', args.test_name)
    print('-----------------------')
    print('test parameters')
    print(args)
    print('-----------------------')
    if args.dataset == 'CIFAR10':
        ncat = 10 # number of categories
    elif args.dataset == 'TIN':
        ncat = 200

    net = multires_model(ncat=ncat, channels=args.channels, leng=args.leng, max_scales=args.max_scales) # create model
    net.cuda()

    '''
    complexity info is inaccurate since it does not count functions that are defined using nn.functional in MACs/FLOPS
    '''
    # print('Complexity info')
    # '''Calculation of number of parameters and MACs'''
    # input = torch.randn(1, 3, 32, 32)
    # input = input.cuda()
    # macs, params = profile(net, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('-----------------------------------')
    # print(macs, params)
    # print('----------------------------')
    # macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # #scale invariant
    # # for i in range(net.module.leng):
    # #     net.module.layer[i].block.gamma = args.gamma
    # #     a = [(i + 1) ** args.gamma for i in range(args.max_scales)]
    # #     net.module.layer[i].block.sia_multiplier = torch.FloatTensor(a).view(-1, 1, 1, 1, 1).cuda()
    #
    # print("param size = %fMB" %(count_parameters_in_MB(net)))

    print('-----------------------')
    print('Model:')
    print(net)

    criterion = nn.CrossEntropyLoss() # classification loss criterion
    criterion = criterion.cuda()

    weight_parameters, alpha_parameters = parameters(net) # get network and alpha parameters
    weight_optimizer = optim.SGD(weight_parameters, lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, args.epochs+1, eta_min=args.min_learning_rate) # scheduler for weight learning rate
    alpha_optimizer = optim.Adam(alpha_parameters, lr=args.alr, weight_decay=args.awd)

    save_dir = './checkpoint/ckpt_' + str(args.test_name) + '.t7' #checkpoint save directory
    if path.exists(save_dir): # load from checkpoint if it exists
        best_model, current_epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index\
            = load_checkpoint(save_dir, net, weight_optimizer, scheduler, alpha_optimizer)
    else: # initialize checkpoint parameters if not resuming training
        best_model, current_epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index\
            = initialize_save_parameters(net)

    # loading dataset
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/' # my default dataset directory

    train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent, args.batchsize,
                                                                                   args.number_train,
                                                                                   indices=index,
                                                                                   dataset_dir=dataset_dir,
                                                                                   workers=args.workers)

    # training
    for epoch in range(current_epoch+1, args.epochs+1):
        print('epoch ', epoch)
        print('net learning rate: ', weight_optimizer.param_groups[0]['lr']) # print learning rate
        # train
        train_loss, validation_loss, train_accuracy, validation_accuracy = train_valid(train_loader, validation_loader, net, weight_optimizer, alpha_optimizer, criterion, epoch)
        scheduler.step()
        print('Training time: ', datetime.now() - startTime) # print  time from the start of training
        if epoch % 5 == 0 or epoch == args.epochs: # test and save checkpoint every 5 epochs
            print('Testing...')
            test_loss, test_accuracy = test(test_loader, net)
            # record loss
            loss_progress['train'].append(train_loss)
            loss_progress['validation'].append(validation_loss)
            loss_progress['test'].append(test_loss)
            # record accuracy
            accuracy_progress['train'].append(train_accuracy)
            accuracy_progress['validation'].append(validation_accuracy)
            accuracy_progress['test'].append(test_accuracy)
            # record alpha
            alpha_progress.append(get_alpha(net))

            if test_accuracy > best_accuracy: # update best test accuracy
                print('--------------> Best accuracy')
                # record model with the best test accuracy
                best_model = {}
                for key in net.state_dict():
                    best_model[key] = net.state_dict()[key].clone().detach()
                best_accuracy = test_accuracy
                best_epoch = epoch
                best_alpha = get_alpha(net) # alpha of model with best test accuracy

            print('train accuracy: ', train_accuracy, ' ....... validation accuracy: ', validation_accuracy, ' ....... test accuracy: ', test_accuracy)
            print('best accuracy:', best_accuracy,' at epoch ', best_epoch)
            print('...........SAVING............')
            save_checkpoint(save_dir, net, best_model, weight_optimizer, scheduler, alpha_optimizer, epoch, loss_progress,
                            accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index) # save checkpoint
        print('Training time: ', datetime.now() - startTime) # print  time from the start of training


def initialize_save_parameters(model):
    epoch = 0
    index = 0
    best_epoch = 0
    best_model = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    alpha_progress = []
    best_alpha = get_alpha(model)
    return best_model, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index


def load_checkpoint(save_dir, model, weight_optimizer, scheduler, alpha_optimizer):
    print('Loading from checkpoint...')
    checkpoint = torch.load(save_dir)
    epoch = checkpoint['epoch']
    loss_progress = checkpoint['loss_progress']
    accuracy_progress = checkpoint['accuracy_progress']
    alpha_progress = checkpoint['alpha_progress']
    best_model = checkpoint['best_model']
    best_alpha = checkpoint['best_alpha']
    best_epoch = checkpoint['best_epoch']
    best_accuracy = checkpoint['best_accuracy']
    index = checkpoint['indices']

    model.load_state_dict(checkpoint['model'])
    weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
    return best_model, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index


def save_checkpoint(save_dir, model, best_model, weight_optimizer, scheduler, alpha_optimizer, epoch, loss_progress, accuracy_progress, alpha_progress, best_alpha, best_epoch, best_accuracy, index):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'indices': index,
        'best_epoch': best_epoch,
        'best_model': best_model,
        'loss_progress': loss_progress,
        'accuracy_progress': accuracy_progress,
        'alpha_progress': alpha_progress,
        'best_alpha': best_alpha,
        'best_accuracy': best_accuracy,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'alpha_optimizer': alpha_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, save_dir)


def calculate_accuracy(logits, target, cul_total=0, cul_prediction=0):
    _, test_predicted = logits.max(1)
    test_total = target.size(0)
    # correct_prediction = test_predicted.eq(target).sum().item()
    # cul_prediction += correct_prediction
    cul_total += test_total
    return cul_prediction, cul_total


def get_alpha( model):
    alpha = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad and 'alpha' in name:
            # alpha.append(param.cpu().detach())
    # alpha = (torch.stack(alpha, 0)).numpy()
    return alpha


def parameters(model):
    all_parameter_names = [x for x, y in model.named_parameters()]
    weight_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is -1]
    alpha_param_idx = [idx for idx, x in enumerate(all_parameter_names) if x.find('alpha') is not -1]
    all_parameters = [y for x, y in model.named_parameters()]
    sum_param = sum([np.prod(p.size()) for p in all_parameters])
    print('sum parameters: ', sum_param)
    weight_parameters = [all_parameters[idx] for idx in weight_param_idx]
    alpha_parameters = [all_parameters[idx] for idx in alpha_param_idx]

    return weight_parameters, alpha_parameters


def train_valid(train_queue, validation_queue, model, weight_optimizer, alpha_optimizer, criterion=nn.CrossEntropyLoss(), epoch=0):
    '''
    train and validate . training means training convolution filters on training set, validation means training alpha values on a validation set.
    '''
    model.train()
    train_loss = 0
    validation_loss = 0

    train_correct = 0
    validation_correct = 0

    train_total = 0
    validation_total = 0

    train_accuracy = 0
    validation_accuracy = 0
    validation_iterator = iter(validation_queue)
    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        # train
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        weight_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()
        weight_optimizer.step()
        # train_loss += train_minibatch_loss.detach().cpu().item()
        train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)
        if epoch >= args.alpha_train_start:# validation
            validation_inputs, validation_targets = next(iter(validation_iterator))
            validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
            alpha_optimizer.zero_grad()
            validation_outputs = model(validation_inputs)
            validation_minibatch_loss = criterion(validation_outputs, validation_targets)
            validation_minibatch_loss.backward()
            alpha_optimizer.step()
            # validation_loss += validation_minibatch_loss.detach().cpu().item()
            validation_correct, validation_total = calculate_accuracy(validation_outputs, validation_targets, validation_total, validation_correct)
    validation_accuracy = validation_correct / validation_total # accuracy over epoch
    # train_loss = train_loss / (batch_idx + 1) # average loss in epoch
    validation_loss = validation_loss / (batch_idx + 1) # average loss in epoch
    train_accuracy = train_correct / train_total # accuracy over epoch

    return train_loss, validation_loss, train_accuracy, validation_accuracy

def test(test_queue, model, criterion=nn.CrossEntropyLoss()): # test
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (test_inputs, test_targets) in enumerate(test_queue):
            test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
            test_outputs = model(test_inputs)
            test_minibatch_loss = criterion(test_outputs, test_targets)
            # test_loss += test_minibatch_loss.detach().cpu().item()
            test_correct, test_total = calculate_accuracy(test_outputs, test_targets, test_total, test_correct)
    test_loss = test_loss / (batch_idx + 1)
    return test_loss, test_correct/test_total

def count_parameters_in_MB(model): # count number of parameters
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6





if __name__ == '__main__':
  main()