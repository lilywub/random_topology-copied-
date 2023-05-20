#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print("Type if dataset is: ", type(train_dataset))
    print("Length of dataset is: ", len(train_dataset))

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    num_shared_layers = 1

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()


    print("Test 1 ---------------------------------------")
    for param_tensor in global_weights:
        print(param_tensor, "\t", global_weights[param_tensor].size())
    print(global_weights.keys())
    print("Test 1 ---------------------------------------")

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    countprint = 0
    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        w = [[],[],[],[],[]]
        print("m = ", m)
        print("idxs_users: ", idxs_users)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w[idx], loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w[idx]))
            local_weights.append(copy.deepcopy(w[idx]))
            local_losses.append(copy.deepcopy(loss))

        print("type of w: ", type(w))
        print("type of w[1]: ", type(w[1]))

        print("Test 2 ---------------------------------------")
        for param_tensor in w[1]:
            print(param_tensor, "\t", w[1][param_tensor].size())
        print(w[1].keys())
        #print(w[1])
        print(w[1]['fc3.bias'])
        print("Test 2 ---------------------------------------")

        print("Test 3 ------------------------------------------------------------")
        print(global_weights['fc3.bias'])
        temp = global_weights
        print("before::::::::::::::::")
        print(temp['fc3.bias'])
        temp['fc3.bias'] = w[1]['fc3.bias']
        print("After:::::::::::::::::")
        print(temp['fc3.bias'])

        print("Test 3 ------------------------------------------------------------")



        # update global weights
        global_weights = average_weights(local_weights)
        print("countprint = ", countprint)
        countprint = countprint + 1
        #print(type(global_weights)) # <class 'collections.OrderedDict'>
        #print(global_weights)

        # -------------------------------------------------- Sili
        print("Model's state_dict:")
        for param_tensor in global_weights:
            print(param_tensor, "\t", global_weights[param_tensor].size())
        print(global_weights.keys())
        print(global_weights['fc3.bias'])


        # update global weights
        global_model.load_state_dict(global_weights)

        print("Test for model size -----------------------------------------")
        param_size = 0
        for param in global_model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in global_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        print("Total size = ", param_size+buffer_size)
        print("Test for model size -----------------------------------------")

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '/home/siliwu/PhD/Federated-Learning-PyTorch-master/save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/siliwu/PhD/Federated-Learning-PyTorch-master/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/siliwu/PhD/Federated-Learning-PyTorch-master/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
