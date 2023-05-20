#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# python3 src/decentralised_federated_main_Sili.py --model=mlp --dataset=mnist --epochs=500 --num_users=10 --frac=1 --iid=0

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
np.set_printoptions(threshold=sys.maxsize)

import torch
import torchvision
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

    np.random.seed(args.seed)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    print("Type if dataset is: ", type(train_dataset))
    print("Length of dataset is: ", len(train_dataset))
    # print("--------", len(train_dataset.targets))
    # print("--------", train_dataset.targets)
    # print("--------", train_dataset.classes)
    # print("--------", test_dataset.train_labels)
    # for i in user_groups[9]:
    #     print((((train_dataset.targets)[int(i)])))
    # print(test_dataset.targets)
    # breakpoint()

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)

        elif args.dataset == 'emnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
            # global_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            # global_model = torchvision.models.resnet18(pretrained=True)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        # MNIST image size = 28*28
        for x in img_size:
            len_in *= x
            # len_in = 28*28=784, dim_hidden=64 by default
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    num_shared_layers = 1

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    #print(type(global_model.layer[0].weight))

    # copy weights
    global_weights = global_model.state_dict()
    temp_model = global_model
    temp_model.load_state_dict(global_weights)

    local_w = []
    for n in range(args.num_users):
        local_w.append(global_weights)

    # print("Test 1 ---------------------------------------")
    # for param_tensor in global_weights:
    #     print(param_tensor, "\t", global_weights[param_tensor].size())
    # print(global_weights.keys())
    # print("Test 1 ---------------------------------------")

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    countprint = 0

    # # MNIST and F-MNIST
    # # All layers
    # users_1 = {
    #     0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9]
    # }
    # # 3 layers
    # users_2 = {
    #     0: [0,1,9], 1: [1,2,0], 2: [2,3,1], 3: [3,4,2],
    #     4: [4,5,3], 5: [5,6,4], 6: [6,7,5],
    #     7: [7,8,6], 8: [8,9,7], 9: [9,0,8]
    # }
    # # 2 layers
    # users_3 = {
    #     0: [0,1,2,8,9], 1: [0,1,2,3,9], 2: [0,1,2,3,4], 3: [1,2,3,4,5],
    #     4: [2,3,4,5,6], 5: [3,4,5,6,7], 6: [4,5,6,7,8],
    #     7: [5,6,7,8,9], 8: [0,6,7,8,9], 9: [0,1,7,8,9]
    # }
    # # 1 layer
    # users_4 = {
    #     0: [0,1,2,3,4,5,6,7,8,9], 1: [0,1,2,3,4,5,6,7,8,9], 2: [0,1,2,3,4,5,6,7,8,9], 3: [0,1,2,3,4,5,6,7,8,9],
    #     4: [0,1,2,3,4,5,6,7,8,9], 5: [0,1,2,3,4,5,6,7,8,9], 6: [0,1,2,3,4,5,6,7,8,9],
    #     7: [0,1,2,3,4,5,6,7,8,9], 8: [0,1,2,3,4,5,6,7,8,9], 9: [0,1,2,3,4,5,6,7,8,9]
    # }

    # CIFAR

    # users_0 = {
    #     0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9]
    # }
    # # 5 layers
    # users_1 = {
    #     0: [0,1], 1: [1,2], 2: [2,3], 3: [3,4],
    #     4: [4,5], 5: [5,6], 6: [6,7],
    #     7: [7,8], 8: [8,9], 9: [9,0]
    # }
    # # 4 layers
    # users_3 = {
    #     0: [0,1,8,9], 1: [0,1,2,9], 2: [0,1,2,3], 3: [1,2,3,4],
    #     4: [2,3,4,5], 5: [3,4,5,6], 6: [4,5,6,7],
    #     7: [5,6,7,8], 8: [6,7,8,9], 9: [0,7,8,9]
    # }
    # # 3 layer
    # users_6 = {
    #     0: [0,1,2,3,7,8,9], 1: [0,1,2,3,4,5,8,9], 2: [0,1,2,3,4,5,9], 3: [0,1,2,3,4,5,6],
    #     4: [1,2,3,4,5,6,7], 5: [2,3,4,5,6,7,8], 6: [3,4,5,6,7,8,9],
    #     7: [0,4,5,6,7,8,9], 8: [0,1,5,6,7,8,9], 9: [0,1,2,6,7,8,9]
    # }
    # 2 layer
    # users_8 = {
    #     0: [0,1,2,3,4,6,7,8,9], 1: [0,1,2,3,4,5,7,8,9], 2: [0,1,2,3,4,5,6,8,9], 3: [0,1,2,3,4,5,6,7,9],
    #     4: [0,1,2,3,4,5,6,7,8], 5: [1,2,3,4,5,6,7,8,9], 6: [0,2,3,4,5,6,7,8,9],
    #     7: [0,1,3,4,5,6,7,8,9], 8: [0,1,2,4,5,6,7,8,9], 9: [0,1,2,3,5,6,7,8,9]
    # }
    # # 1 layer
    # users_9 = {
    #     0: [0,1,2,3,4,5,6,7,8,9], 1: [0,1,2,3,4,5,6,7,8,9], 2: [0,1,2,3,4,5,6,7,8,9], 3: [0,1,2,3,4,5,6,7,8,9],
    #     4: [0,1,2,3,4,5,6,7,8,9], 5: [0,1,2,3,4,5,6,7,8,9], 6: [0,1,2,3,4,5,6,7,8,9],
    #     7: [0,1,2,3,4,5,6,7,8,9], 8: [0,1,2,3,4,5,6,7,8,9], 9: [0,1,2,3,4,5,6,7,8,9]
    # }

# random topology 1
    users_0 = {
        0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9]
    }

    users_1 = {
        0: [0, 1], 1: [1, 2], 2: [2], 3: [3],
        4: [4], 5: [4, 5, 6], 6: [5, 6, 7],
        7: [5, 6, 7, 8], 8: [8, 9], 9: [9]
    }

    users_2 = {
        0: [0, 1, 9], 1: [1, 2], 2: [2, 3], 3: [3],
        4: [2, 3, 4, 5, 6], 5: [5, 6], 6: [6, 7],
        7: [5, 6, 7, 8, 9], 8: [6, 7, 8, 9], 9: [7, 8, 9, 0]
    }

    users_3 = {
        0: [0, 1, 2, 8, 9], 1: [0, 1, 9], 2: [1, 2, 3], 3: [3, 4],
        4: [2, 3, 4, 5, 6], 5: [4, 5, 6], 6: [5, 6, 7],
        7: [5, 6, 7, 8, 9], 8: [6, 7, 8, 9, 0], 9: [0, 1, 2, 7, 8, 9]
    }

    users_4 = {
        0: [0, 1, 2, 8, 9], 1: [0, 1, 9], 2: [0, 1, 2, 3, 4], 3: [2, 3, 4],
        4: [3, 4, 5, 6, 7], 5: [3, 4, 5, 6, 7], 6: [4, 5, 6, 7, 8],
        7: [5, 6, 7, 8, 9], 8: [0, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 6, 7, 8, 9]
    }

    users_5 = {
        0: [0, 1, 2, 8, 9], 1: [0, 1, 2, 3, 8, 9], 2: [0, 1, 2, 3, 4, 5], 3: [2, 3, 4],
        4: [2, 3, 4, 5, 6, 7], 5: [3, 4, 5, 6, 7], 6: [2, 3, 4, 5, 6, 7, 8, 9],
        7: [5, 6, 7, 8, 9], 8: [0, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 5, 6, 7, 8, 9]
    }

    users_6 = {
        0: [0, 1, 2, 8, 9], 1: [0, 1, 2, 3, 7, 8, 9], 2: [9, 0, 1, 2, 3, 4, 5], 3: [2, 3, 4],
        4: [2, 3, 4, 5, 6, 7], 5: [2, 3, 4, 5, 6, 7, 8], 6: [2, 3, 4, 5, 6, 7, 8, 9, 0],
        7: [0, 3, 4, 5, 6, 7, 8, 9], 8: [0, 1, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 4, 5, 6, 7, 8, 9]
    }

    users_7 = {
        0: [0, 1, 2, 8, 9], 1: [0, 1, 2, 3, 6, 7, 8, 9], 2: [9, 0, 1, 2, 3, 4, 5, 6], 3: [0, 1, 2, 3, 4, 5],
        4: [0, 1, 2, 3, 4, 5, 6, 7, 8], 5: [2, 3, 4, 5, 6, 7, 8], 6: [2, 3, 4, 5, 6, 7, 8, 9, 0],
        7: [0, 3, 4, 5, 6, 7, 8, 9], 8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    users_8 = {
        0: [0, 1, 2, 8, 9], 1: [0, 1, 2, 3, 4, 5, 7, 8, 9], 2: [0, 1, 2, 3, 4, 6, 7, 8, 9],
        3: [0, 1, 2, 3, 4, 5, 7, 8, 9],
        4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5: [0, 1, 2, 3, 4, 5, 6, 7, 9], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    users_9 = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 8: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    # a = users_3
    # for idx in range(10):
    #     print(len(a[idx])-1)
    # breakpoint()

# size is MB
    param_size = 0
    count = 0
    for param in global_model.parameters():
        print(param.nelement())
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in global_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Model size: {:.3f}MB'.format(size_all_mb))

    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)



        w = [[],[],[],[],[],[],[],[],[],[]]
        loss = [[],[],[],[],[],[],[],[],[],[]]
        # print("m = ", m)
        # print("idxs_users: ", idxs_users)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            temp_model.load_state_dict(local_w[idx])
            w[idx], loss[idx] = local_model.update_weights(
                model=copy.deepcopy(temp_model), global_round=epoch)
            # Delete
            local_weights.append(copy.deepcopy(w[idx]))
            local_losses.append(copy.deepcopy(loss[idx]))

        group_w = [[], [], [], [], [], [], [], [], [], []]
        group_loss = [[], [], [], [], [], [], [], [], [], []]
        averaged_w = [[], [], [], [], [], [], [], [], [], []]
        user_x = users_4

        for idx in idxs_users:
            # group_w[idx] = w[idx]
            for neighbour in user_x[idx]:
                group_w[idx].append(copy.deepcopy(w[neighbour]))
                group_loss[idx].append(copy.deepcopy(loss[neighbour]))
            averaged_w[idx] = average_weights(group_w[idx])

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        local_w = w

        for idx in idxs_users:
            # local_w[idx]['fc4.weight'] = averaged_w[idx]['fc4.weight']
            # local_w[idx]['fc4.bias'] = averaged_w[idx]['fc4.bias']
            # local_w[idx]['fc3.weight'] = averaged_w[idx]['fc3.weight']
            # local_w[idx]['fc3.bias'] = averaged_w[idx]['fc3.bias']
            local_w[idx]['fc2.weight'] = averaged_w[idx]['fc2.weight']
            local_w[idx]['fc2.bias'] = averaged_w[idx]['fc2.bias']
            local_w[idx]['fc1.weight'] = averaged_w[idx]['fc1.weight']
            local_w[idx]['fc1.bias'] = avazeraged_w[idx]['fc1.bias']
            # local_w[idx]['conv3.weight'] = averaged_w[idx]['conv3.weight']
            # local_w[idx]['conv3.bias'] = averaged_w[idx]['conv3.bias']
            # local_w[idx]['conv2.weight'] = averaged_w[idx]['conv2.weight']
            # local_w[idx]['conv2.bias'] = averaged_w[idx]['conv2.bias']
            # local_w[idx]['conv1.weight'] = averaged_w[idx]['conv1.weight']
            # local_w[idx]['conv1.bias'] = averaged_w[idx]['conv1.bias']

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        list_cv_acc, list_cv_loss = [], []
        global_model.eval()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            temp_model.load_state_dict(local_w[idx])
            temp_model.eval()
            acc, loss = local_model.inference(model=temp_model)
            list_acc.append(acc)
            list_loss.append(loss)

            cv_acc_temp, cv_loss_temp = test_inference(args, temp_model, test_dataset)
            list_cv_acc.append(cv_acc_temp)
            list_cv_loss.append(cv_loss_temp)

        train_accuracy.append(sum(list_acc)/len(list_acc))
        cv_acc.append(sum(list_cv_acc) / len(list_cv_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('CV Accuracy: {:.2f}% \n'.format(100*cv_acc[-1]))
            print(list_cv_acc)

    # Test inference after completion of training
    test_acc_list, test_loss_list = [], []
    for idx in range(args.num_users):
        temp_model.load_state_dict(local_w[idx])
        test_acc, test_loss = test_inference(args, temp_model, test_dataset)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    print(test_acc_list)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*sum(test_acc_list)/len(test_acc_list)))

    now = datetime.now()
    print("Time: ", now.strftime("%d-%m-%Y %H:%M:%S"))
    # Saving the objects train_loss and train_accuracy:
    file_name = '/home/sili/Documents/PhD/Federated-Learning-PyTorch-master/save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{}.pkl'.\
        format(args.dataset, args.model, len(user_x[0]), args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, now.strftime("%d-%m-%Y %H:%M:%S"))

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy,cv_acc], f)

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
    plt.savefig('/home/sili/Documents/PhD/Federated-Learning-PyTorch-master/save/fed_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss_{}.png'.
                format(args.dataset, args.model, len(user_x[0]), args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs,now.strftime("%d-%m-%Y %H:%M:%S")))

    # Plot Average Accuracy vs Communication rounds
    max_index_train = train_accuracy.index(max(train_accuracy))
    max_index_cv = cv_acc.index(max(cv_acc))
    max_train_accuracy = max(train_accuracy)
    max_train_cv = max(cv_acc)
    communication_rounds = range(len(train_accuracy))
    print(f"{max_train_accuracy:.4f}",f"{max_train_cv:.4f}")
    print(len(user_x[0])-1)

    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')

    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')

    plt.plot(communication_rounds[max_index_train], max_train_accuracy, 'o', markersize=4, markeredgecolor='k', markerfacecolor='none')
    plt.annotate(f'{max_train_accuracy:.4f}', xy=(communication_rounds[max_index_train], max_train_accuracy),
                 xytext=(communication_rounds[max_index_train] - 0.2, max_train_accuracy - 0.01), color='k')

    plt.plot(range(len(cv_acc)), cv_acc, color='g')

    plt.plot(communication_rounds[max_index_cv], max_train_cv, 'o', markersize=4, markeredgecolor='g', markerfacecolor='none')
    plt.annotate(f'{max_train_cv:.4f}', xy=(communication_rounds[max_index_cv], max_train_cv),
                 xytext=(communication_rounds[max_index_cv] - 0.2, max_train_cv - 0.01), color='g')

    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/sili/Documents/PhD/Federated-Learning-PyTorch-master/save/fed_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_{}.png'.
                format(args.dataset, args.model, len(user_x[0]), args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs,now.strftime("%d-%m-%Y %H:%M:%S")))
