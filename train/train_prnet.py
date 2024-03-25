#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
import numpy as np
import argparse
from dataloaders.Position_dataloader import PositionDataloader1, RandomDoubleCrop1, ToPositionTensor1
import torchvision.utils as vutils
import cv2
def train(config_file):
    # Load configuration parameters
    writer = SummaryWriter()
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']
    config_prnet = config['prnetwork']
    config_train = config['training']
    batch_size = config_data.get('batch_size', 4)
    best_loss = config_train.get('best_loss', 0.5)
    num_workers = config_train.get('num_workers', 1)

    # Create the model
    print('Create model')

    prnet = NetFactory.create(config_prnet['net_type'])(
        inc=config_prnet.get('input_channel', 1),
        patch_size=np.asarray(config_data['patch_size']),
        base_chns=config_prnet.get('base_feature_number', 16),
        norm='in',
        dilation=config_prnet.get('dilation', 1),
        n_classes=config_prnet['class_num'],
        droprate=config_prnet.get('drop_rate', 0.2),
    )

    # Define the optimizer and loss function
    Adamoptimizer = optim.Adam(prnet.parameters(), lr=config_train['learning_rate'], weight_decay=config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.StepLR(Adamoptimizer, step_size=15, gamma=0.8)
    loss_func = nn.MSELoss()
    # needs taken care of
    dis_ratio = torch.FloatTensor(config_data.get('distance_ratio')).unsqueeze(dim=0)

    # DataLoader: load data
    trainData = PositionDataloader1(config=config_data,
                                    image_path=config_data['train_image_path'],
                                    transform=transforms.Compose([
                                       RandomDoubleCrop1(config_data['patch_size'],
                                                         small_move=config_train['small_move'],
                                                         fluct_range=config_train['fluct_range']),
                                       ToPositionTensor1(),
                                   ]),
                                   out_size=config_data['patch_size'])
    config_data['iter_num'] = 10
    trainLoader = DataLoader(trainData, shuffle=True, batch_size=batch_size, pin_memory=True)

    print('Start to train')
    start_it = config_train.get('start_iteration', 0)
    print_iter = 0
    output_dir1 = '../data/predicted_images'
    os.makedirs(output_dir1, exist_ok=True)

    output_dir2 = '../data/real_images'
    os.makedirs(output_dir2, exist_ok=True)
    # Training loop
    for epoch in range(start_it, config_train['maximal_epoch']):
        print('#######epoch:', epoch)
        optimizer = Adamoptimizer

        'train'
        prnet.train()
        for i_batch, sample_batch in enumerate(trainLoader):
            img_batch0, img_batch1, rela_distance_batch = sample_batch['random_crop_image_0'], \
                                                          sample_batch['random_crop_image_1'], \
                                                          sample_batch['rela_distance']
            # print(img_batch0)
            # print(img_batch0.shape)
            # Forward pass
            image1 = img_batch0/255
            image2 = img_batch1/255
            predic_0 = prnet(image1)
            predic_1 = prnet(image2)
            predic_ae_0, predic_cor_fc_0 = torch.sigmoid(predic_0['ae']), predic_0['fc_position']
            # print(predic_ae_0)
            # print(predic_ae_0.shape)
            predic_ae_1, predic_cor_fc_1 = torch.sigmoid(predic_1['ae']), predic_1['fc_position']

            vutils.save_image(predic_ae_0.data,
                              os.path.join(output_dir1, f'predic_ae_0_epoch_{epoch}_batch_{i_batch}.png'),
                              normalize=True)

            vutils.save_image(image1.data,
                              os.path.join(output_dir2, f'img_batch0_epoch_{epoch}_batch_{i_batch}.png'),
                              normalize=False)

            ae_train_loss = loss_func(predic_ae_0, image1) + loss_func(predic_ae_1, image2)
            fc_predic = dis_ratio * torch.tanh(predic_cor_fc_0 - predic_cor_fc_1)
            fc_train_loss = loss_func(fc_predic, rela_distance_batch)
            train_loss = ae_train_loss + fc_train_loss
            # print(train_loss)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Logging
            writer.add_scalar('Loss/train', train_loss.item(), epoch * len(trainLoader) + i_batch)
            print(f'Epoch: {epoch}, Batch: {i_batch}, Loss: {train_loss.item()}')
        Adamscheduler.step()

        # Save model checkpoint
        if epoch % config_train['snapshot_epoch'] == 0:
            checkpoint_path = os.path.join(config_train['model_save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': prnet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/train/train_pet', help='Config file path')
    args = parser.parse_args()
    assert os.path.isfile(args.config_path), "Config file not found."
    train(args.config_path)
