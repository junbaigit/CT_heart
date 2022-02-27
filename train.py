
import os
from os.path import exists
import time
from glob import glob
import argparse
import random
import torch
from PIL import Image
from torch.utils.data import DataLoader
import monai
from monai.data import ArrayDataset
from monai.data import PersistentDataset, list_data_collate, SmartCacheDataset, partition_dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai import transforms as mt
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
from torch.nn.functional import interpolate
from monai.optimizers import Novograd
from model import UNet3D
import numpy as np
#import wandb
import matplotlib.pyplot as plt
from os.path import exists

dimension = 28
pjoin = os.path.join

class ConvertToMultiChannelBasedOnBratsClassesd(mt.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:


            result = []

            for i in range(1, 11):
                result.append(d[key] == i)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


def compute_loss_list(loss_fn, preds, label):
    labels = [label] + [
        interpolate(label, pred.shape[2:]) for pred in preds[1:]
    ]
    return sum(
        0.5 ** i * loss_fn(p, l)
        for i, (p, l) in enumerate(zip(preds, labels))
    )


def get_transforms(args):
    train_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
            #mt.Resized(keys=['img', 'seg'], spatial_size = [64,64,64], mode ='nearest'),
            mt.EnsureChannelFirstd(keys=['img']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            mt.ScaleIntensityD(keys=['img']),
            mt.RandGaussianNoiseD(keys=['img']),
            mt.ToTensorD(keys=['img', 'seg']),
        ]
    )

    val_trans = mt.Compose(
        [
            mt.LoadImageD(keys=['img', 'seg']),
           # mt.Resized(keys = ['img', 'seg'], spatial_size=[64, 64, 64], mode='nearest'),
            mt.EnsureChannelFirstd(keys=['img']),
            ConvertToMultiChannelBasedOnBratsClassesd(keys='seg'),
            mt.ScaleIntensityD(keys=['img']),
            mt.RandGaussianNoiseD(keys=['img']),
            mt.ToTensorD(keys=['img', 'seg']),
            #mt.AsDiscreteD(keys=['seg'], to_onehot=10),
        ]
    )
    return train_trans, val_trans

def main_worker(args):
    PATH = 'model.pt'
    #data_folder = 'E:\\chd_seg_10_classes_four_centers\\numpy\\'
    data_folder = './train_data/'
    #data_folder = args.dataDic
    images = sorted(glob(pjoin(data_folder,'rmyy_nii_img','*.nii.gz')))
    segs = sorted(glob(pjoin(data_folder,'rmyy_nii_label','*.nii.gz')))

    data_dicts = [
        {"img": image_name, "seg": label_name}
        for image_name, label_name in zip(images, segs)
    ]

    random.shuffle(data_dicts)

    val_idx = int(0.1 * len(images))


    train_files, val_files  = data_dicts[:-val_idx], data_dicts[-val_idx:]

    train_trans, val_trans = get_transforms(args)


    train_ds = PersistentDataset(data=train_files, transform=train_trans,cache_dir='./train_cache')
    val_ds = PersistentDataset(data=val_files, transform=val_trans,cache_dir='./val_cache')



    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.test_batch_size,
                            num_workers=1)  # , pin_memory=torch.cuda.is_available())

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = mt.Compose([
        mt.Activations(sigmoid=True),
        mt.AsDiscrete(threshold_values=True),
    ])

    # ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    #wandb.init(project="unet3D-project")

    loss_function = monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

    if args.fast:
        optimizer = Novograd(model.parameters(), args.lr * 50)
        scaler = torch.cuda.amp.GradScaler()

        file_exists = exists(PATH)
        if file_exists:
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(checkpoint.keys())
            EPOCH = checkpoint['epoch:']
            loss = checkpoint['loss']
        else:
            EPOCH = 0
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        file_exists = exists(PATH)
        if file_exists:
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(checkpoint.keys())
            EPOCH = checkpoint['epoch:']
            loss = checkpoint['loss']
        else:
            EPOCH = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    ###################################
    #         Training
    ###################################
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    best_metrics_epochs_and_time = [[], [], []]
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()

    count = 0

    for epoch in range(EPOCH, args.epochs):
        print("-" * 50)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:

            step_start = time.time()
            step += 1
            inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
           # print('labels shape: ',labels.shape)
            optimizer.zero_grad()
            if args.fast:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if args.arch == 'dynunet':
                        loss = compute_loss_list(loss_function, outputs, labels)
                    else:
                        loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
               # print('output: ',outputs.shape)
                loss = loss_function(outputs, labels)
                # print(torch.max(labels), torch.min(labels))
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if step % args.print_freq == 0:
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}" 
                      f", step time: {(time.time() - step_start):.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        ###################################
        #         Validation
        ###################################

        if (epoch + 1) % args.val_inter == 0:
            count +=1
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)
                    val_outputs = model(val_images)

                    val_outputs = post_trans(val_outputs)

                    # if count == 30:
                    #     cpu_pred = val_outputs.cpu()
                    #     result = cpu_pred.data.numpy()
                    #     np.save(result)

                    #value= dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    # metric_count += len(value)
                    # metric_sum += value.item() * len(value)

                #metric = metric_sum / metric_count
                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), pjoin('checkpoints', f'{args.arch}_best.pth'))
                    torch.save({'epoch:': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'loss': epoch_loss}, PATH)
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

                fldr = "plot/ultra_" + args.ext
                try:
                    os.makedirs(fldr, exist_ok=True)
                except TypeError:
                    raise Exception("Direction not create!")
            scheduler.step(metric)

    totol_time = time.time() - total_start
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {totol_time}.")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [args.val_inter * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig('loss_dice.jpg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--data", default='gd', type=str)
    parser.add_argument("--dims", default=(14,14,14), type=list)
    parser.add_argument("--arch", default='unet', type=str)
    parser.add_argument("--val_inter", default=1, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--fast", default=False, type=bool)
    parser.add_argument("--dataDic", default='./numpy')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--ext", default='unet', type=str)
    args = parser.parse_args()
    print(args)
    main_worker(args)

if __name__ == "__main__":
    main()
