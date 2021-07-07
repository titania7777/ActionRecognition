import time
import os
import random
import argparse
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from Core.utils import initialize_weights, path_manager, args_printer, models_printer, status_printer, StatusCalculator, ModelSaver, get_device, get_model, split_model, merge_model
from Core.VideoDataset import VideoDataset

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def main(frames_path:str, train_annotation_path:str, val_annotation_path:str, save_path:str, args):
    # random.seed(7777)
    # np.random.seed(7777)
    # torch.manual_seed(7777)

    # Display and Save of Arguments
    _, _, _, args_dict = inspect.getargvalues(inspect.currentframe())
    args_printer(args_dict=args_dict, save_path=save_path)

    # Train Dataset and DataLoader
    train_dataset = VideoDataset(
        frames_path=frames_path,
        annotation_path=train_annotation_path,
        sampled_split=args.sampled_split,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        max_interval=args.max_interval,
        random_start_position=args.random_start_position,
        uniform_frame_sample=args.no_uniform_frame_sample,
        random_pad_sample=args.random_pad_sample,
        train=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0 if os.name == 'nt' else args.num_workers,
        shuffle=True,
        pin_memory=True
    )

    # Train Dataset and DataLoader
    validate_datast = VideoDataset(
        frames_path=frames_path,
        annotation_path=val_annotation_path,
        sampled_split=False,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        max_interval=-1,
        random_start_position=False,
        uniform_frame_sample=True,
        random_pad_sample=False,
        train=False
    )

    validate_loader = DataLoader(
        dataset=validate_datast,
        batch_size=args.batch_size,
        num_workers=0 if os.name == 'nt' else args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # Get Model
    model = get_model(args.model_name, n_classes=train_dataset.num_classes, pretrained=args.pretrained, progress=True)
    model_freeze, model_tune = split_model(model, args.tune_layer)

    # Display of Model Structure
    models_printer(freeze=model_freeze, tune=model_tune, save_path=save_path)

    # DataParallel
    if args.data_parallel and torch.cuda.device_count() > 1:
        if model_freeze:
            model_freeze = DataParallel(model_freeze, device_ids=args.parallel_gpu_numbers, output_device=args.main_gpu_number)
        model_tune = DataParallel(model_tune, device_ids=args.parallel_gpu_numbers, output_device=args.main_gpu_number)


    # To Device
    device = get_device(gpu_number=args.main_gpu_number, cudnn_benchmark=True)
    if model_freeze:
        model_freeze.to(device)
        model_freeze.eval()
    model_tune.to(device)

    # Initialize
    model_tune.apply(initialize_weights)
    
    # Optimizer
    optimizer = torch.optim.SGD(model_tune.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    # Display Training Informations
    print("[train] number of videos: {}, [val] number of videos: {}".format(len(train_dataset), len(validate_datast)))

    # Accuracy and Weights Saver
    train_calculator = StatusCalculator(num_batchs=len(train_loader), topk=1)
    validate_calculator = StatusCalculator(num_batchs=len(validate_loader), topk=1)
    saver = ModelSaver(per=10)
    # ================================================
    # Main Loop
    # ================================================
    for e in range(1, args.num_epochs+1):
        # ================================================
        # Training Loop
        # ================================================
        train(e, train_calculator, model_freeze, model_tune, train_loader, optimizer, save_path, device)

        # ================================================
        # Validation Loop
        # ================================================
        val(e, validate_calculator, model_freeze, model_tune, validate_loader, save_path, device)

        # Display Final Status and Save
        saver.auto_save(merge_model(freeze=model_freeze, tune=model_tune), save_path, best_acc=validate_calculator.get_acc(best=True), best_loss=validate_calculator.get_loss(best=True))
        print(" Best Accuracy: {:.2f}%".format(validate_calculator.get_acc(best=True) * 100))
        
        # Initialize Calculator
        lr_scheduler.step(validate_calculator.get_loss(total=True))
        train_calculator.reset()
        validate_calculator.reset()

def train(current_epoch:int, status_calculator:StatusCalculator, model_freeze:nn.Sequential, model_tune:nn.Sequential, data_loader:DataLoader, optimizer:torch.optim, save_path:str, device:str):
    model_tune.train()
    for i, (datas, labels) in enumerate(data_loader):
        datas = Variable(datas, requires_grad=True).to(device, non_blocking=True)
        labels = Variable(labels, requires_grad=False).to(device, non_blocking=True)
        
        # Prediction
        if model_freeze:
            with torch.no_grad():
                datas = model_freeze(datas)
        pred = model_tune(datas)

        # Loss
        loss = F.cross_entropy(pred, labels)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Status
        status_calculator.set_acc(pred, labels)
        status_calculator.set_loss(loss)

        # Display Status
        status_printer(
            "train", current_epoch, args.num_epochs, i+1, len(data_loader),
            status_calculator.get_loss(), status_calculator.get_loss(mean=True),
            status_calculator.get_acc() * 100, status_calculator.get_acc(mean=True) * 100, save_path=save_path
        )
    print("")
        
def val(current_epoch:int, status_calculator:StatusCalculator, model_freeze:nn.Sequential, model_tune:nn.Sequential, data_loader:DataLoader, save_path:str, device:str):
    model_tune.eval()
    with torch.no_grad():
        for i, (datas, labels) in enumerate(data_loader):
            datas = Variable(datas, requires_grad=False).to(device, non_blocking=True)
            labels = Variable(labels, requires_grad=False).to(device, non_blocking=True)
            
            # Prediction
            if model_freeze:
                datas = model_freeze(datas)
            pred = model_tune(datas)

            # Loss
            loss = F.cross_entropy(pred, labels)

            # Status
            status_calculator.set_acc(pred, labels)
            status_calculator.set_loss(loss)

            # Display Status
            status_printer(
                f"val", current_epoch, args.num_epochs, i+1, len(data_loader),
                status_calculator.get_loss(), status_calculator.get_loss(mean=True),
                status_calculator.get_acc() * 100, status_calculator.get_acc(mean=True) * 100, save_path=save_path
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./Data/")
    parser.add_argument("--models-path", type=str, default="./Save/")
    parser.add_argument("--dataset-name", type=str, default="UCF101", choices={"UCF101", "HMDB51", "ActivityNet"})
    parser.add_argument("--id", type=int, default=1)
    # For Training
    parser.add_argument("--model-name", type=str, default="R3D18", choices={"R3D18", "R3D34", "R3D50", "R2Plus1D18", "R2Plus1D34", "R2Plus1D50"})
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--max-interval", type=int, default=-1)
    parser.add_argument("--random-start-position", action="store_true")
    parser.add_argument("--random-pad-sample", action="store_true") # Default is repeated pad
    parser.add_argument("--no-uniform-frame-sample", action="store_false") # Default is uniform sampling
    parser.add_argument("--sampled-split", action="store_true") # Testing for our method, See this https://github.com/titania7777/VideoFrameSampler
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tune-layer", type=int, default=-1) # -1 means using the all layers
    parser.add_argument("--num-workers", type=int, default=4)
    # For Parallel Mode
    parser.add_argument("--data-parallel", action="store_true") # [Single mode, DataParallel mode]
    parser.add_argument("--main-gpu-number", type=int, default=0)
    parser.add_argument("--parallel-gpu-numbers", nargs="+", default=[0, 1])
    args = parser.parse_args()

    # Path organize
    frames_path = os.path.join(args.data_path, f"{args.dataset_name}/frames/")
    if args.sampled_split:
        train_annotation_path = os.path.join(args.data_path, f"{args.dataset_name}/labels/sampled_split/", f"train_{args.id}.json")
        # val_annotation_path = os.path.join(args.data_path, f"{args.dataset_name}/labels/sampled_split/", f"test_{args.id}.json" if args.dataset_name == "UCF101" else f"val_{args.id}.json")
    else:
        train_annotation_path = os.path.join(args.data_path, f"{args.dataset_name}/labels/custom_split/", f"train_{args.id}.csv")
    val_annotation_path = os.path.join(args.data_path, f"{args.dataset_name}/labels/custom_split/", f"test_{args.id}.csv" if args.dataset_name == "UCF101" else f"val_{args.id}.csv")
    save_path = os.path.join(args.models_path, f"{args.model_name}/", f"{args.dataset_name}_All/" if args.tune_layer == -1 else f"{args.dataset_name}_{args.tune_layer}/")

    # Path Check
    path_manager(frames_path, train_annotation_path, val_annotation_path, raise_error=True, path_exist=True)
    path_manager(save_path, raise_error=False, create_new=True, remove_response=True)

    # Run
    main(frames_path, train_annotation_path, val_annotation_path, save_path, args)