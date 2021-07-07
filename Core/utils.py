import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
import scipy.stats as stats
from sklearn.metrics import average_precision_score
from torch.backends import cudnn
from torch.hub import load_state_dict_from_url
from collections import OrderedDict

from Core.Models.R3D import r3d_18, r3d_34, r3d_50
from Core.Models.R2Plus1D import r2plus1d_18, r2plus1d_34, r2plus1d_50

"""
help to calculate accuracy and loss
"""
class StatusCalculator():
    def __init__(self, num_batchs:int, topk:int = 1):
        self.num_batchs = num_batchs
        self.topk = topk
        self.reset()

        # for best loss score
        self.best_loss = 10000000000.0

        # for best accuracy score
        self.best_acc = 0.0

    def set_loss(self, loss:torch.Tensor):
        with torch.no_grad():
            # counting and memorize loss value
            self.counter_loss += 1
            self.current_loss = loss.item()
            self.total_loss += self.current_loss
            
            # mean loss calculation of stacked total iterations
            self.mean_loss = self.total_loss / self.counter_loss
            
            # set best loss per batch
            if self.counter_loss == self.num_batchs and self.mean_loss < self.best_loss:
                self.best_loss = self.mean_loss
    
    """
    (True) best => return best loss score, (False) best => (True) mean => return mean loss score
                                                           (False) mean => return current loss score
    """
    def get_loss(self, best:bool=False, mean:bool=False, total:bool=False) -> float:
        if best:
            return self.best_loss
        else:
            if mean:
                return self.mean_loss
            elif total:
                return self.total_loss
            else:
                return self.current_loss

    def set_acc(self, pred:torch.Tensor, labels:torch.Tensor):
        with torch.no_grad():
            batch_size = len(labels) # get batch size from labels
            self.counter_batch += 1
            self.counter_acc += batch_size

            # calculate topk accuracy score
            _, indices = torch.topk(pred, k=self.topk, dim=1, largest=True, sorted=False)
            self.current_corrects = sum(torch.any(indices == labels.unsqueeze(1).repeat(1, self.topk), dim=1)).item()
            self.total_corrects += self.current_corrects

            # accuracy score calculation
            self.current_acc = self.current_corrects / batch_size # current batch accuracy score
            self.mean_acc = self.total_corrects / self.counter_acc # mean accuracy score of stacked total iterations
            
            # set best accuracy score per batch
            if self.counter_batch == self.num_batchs and self.mean_acc > self.best_acc:
                self.best_acc = self.mean_acc
    
    """
    (True) best => return best accuracy score, (False) best => (True) mean => return mean accuracy score
                                                               (False) mean => return mean accuracy score
    """
    def get_acc(self, best:bool=False, mean:bool=False) -> float:
        if best:
            return self.best_acc
        else:
            if mean:
                return self.mean_acc
            else:
                return self.current_acc
    
    """
    must be called after the end of train or test iteration
    """
    def reset(self, best=False):
        # for loss
        self.total_loss = 0.0
        self.current_loss = 0.0
        self.mean_loss = 0.0
        self.counter_loss = 0

        # for accuracy
        self.total_corrects = 0
        self.current_corrects = 0
        self.mean_acc = 0.0
        self.current_acc = 0.0
        self.counter_batch = 0
        self.counter_acc = 0

        if best:
            self.best_loss = 10000000000.0
            self.best_acc = 0.0

"""
help to save the model
"""
class ModelSaver():
    def __init__(self, per:int = 5):
        self.counter = 0
        self.per = per
        self.best_acc = 0.0
        self.best_loss = 10000000000.0
    
    """
    (Not None) best_acc => save model with best accuracy score
    (Not None) best_loss => save model with best loss score
    """
    def auto_save(self, model:nn.Sequential, save_path:str, best_acc:float = None, best_loss:float = None):
        self.counter += 1
        if self.counter % self.per == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"{self.counter}ep.pth"))
        
        if best_acc:
            if best_acc > self.best_acc:
                self.best_acc = best_acc
                torch.save(model.state_dict(), os.path.join(save_path, f"best_acc.pth"))
        
        if best_loss:
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                torch.save(model.state_dict(), os.path.join(save_path, f"best_loss.pth"))

"""
help to management of path
(True) raise_error => raise error when conduct path_exist
(True) path_exist => check a path exist
(True) create_new => make a new directory
(True) remove_enforcement => remove a path by enforcement
(True) remove_response => remove a path by considering of user respond
"""
def path_manager(*paths, raise_error:bool = False, path_exist:bool = False, create_new:bool = False, remove_enforcement:bool = False, remove_response:bool = False) -> bool:
    for path in paths:
        if path == None:
            continue

        exist = os.path.exists(path)
        # path check
        if path_exist:
            if raise_error:
                assert exist, f"{path} is not exist !!"
            else:
                if not exist:
                    return False
        
        # remove => create (possible)
        # path remove(enforcement)
        if remove_enforcement and exist:
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        # path remove(response)
        elif remove_response and exist:
            while True:
                print(f"'{path}' is already exist, do you want to continue after remove that ? [y/n]")
                response = input()

                # yes
                if response == "y" or response == "Y" or response == "yes":
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                        os.makedirs(path)
                    break
            
                # no
                if response == "n" or response == "N" or response == "no":
                    print("this script was terminated by a user")
                    sys.exit()
        
        # path create
        if create_new and not exist:
            os.makedirs(path)

    return True

"""
help to print of arguments
"""
def args_printer(args_dict:dict, save_path:str = None):
    lines = []
    lines.append("=================================================")
    for key in args_dict:
        if key == "args":
            sub_args_dict = vars(args_dict[key])
            for sub_key in sub_args_dict:
                lines.append(f"{sub_key}:{sub_args_dict[sub_key]}")
            continue
        lines.append(f"{key}:{args_dict[key]}")
    lines.append("=================================================")

    # print
    [print(line)for line in lines]

    # save
    if save_path:
        with open(os.path.join(save_path, "args.txt"), "w") as f:
            f.write("\n".join(lines))

"""
help to print of models layer
"""
def models_printer(freeze:nn.Sequential=None, tune:nn.Sequential = None, classifier:nn.Linear = None, save_path:str = None):
    lines = []
    if freeze:
        lines.append("====================FREEZING LAYER====================")
        lines.append(str(freeze))
    if tune:
        lines.append("=====================TUNING LAYER=====================")
        lines.append(str(tune))
    if classifier:
        lines.append("===================CLASSIFIER LAYER===================")
        lines.append(str(classifier))
    lines.append("======================================================")
    
    [print(line)for line in lines]
    if save_path:
        with open(os.path.join(save_path, "models.txt"), "w") as f:
            f.write("\n".join(lines))

"""
help to print of iteration status
"""
def status_printer(state:str, current_epoch:int, total_epoch:int, current_batch:int, total_batch:int, loss:float, loss_mean:float, acc:float, acc_mean:float, save_path:str = None):
    msg = "[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.4f} (mean: {:.4f}), Acc: {:.2f}% (mean: {:.2f}%)]".format(
            state,
            current_epoch,
            total_epoch,
            current_batch,
            total_batch,
            loss,
            loss_mean,
            acc,
            acc_mean
        )
    
    sys.stdout.write("\r" + msg)

    if save_path:
        with open(os.path.join(save_path, "status.txt"), "a") as f:
            f.write(msg + "\n")

"""
help to gain some specific model
"""
def get_model(model_name:str, n_classes:int, pretrained:bool = False, model_path:str = None, progress:bool = True) -> (int, nn.Module):
    # pretrained on kinetics 700
    # original weights: https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4
    # these weights modified for our experiments
    weights_on_kinetics_700 = {
        "R3D18": "https://www.dropbox.com/s/afoqrewod4meewk/r3d18.pth?dl=1",
        "R3D34": "https://www.dropbox.com/s/0vn1vlts8rjo7cw/r3d34.pth?dl=1",
        "R3D50": "https://www.dropbox.com/s/21e539j3kw0dg1s/r3d50.pth?dl=1",
        "R2Plus1D18": "https://www.dropbox.com/s/jby9zmcdz28bwbo/r2plus1d18.pth?dl=1",
        "R2Plus1D34": "https://www.dropbox.com/s/ei1qi0vap7huk7c/r2plus1d34.pth?dl=1",
        "R2Plus1D50": "https://www.dropbox.com/s/8zoz9yeffckvi5y/r2plus1d50.pth?dl=1",
    }

    if model_name == "R3D18":
        model = r3d_18(n_classes)
    elif model_name == "R3D34":
        model = r3d_34(n_classes)
    elif model_name == "R3D50":
        model = r3d_50(n_classes)
    elif model_name == "R2Plus1D18":
        model = r2plus1d_18(n_classes)
    elif model_name == "R2Plus1D34":
        model = r2plus1d_34(n_classes)
    elif model_name == "R2Plus1D50":
        model = r2plus1d_50(n_classes)
    else:
        assert False, f"Model '{model_name}' is not supported :("

    # load a pretrained weights on kinetics 700
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(weights_on_kinetics_700[model_name], progress=progress))
    elif model_path:
        model.load_state_dict(torch.load(model_path))
    
    return model

"""
help to turn-on the GPU acceleration
"""
def get_device(only_cpu:bool = False, gpu_number:int = 0, cudnn_benchmark:bool = True) -> torch.device:
    if torch.cuda.is_available() and not only_cpu:
        if cudnn_benchmark:
            cudnn.benchmark = True
        return torch.device(f"cuda:{gpu_number}")
    else:
        return torch.device("cpu")

"""
help to gain some specific layer of the model
"""
def split_model(model:nn.Module, layer:int) -> (nn.Sequential, nn.Sequential):
    # (header)-(layer1)-(layer2)-(layer3)-(layer4)-(classifier)
    #      [freeze layer]                                    [tune layer]
    # 1 => [(header)-(layer1)-(layer2)-(layer3)-(layer4)]    [(classifier)]
    # 2 => [(header)-(layer1)-(layer2)-(layer3)]             [(layer4)-(classifier)]
    # 3 => [(header)-(layer1)-(layer2)]                      [(layer3)-(layer4)-(classifier)]
    # 4 => [(header)-(layer1)]                               [(layer2)-(layer3)-(layer4)-(classifier)]
    # 5 => [(header)]                                        [(layer1)-(layer2)-(layer3)-(layer4)-(classifier)]
    # -1 => None                                             [(header)-(layer1)-(layer2)-(layer3)-(layer4)-(classifier)]

    if layer == -1:
        return None, model
    layer_map = {1: -1, 2: -2, 3: -3, 4: -4, 5: -5}
    model = list(model.children())
    encoder_freeze = freeze_all(nn.Sequential(*model[:layer_map[layer]]))
    encoder_tune = nn.Sequential(*model[layer_map[layer]:])

    return encoder_freeze, encoder_tune

"""
help to merge the specific layers
"""
def merge_model(freeze:nn.Sequential=None, tune:nn.Sequential=None) -> nn.Sequential:
    model = OrderedDict()
    layer_num = 0
    
    if freeze:
        for m in list(freeze.children()):
            model[f"layer{layer_num}"] = m
            layer_num += 1

    if tune:
        for m in list(tune.children()):
            model[f"layer{layer_num}"] = m
            layer_num += 1

    return nn.Sequential(model)

"""
help to weights freezing of the model
"""
def freeze_all(model:nn.Module):
    for param in model.parameters():
        param.requires_grad = False

"""
help to weights initialization of the model
"""
def initialize_weights(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if not module.bias == None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        if not module.bias == None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=1.0)
        if not module.bias == None:
            nn.init.constant_(module.bias, 0.01)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h # m +-h

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)