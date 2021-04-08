
import os
import torch
import logging
logging.basicConfig(level=logging.DEBUG)
from fltk.nets import Cifar10CNN, FashionMNISTCNN, Cifar100ResNet, FashionMNISTResNet, Cifar10ResNet, Cifar100VGG
from fltk.util.arguments import Arguments

if __name__ == '__main__':
    args = Arguments(logging)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # ---------------------------------
    # ----------- Cifar10CNN ----------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN.model")
    torch.save(Cifar10CNN().state_dict(), full_save_path)
    # ---------------------------------
    # --------- Cifar10ResNet ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10ResNet.model")
    torch.save(Cifar10ResNet().state_dict(), full_save_path)

    # ---------------------------------
    # -------- FashionMNISTCNN --------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN.model")
    torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # ------ FashionMNISTResNet -------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTResNet.model")
    torch.save(FashionMNISTResNet().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- Cifar100CNN ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar100ResNet.model")
    torch.save(Cifar100ResNet().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- Cifar100VGG ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar100VGG.model")
    torch.save(Cifar100VGG().state_dict(), full_save_path)