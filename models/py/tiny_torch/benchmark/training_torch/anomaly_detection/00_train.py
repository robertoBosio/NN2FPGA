"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from brevitas.export import export_onnx_qcdq
from tqdm import tqdm
import common as com
from torch_model import Autoencoder
from sklearn import metrics
import eval_functions_eembc
from torch.utils.data import Dataset, DataLoader
########################################################################
########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

class CustomDatasetTrain(Dataset):
    def __init__(self, data_array):
        self.data = torch.from_numpy(data_array).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CustomDatasetTest(Dataset):
    def __init__(self, data_array):
        self.data = torch.from_numpy(data_array).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files
########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hd5f".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        train_data = list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"])

        anomaly_score_list = []

        machine_id_list = com.get_machine_id_list_for_test(target_dir)
        test_files, y_true = com.test_file_list_generator(target_dir, machine_id_list[0], mode)
        test_data = list_to_vector_array(test_files,
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])


        # train model
        print("============== MODEL TRAINING ==============")
        
        model = Autoencoder(param["feature"]["n_mels"] * param["feature"]["frames"])
        loss_function = nn.MSELoss()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Define the loss function and optimizer
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4
        )

        train_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=lambda epoch: 0.99**epoch)

        # Number of training epochs
        epochs = param["fit"]["epochs"]
        batch_size = param["fit"]["batch_size"]
        criterion = torch.nn.MSELoss()
        #train_data = train_data.reshape((7000, 196, 640))

        train_dataset = CustomDatasetTrain(train_data)
        test_dataset = CustomDatasetTest(test_data)

        # Create DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, num_workers=2)
        test_loader = DataLoader(dataset=test_dataset, batch_size=196, shuffle=False, num_workers=2)

        # Training loop
        epochs = 100
        
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()

            # Adjust batch size during training
            #with tqdm(train_loader, unit="batch") as tepoch:
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch_data in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    #batch_data = train_data[batch_start:batch_start + batch_size]
                    #batch_data_tensor = torch.from_numpy(batch_data).float()
                    batch_data_exp = batch_data.unsqueeze(2)
                    batch_data_tensor = batch_data_exp.cuda()
                    optimizer.zero_grad()

                    outputs = model(batch_data_tensor)
                    loss = criterion(outputs, batch_data_tensor)
                    loss.backward()
                    optimizer.step()
                    #tepoch.set_postfix({"Acc": f"{100.0 * train_correct / total:.2f}%", "Loss": f"{loss.item():.4f}"})
                    tepoch.set_postfix({"Loss": f"{loss.item():.4f}"})
                    #print(f"Epoch [{epoch + 1}/{param['fit']['epochs']}], Loss: {loss.item()} LR: {optimizer.param_groups[0]['lr']:0.6f}")
            
            train_scheduler.step()
            
            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            #y_pred = [0. for k in test_files]
            y_pred = torch.zeros([len(test_files)])
            
            results = []
            model.eval()
            with tqdm(test_loader, unit="batch") as tepoch:
                for data in tepoch:
                    data_tensor = data.cuda()
                    data_tensor = data_tensor.unsqueeze(2)
                    pred = model(data_tensor)
                    errors = torch.mean(torch.square(data_tensor - pred), axis=1)
                    y_pred = torch.mean(errors)
                    results.append(y_pred)
                    #anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            if mode:
                # append AUC and pAUC to lists
                y_pred = torch.stack(results, axis=0)
                y_pred = y_pred.cpu().detach().numpy()
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

                    # acc_eembc = eval_functions_eembc.calculate_ae_accuracy(y_pred, y_true)
                    # pr_acc_eembc = eval_functions_eembc.calculate_ae_pr_accuracy(y_pred, y_true)
                    # auc_eembc = eval_functions_eembc.calculate_ae_auc(y_pred, y_true, "dummy")

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        # Save the trained model
        model = model.cpu()
        export_onnx_qcdq(model, torch.randn(1, 640, 1), export_path=f'model/model_{machine_type}.onnx')
        torch.save(model.state_dict(), model_file_path)
        # visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        # visualizer.save_figure(history_img)
        # com.logger.info("save_model -> {}".format(model_file_path))
        # model = keras_model.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])
        # model.summary()
        #
        #model.compile(**param["fit"]["compile"])
        #history = model.fit(train_data,
        #                    train_data,
        #                    epochs=param["fit"]["epochs"],
        #                    batch_size=param["fit"]["batch_size"],
        #                    shuffle=param["fit"]["shuffle"],
        #                    validation_split=param["fit"]["validation_split"],
        #                    verbose=param["fit"]["verbose"])
        #
        #visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        #visualizer.save_figure(history_img)
        #model.save(model_file_path)
        #com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")
