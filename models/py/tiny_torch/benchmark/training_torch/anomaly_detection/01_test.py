"""
 @file   01_test.py
 @brief  Script for test
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
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
import eval_functions_eembc
import torch
import torch.nn as nn
from torch_model import Autoencoder
from torch.utils.data import Dataset, DataLoader
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################

class CustomDatasetTest(Dataset):
    def __init__(self, data_array):
        self.data = torch.from_numpy(data_array).float()
        self.data = self.data.unsqueeze(2)
        self.data = self.data.unsqueeze(3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

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

########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # set model path
        # model_file = "{model}/model_{machine_type}.hd5f".format(model=param["model_directory"],
        #                                                         machine_type=machine_type)
        
        model_file = f"{param['model_directory']}/checkpoint_quant_fx.t7"
        # load model file
        model = Autoencoder(param["feature"]["n_mels"] * param["feature"]["frames"])
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(param["model_directory"]))
            sys.exit(-1)
        model.load_state_dict(torch.load(model_file)["model_state_dict"])
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = com.get_machine_id_list_for_test(target_dir)

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                     result=param["result_directory"],
                                                                                     machine_type=machine_type,
                                                                                     id_str=id_str)
            anomaly_score_list = []
            test_data = list_to_vector_array(test_files,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])
            print(test_data.shape())
            test_dataset = CustomDatasetTest(test_data)
            test_loader = DataLoader(dataset=test_dataset, batch_size=196, shuffle=False, num_workers=2)

            results = []
            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            with tqdm(test_loader, unit="batch") as tepoch:
                for data in tepoch:
                    data_tensor = data.cuda()
                    pred = model(data_tensor)
                    errors = torch.mean(torch.square(data_tensor - pred), axis=1)
                    y_pred = torch.mean(errors)
                    results.append(y_pred)
            
            y_pred = torch.stack(results, axis=0)
            y_pred = y_pred.cpu().detach().numpy()
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
            performance.append([auc, p_auc])
            acc_eembc = eval_functions_eembc.calculate_ae_accuracy(y_pred, y_true)
            pr_acc_eembc = eval_functions_eembc.calculate_ae_pr_accuracy(y_pred, y_true)
            auc_eembc = eval_functions_eembc.calculate_ae_auc(y_pred, y_true, "dummy")
            com.logger.info("EEMBC Accuracy: {}".format(acc_eembc))
            com.logger.info("EEMBC Precision/recall accuracy: {}".format(pr_acc_eembc))
            com.logger.info("EEMBC AUC: {}".format(auc_eembc))
            com.logger.info("AUC : {}".format(auc))
            com.logger.info("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        # calculate averages for AUCs and pAUCs
        averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        com.save_csv(save_file_path=result_path, save_data=csv_lines)
