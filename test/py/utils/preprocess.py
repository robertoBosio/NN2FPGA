import time
import torch.utils.data
# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
import torchvision.datasets as datasets
# from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
# from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
import torchvision.transforms as transforms

import tiny_torch.benchmark.training_torch.anomaly_detection.common as com
from tqdm import tqdm
import numpy

def cifar_transform(is_training=True):
    if is_training:
      transform_list = [transforms.RandomHorizontalFlip(),
                        transforms.Pad(padding=4, padding_mode='reflect'),
                        transforms.RandomCrop(32, padding=0),
                        transforms.ToTensor(),
                        lambda x: x * 255.0 / 256.0,
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    else:
    #   transform_list = [transforms.ToTensor()]
      transform_list = [
          transforms.ToTensor(),
          lambda x: x * 255.0 / 256.0,
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

    transform_list = transforms.Compose(transform_list)

    return transform_list


def imgnet_transform(is_training=True):
    if is_training:
        transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.5,
                                                                    contrast=0.5,
                                                                    saturation=0.3),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    else:
        transform_list = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    return transform_list

def fft_transform_train(file_list, param):
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

    return list_to_vector_array(file_list, "calc...", param["n_mels"],
    param["frames"], param["n_fft"], param["hop_length"], param["power"])

def fft_transform_test(target_dir, param):
    machine_id_list = com.get_machine_id_list_for_test(target_dir)
    dataset = []
    anomaly = []
    print(machine_id_list)
    for id_str in machine_id_list:
        test_files, y_true = com.test_file_list_generator(target_dir, id_str, True)
        dataset.append(fft_transform_train(test_files, param))
        anomaly.append(y_true)
    return dataset, anomaly

# class HybridTrainPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         dali_device = "gpu"
#         self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
#         self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
#         self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#         self.coin = ops.CoinFlip(probability=0.5)
#         print('DALI "{0}" variant'.format(dali_device))

#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images, mirror=rng)
#         return [output, self.labels]


# class HybridValPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
#         super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
#                                     random_shuffle=False)
#         self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
#         self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             output_dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

#     def define_graph(self):
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images)
#         return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                           world_size=1,
                           local_rank=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                    data_dir=image_dir + '/train',
                                    crop=crop, world_size=world_size, local_rank=local_rank)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size, auto_reset=True)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size, auto_reset=True)
        return dali_iter_val


def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                            world_size=1, local_rank=0):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                                 pin_memory=True)
    return dataloader

def vww_transform(is_training=True, IMAGE_SIZE=128):
    if is_training:
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            # lambda x: x / 255.0,
            # normalize between 0 and 1
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            # lambda x: x / 255.0,
            # normalize between 0 and 1
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
