import logging
import os
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from .sampling import split_and_shuffle
from .sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, tiny_imagenet_iid, tiny_imagenet_noniid, imagenet_iid, imagenet_noniid
from tqdm import tqdm
from .train_test_split.train_test_split_cub200 import prepare_cub200
from .train_test_split.train_test_split_flowers102 import prepare_flowers102
from .train_test_split.train_test_split_pets import prepare_pets

from datasets import load_dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': label}


class ClsDatasetA(LightningDataModule):
    def __init__(self, data_dir, name='mnist',
                    train_split=0.8,
                    batch_size=32, train_shuffle=True,
                    width=224, height=224,
                    train_workers=4, val_workers=1, num_train_batch=None, num_val_batch=None, num_test_batch=None, max_length=512):
        super(ClsDatasetA, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.train_split = train_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.width = width
        self.height = height
        self.num_classes = 0
        self.num_train_batch = num_train_batch
        self.num_val_batch = num_val_batch
        self.num_test_batch = num_test_batch
        self.max_length = max_length

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def setup(self, stage):
        if self.name == 'pets':
            prepare_pets(self.data_dir)
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            raw_train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=apply_transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "test"), transform=apply_transform)
            
            self.num_classes = 37

        elif self.name == 'cub200':
            prepare_cub200(self.data_dir)
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            raw_train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=apply_transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "test"), transform=apply_transform)
            
            self.num_classes = 200

        elif self.name == 'flowers102':
            prepare_flowers102(self.data_dir)
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            train_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"), transform=apply_transform)
            val_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "val"), transform=apply_transform)
            test_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, "test"), transform=apply_transform)
            
            self.num_classes = 102

        elif self.name == 'cifar10':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True,
                                        transform=apply_transform)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True,
                                        transform=apply_transform)
            self.num_classes = 10
        elif self.name == 'cifar100':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_train_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True,
                                            transform=apply_transform)
            test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True,
                                            transform=apply_transform)
            self.num_classes = 100
        elif self.name == 'imagenet':
            apply_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.width),       # Cắt ảnh ngẫu nhiên theo tỉ lệ và resize về 224x224
                transforms.RandomHorizontalFlip(),      # Lật ngang ngẫu nhiên
                transforms.ToTensor(),                  # Chuyển đổi ảnh thành tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa giá trị
            ])
            raw_train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=apply_transform)
            test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=apply_transform)
            self.num_classes = 1000

        elif self.name == 'BoolQ':
            dataset = load_dataset("super_glue", "boolq", trust_remote_code=True)
        elif self.name == 'AG_News':
            dataset = load_dataset("ag_news", trust_remote_code=True)

        else:
            raise NotImplementedError
        if self.name != 'flowers102' and self.name != 'BoolQ' and self.name != 'AG_News':
            idx_train, idx_val = split_and_shuffle(raw_train_dataset, self.train_split)
            if self.num_train_batch != None: idx_train = idx_train[:self.batch_size*self.num_train_batch]
            if self.num_val_batch != None: idx_val = idx_val[:self.batch_size*self.num_val_batch]
            
            self.train_dataset = DatasetSplit(raw_train_dataset, idx_train)
            self.val_dataset = DatasetSplit(raw_train_dataset, idx_val)

            if self.num_test_batch != None: self.test_dataset = DatasetSplit(test_dataset, np.arange(self.batch_size*self.num_test_batch))
            else: self.test_dataset = DatasetSplit(test_dataset, np.arange(len(test_dataset)))
        elif self.name == 'flowers102':
            if self.num_train_batch != None: 
                self.train_dataset = DatasetSplit(train_dataset, np.arange(self.batch_size*self.num_train_batch))
            else: self.train_dataset = DatasetSplit(train_dataset, np.arange(len(train_dataset)))
            
            if self.num_val_batch != None: self.val_dataset = DatasetSplit(val_dataset, np.arange(self.batch_size*self.num_val_batch))
            else: self.val_dataset = DatasetSplit(val_dataset, np.arange(len(val_dataset)))

            if self.num_test_batch != None: self.test_dataset = DatasetSplit(test_dataset, np.arange(self.batch_size*self.num_test_batch))
            else: self.test_dataset = DatasetSplit(test_dataset, np.arange(len(test_dataset)))
        elif self.name == 'BoolQ':
            train_data = dataset["train"]
            val_data = dataset["validation"]  # BoolQ dùng "validation" thay vì "test"

            # Cắt train dataset
            if self.num_train_batch is not None:
                num_train_samples = min(self.batch_size * self.num_train_batch, len(train_data))
                train_data = train_data.select(range(num_train_samples))
            # Cắt val dataset
            if self.num_val_batch is not None:
                num_val_samples = min(self.batch_size * self.num_val_batch, len(val_data))
                val_data = val_data.select(range(num_val_samples))

            # Hàm tiền xử lý cho BoolQ
            def preprocess_function(examples):
                inputs = [f"{q} [SEP] {p}" for q, p in zip(examples["question"], examples["passage"])]
                encodings = self.tokenizer(inputs, truncation=True, padding="max_length", max_length=self.max_length)
                # Sửa từ "answer" thành "label"
                encodings["label"] = [1 if ans else 0 for ans in examples["label"]]
                return encodings

            self.train_dataset = train_data.map(preprocess_function, batched=True)
            self.val_dataset = val_data.map(preprocess_function, batched=True)
            self.train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            self.val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        elif self.name =='AG_News':
            def preprocess_function(examples):
                return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=self.max_length)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            
            train_data = tokenized_dataset["train"]
            val_data = tokenized_dataset["test"]

            # Train dataset
            if self.num_train_batch is not None:
                num_train_samples = self.batch_size * self.num_train_batch
                if num_train_samples < len(train_data):
                    self.train_dataset = train_data.select(range(num_train_samples))
                else:
                    self.train_dataset = train_data  # Dùng hết nếu vượt quá
            else:
                self.train_dataset = train_data  # Dùng toàn bộ

            # Val dataset
            if self.num_val_batch is not None:
                num_val_samples = self.batch_size * self.num_val_batch
                if num_val_samples < len(val_data):
                    self.val_dataset = val_data.select(range(num_val_samples))
                else:
                    self.val_dataset = val_data  # Dùng hết nếu vượt quá
            else:
                self.val_dataset = val_data  # Dùng toàn bộ

            
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle,
            num_workers=self.train_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.val_workers, pin_memory=True, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)

# Source: https://github.com/SLDGroup/GradientFilter-CVPR23/blob/main/classification/dataloader/pl_dataset.py
class ClsDatasetB(LightningDataModule):
    def __init__(self, data_dir, name='mnist',
                 num_partitions=2, iid=False, train_split=0.8,
                 batch_size=32, train_shuffle=True,
                 width=224, height=224,
                 train_workers=4, val_workers=1,
                 usr_group=None, partition=0, shards_per_partition=2, num_train_batch=None, num_val_batch=None, num_test_batch=None):
        super(ClsDatasetB, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.num_partitions = num_partitions
        self.iid = iid
        self.current_partition = partition
        if usr_group is not None:
            grp = np.load(usr_group)
            self.usr_group = {v[0]: v[1:] for v in grp}
            assert len(self.usr_group) == self.num_partitions
        else:
            self.usr_group = None
        self.train_split = train_split
        self.train_dataset = None
        self.val_dataset = None
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.width = width
        self.height = height
        self.shards_per_partition = shards_per_partition
        self.num_classes = 0

        self.num_train_batch = num_train_batch
        self.num_val_batch = num_val_batch
        self.num_test_batch = num_test_batch

    def switch_partition(self, partition):
        logging.info(f"Load partition {partition}")
        update_usr_group = self.usr_group is None
        if update_usr_group:
            logging.warn("No partition def found")
        if self.name == 'mnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.1307,), (0.3081,))])
            raw_dataset = datasets.MNIST(self.data_dir, train=True, download=True,
                                         transform=apply_transform)
            self.num_classes = 10
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = mnist_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = mnist_noniid(
                        raw_dataset, self.num_partitions)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'fmnist':
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.1307,), (0.3081,))])
            raw_dataset = datasets.FashionMNIST(self.data_dir, train=True, download=True,
                                                transform=apply_transform)
            self.num_classes = 10
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = mnist_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = mnist_noniid(
                        raw_dataset, self.num_partitions)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'cifar10':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True,
                                           transform=apply_transform)
            self.num_classes = 10
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = cifar_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = cifar_noniid(raw_dataset, self.num_partitions,
                                                  shards_per_user=self.shards_per_partition)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'cifar100':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True,
                                            transform=apply_transform)
            self.num_classes = 100
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = cifar_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = cifar_noniid(raw_dataset, self.num_partitions,
                                                  shards_per_user=self.shards_per_partition)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'tiny-imagenet':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((self.width, self.height)),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            raw_dataset = datasets.ImageFolder(
                self.data_dir, transform=apply_transform)
            self.num_classes = 1000
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = tiny_imagenet_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = tiny_imagenet_noniid(
                        raw_dataset, self.num_partitions)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        elif self.name == 'imagenet':
            # apply_transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Resize((self.width, self.height)),
            #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # raw_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=apply_transform)
            raw_dataset = datasets.ImageFolder(
                os.path.join(self.data_dir, 'train'))
            self.num_classes = 1000
            if self.usr_group is None:
                if self.iid:
                    self.usr_group = imagenet_iid(
                        raw_dataset, self.num_partitions)
                else:
                    self.usr_group = imagenet_noniid(raw_dataset, self.num_partitions,
                                                     shards_per_user=self.shards_per_partition)
                for usr in self.usr_group.keys():
                    np.random.shuffle(self.usr_group[usr])
        else:
            raise NotImplementedError
        logging.info(f"Setup Data Partition{partition}")
        idx = self.usr_group[partition].astype(int)
        idx_train = idx[:int(self.train_split * len(idx))]
        idx_val = idx[len(idx_train):]

        if self.num_train_batch != None: idx_train = idx_train[:self.batch_size*self.num_train_batch]
        if self.num_val_batch != None: idx_val = idx_val[:self.batch_size*self.num_val_batch]
        
        if self.name != 'imagenet':
            self.train_dataset = DatasetSplit(raw_dataset, idx_train)
            self.val_dataset = DatasetSplit(raw_dataset, idx_val)
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            self.train_dataset = DatasetSplit(raw_dataset, idx_train, transform=train_transform)
            self.val_dataset = DatasetSplit(raw_dataset, idx_val, transform=val_transform)
        if update_usr_group:
            logging.warn(
                "Calculating KL divergence for new generated partition")
            class_cnts = np.zeros((self.num_partitions, self.num_classes))
            for p in range(self.num_partitions):
                idx = self.usr_group[p].astype(int)
                for iidx in tqdm(idx):
                    label = raw_dataset.targets[iidx]
                    class_cnts[p, label] += 1
            prob = class_cnts / np.sum(class_cnts, axis=1, keepdims=True)
            kl = np.sum(prob[1:] * np.log(prob[1:] /
                        (prob[0] + 1e-9) + 1e-9), axis=1)
            kl_str = "_".join([f"{k:.2f}" for k in kl])
            grp = np.array([np.concatenate([[k], v])
                           for k, v in self.usr_group.items()])
            save_path = os.path.join(self.data_dir, f"usr_group_{kl_str}.npy")
            np.save(save_path, grp)
            logging.warn(f"Partition def saved to {save_path}")
        logging.info(f"Setup Data Partition{partition}")

    def setup(self, stage):
        self.switch_partition(self.current_partition)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle,
                          num_workers=self.train_workers, pin_memory=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.val_workers, pin_memory=False, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)
    
class ClsDataset(LightningDataModule):
    def __init__(self, setup, data_dir, name='mnist',
                 num_partitions=2, iid=False, train_split=0.8,
                 batch_size=32, train_shuffle=True,
                 width=224, height=224,
                 train_workers=4, val_workers=1,
                 usr_group=None, partition=0, shards_per_partition=2, num_train_batch=None, num_val_batch=None, num_test_batch=None, max_length=512):
        super(ClsDataset, self).__init__()
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.setup_type = setup

        if self.setup_type == 'A':
            self.datamodule = ClsDatasetA(data_dir, name, train_split, batch_size, train_shuffle,
                                       width, height, train_workers, val_workers, num_train_batch, num_val_batch, num_test_batch, max_length)
        elif self.setup_type == 'B':
            self.datamodule = ClsDatasetB(data_dir, name, num_partitions, iid, train_split,
                                       batch_size, train_shuffle, width, height,
                                       train_workers, val_workers, usr_group, partition, shards_per_partition, num_train_batch, num_val_batch, num_test_batch)
        else:
            raise ValueError(f"Invalid setup value: {setup}. It must be 'A' or 'B'.")
    def set_tokenizer(self, tokenizer):
        self.datamodule.set_tokenizer(tokenizer)
    def setup(self, stage):
        self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def predict_dataloader(self):
        return self.datamodule.predict_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
