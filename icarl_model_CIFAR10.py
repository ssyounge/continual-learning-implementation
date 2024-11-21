import sys
sys.path.insert(0, '/kaggle/input/pycil/PyCIL')

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU. Make sure the Colab runtime is set to GPU.")





import logging
import numpy as np
import torch
import copy
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from utils.inc_net import IncrementalNet
from tqdm import tqdm

EPSILON = 1e-8

init_epoch = 20
#init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 20
#epochs = 170
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2

results = []
torch.backends.cudnn.benchmark = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=0, input_channels=1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 64 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks - 1)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        return features

def resnet32(input_channels=3):
    return ResNet(BasicBlock, [5, 5, 5], input_channels=input_channels)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def tensor2numpy(tensor):
    return tensor.cpu().numpy()

class BaseLearner:
    def __init__(self, args):
        self._memory_size = args.get("memory_size", None)
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args.get("device", torch.device("cpu"))
        if isinstance(self._device, list):
            self._device = self._device[0]
        self._multiple_gpus = args.get("multiple_gpus", False)

class DataManager:
    def __init__(self, dataset_name="mnist", batch_size=128, num_workers=8, class_increment=None):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 데이터셋별로 class_increment 값 자동 설정
        if self.dataset_name in ["mnist", "cifar10"]:
            self.class_increment = 2
        elif self.dataset_name in ["cifar100", "imagenet"]:
            self.class_increment = 5
        else:
            if class_increment is not None:
                self.class_increment = class_increment
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

        if self.dataset_name == "mnist":
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)), 
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

        elif self.dataset_name == "cifar10":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
            self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

        elif self.dataset_name == "cifar100":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            self.train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transform)
            self.test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transform)

        elif self.dataset_name == "imagenet":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.train_dataset = datasets.ImageFolder(root='/path/to/imagenet/train', transform=self.transform)
            self.test_dataset = datasets.ImageFolder(root='/path/to/imagenet/val', transform=self.transform)

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self._current_task = -1
        self._increment = self.class_increment
        self._class_order = list(range(len(self.train_dataset.classes)))

    def get_task_size(self, task_idx):
        return self.class_increment

    def new_task(self):
        self._current_task += 1
        start_class = self._current_task * self._increment
        end_class = start_class + self._increment
        self._current_classes = self._class_order[start_class:end_class]

    def get_dataset(self, class_indices=None, source="train", appendent=None):
        if class_indices is None:
            class_indices = self._current_classes
        dataset = self.train_dataset if source == "train" else self.test_dataset
        indices = [i for i, (_, label) in enumerate(dataset) if label in class_indices]
        subset = Subset(dataset, indices)
        if appendent:
            subset = torch.utils.data.ConcatDataset([subset, appendent])
        return subset

    def _get_class_dataset(self, class_idx):
        # DataManager의 get_dataset
        return self.data_manager.get_dataset(class_indices=np.array([class_idx]), source="train")    


class MyIncrementalNet(IncrementalNet):
    def __init__(self, args, pretrained=False, feature_dim=512):
        super().__init__(args, pretrained)
        self._feature_dim = feature_dim  # 내부에서 사용할 변수로 설정

    @property
    def feature_dim(self):
        return self._feature_dim  # 외부에서 접근 가능한 속성으로 정의

class IncrementalNet(nn.Module):
    def __init__(self, num_classes=0, input_channels=3):
        super().__init__()
        self.convnet = resnet32(input_channels=input_channels)
        self.feature_dim = self.convnet.feature_dim
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        features = self.convnet(x)
        logits = self.fc(features)
        return {"features": features, "logits": logits}

    def update_fc(self, num_classes):
        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def tensor2numpy(tensor):
    return tensor.cpu().numpy()

class iCaRL:
    def __init__(self, args, data_manager):
        self.args = args
        self.data_manager = data_manager
        self._device = args['device']
        self.dataset_name = data_manager.dataset_name

        if self.dataset_name == "mnist":
            input_channels = 1
        else:
            input_channels = 3

        self._network = IncrementalNet(num_classes=0, input_channels=input_channels)
        self._network.to(self._device)
        self._network.apply(init_weights)

        self.scaler = torch.amp.GradScaler('cuda')
        self._old_network = None
        self.exemplar_set = []
        self._known_classes = 0
        self._cur_task = -1
        self.T = args.get('T', 2)
        self._total_classes = 0
        self._memory_size = args.get("memory_size", 2000)
        self.batch_size = args.get('batch_size', 128)
        self.num_workers = args.get('num_workers', 4)

    def _get_class_dataset(self, class_idx):
        return self.data_manager.get_dataset(class_indices=[class_idx], source="train")


    def after_task(self):
        self._old_network = copy.deepcopy(self._network)
        self._old_network.eval()
        for param in self._old_network.parameters():
            param.requires_grad = False
        self._known_classes = self._total_classes

    def build_rehearsal_memory(self):
        self.exemplar_set = []
        m = self._memory_size // self._total_classes

        for class_idx in range(self._total_classes):
            class_dataset = self._get_class_dataset(class_idx)
            loader = DataLoader(class_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            features_list = []
            for inputs, _ in loader:
                inputs = inputs.to(self._device)
                with torch.no_grad():
                    features = self._network(inputs)["features"]
                features_list.append(features.cpu())
            
            features = torch.cat(features_list, dim=0)
            features = F.normalize(features, p=2, dim=1)
            class_mean = torch.mean(features, dim=0)

            exemplar_indices = []
            exemplar_features = []
            selected_indices = set()
            for k in range(m):
                if k > 0:
                    S = torch.stack(exemplar_features).sum(dim=0)
                else:
                    S = torch.zeros_like(class_mean)
                mu = class_mean
                distances = ((mu - (S + features) / (k + 1)).pow(2).sum(dim=1)).sqrt()
                distances[list(selected_indices)] = float('inf') 
                idx = torch.argmin(distances)
                exemplar_indices.append(idx.item())
                exemplar_features.append(features[idx])
                selected_indices.add(idx.item())
            self.exemplar_set.append((class_idx, exemplar_indices))

    def incremental_train(self):
        self.data_manager.new_task()
        self._cur_task += 1
        self._total_classes = self._known_classes + self.data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on classes {self._known_classes} to {self._total_classes}")

        appendent = self._get_memory() if self._cur_task > 0 else None
        train_dataset = self.data_manager.get_dataset(source="train", appendent=appendent)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args['batch_size'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        logging.info(f"Starting Task {self._cur_task + 1} with batch size: {self.train_loader.batch_size}")

        test_dataset = self.data_manager.get_dataset(source="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args['batch_size'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(),
                                lr=self.args['init_lr'],
                                momentum=0.9,
                                weight_decay=self.args['init_weight_decay'])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args['init_milestones'],
                gamma=self.args['init_lr_decay'],
            )
            self._init_train(self.train_loader, self.test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(),
                                lr=self.args['lrate'],
                                momentum=0.9,
                                weight_decay=self.args['weight_decay'])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args['milestones'],
                gamma=self.args['lrate_decay'],
            )
            self._update_representation(self.train_loader, self.test_loader, optimizer, scheduler)

        logging.info(f"Finished Task {self._cur_task}")

        self.build_rehearsal_memory()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network:
            self._old_network.to(self._device)

        optimizer = optim.SGD(self._network.parameters(), ...)
        scheduler = optim.lr_scheduler.MultiStepLR(...)

        total_epochs = init_epoch if self._cur_task == 0 else epochs

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:  # 프로파일링 시작
            prog_bar = tqdm(total=total_epochs, leave=True, ncols=100)
            for epoch in range(1, total_epochs + 1):
                self._network.train()
                losses, correct, total_samples = 0.0, 0, 0

                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda'):
                        logits = self._network(inputs)["logits"]
                        loss = F.cross_entropy(logits, targets)

                    # Mixed Precision Training
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    losses += loss.item()
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets).cpu().sum()
                    total_samples += len(targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total_samples, decimals=2)

                test_acc = self._compute_accuracy(self._network, test_loader) if epoch % 5 == 0 else None
                prog_bar.set_description(f"Epoch {epoch}/{total_epochs} => Loss: {losses / len(train_loader):.3f}, Train_accy: {train_acc:.2f}")
                prog_bar.update(1)

                torch.cuda.empty_cache(

            prog_bar.close()
        prof.export_chrome_trace("train_profiler.json")

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        total_epochs = self.args['init_epoch']
        self._network.to(self._device)
        self.scaler = torch.amp.GradScaler('cuda')
        prog_bar = tqdm(range(total_epochs), total=total_epochs, leave=True, ncols=100)

        for epoch in prog_bar:
            self._network.train()
            losses, correct, total = 0.0, 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0 or epoch == total_epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                description = f"Epoch {epoch+1}/{total_epochs} => Loss: {losses / len(train_loader):.3f}, Train_accy: {train_acc:.2f}, Test_accy: {test_acc:.2f}"
            else:
                description = f"Epoch {epoch+1}/{total_epochs} => Loss: {losses / len(train_loader):.3f}, Train_accy: {train_acc:.2f}"
            
            prog_bar.set_description(description)
        prog_bar.close()

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        lambda_kd = 1.0
        total_epochs = self.args['epochs']
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        self.scaler = torch.amp.GradScaler('cuda')
        prog_bar = tqdm(range(total_epochs), total=total_epochs, leave=True, ncols=100)

        for epoch in prog_bar:
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda'):
                    outputs = self._network(inputs)['logits']
                    loss_class = F.cross_entropy(outputs, targets)

                    if self._old_network is not None:
                        with torch.no_grad():
                            old_outputs = self._old_network(inputs)['logits']
                        T = self.T
                        kd_loss = F.kl_div(
                            F.log_softmax(outputs[:, :self._known_classes] / T, dim=1),
                            F.softmax(old_outputs / T, dim=1),
                            reduction='batchmean'
                        ) * (T ** 2)
                    else:
                        kd_loss = 0.0

                    loss = loss_class + lambda_kd * kd_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                losses += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            scheduler.step()
            train_acc = np.around(correct * 100 / total, decimals=2)

            if epoch % 5 == 0 or epoch == total_epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                description = f"Epoch {epoch+1}/{total_epochs} => Loss: {losses / len(train_loader):.3f}, Train_accy: {train_acc:.2f}, Test_accy: {test_acc:.2f}"
            else:
                description = f"Epoch {epoch+1}/{total_epochs} => Loss: {losses / len(train_loader):.3f}, Train_accy: {train_acc:.2f}"

            prog_bar.set_description(description)
        prog_bar.close()

    def _compute_accuracy(self, model, test_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = model(inputs)["logits"]
                _, preds = torch.max(outputs, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        return (correct / total) * 100

    def _get_memory(self):
        if not self.exemplar_set:
            return None
        exemplar_datasets = []
        for class_idx, exemplar_indices in self.exemplar_set:
            class_dataset = self._get_class_dataset(class_idx)
            exemplar_data = Subset(class_dataset, exemplar_indices)
            exemplar_datasets.append(exemplar_data)
        memory_dataset = torch.utils.data.ConcatDataset(exemplar_datasets)
        return memory_dataset

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

def main():
    memory_budgets = [500,1000,1500,2000]
    exemplar_sizes = [20,50,100,200]
    temperatures = [2,5,10]
    batch_sizes = [32,64,128,256]

    dataset_name = "cifar10"

    results = []

    for memory_budget in memory_budgets:
        for exemplar_size in exemplar_sizes:
            for temperature in temperatures:
                for batch_size in batch_sizes:
                    tqdm.write(f"\n==> Running experiment with memory budget: {memory_budget}, exemplar size: {exemplar_size}, "
                               f"T: {temperature}, batch size: {batch_size}")

                    args = {
                        'init_epoch': init_epoch,
                        'init_lr': init_lr,
                        'init_milestones': init_milestones,
                        'init_lr_decay': init_lr_decay,
                        'init_weight_decay': init_weight_decay,
                        'epochs': epochs,
                        'lrate': lrate,
                        'milestones': milestones,
                        'lrate_decay': lrate_decay,
                        'weight_decay': weight_decay,
                        'num_workers': num_workers,
                        'T': temperature,
                        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        'convnet_type': "resnet32",
                        'dataset': dataset_name,
                        'samples_per_class': exemplar_size,
                        'memory_size': memory_budget,
                        'batch_size': batch_size
                    }

                    data_manager = DataManager(dataset_name=dataset_name, batch_size=batch_size, num_workers=num_workers)
                    icarl_model = iCaRL(args, data_manager)

                    num_tasks = len(data_manager._class_order) // data_manager.class_increment
                    accuracies = []
                    for task_idx in range(num_tasks):
                        icarl_model.incremental_train()
                        icarl_model.after_task()
                        test_accuracy = icarl_model._compute_accuracy(icarl_model._network, icarl_model.test_loader)
                        accuracies.append(test_accuracy)

                    avg_accuracy = np.mean(accuracies)
                    results.append({
                        'memory_budget': memory_budget,
                        'exemplar_size': exemplar_size,
                        'temperature': temperature,
                        'batch_size': batch_size,
                        'class_increment': data_manager.class_increment,
                        'average_accuracy': avg_accuracy
                    })
                    tqdm.write(f"Memory Budget: {memory_budget}, Exemplar Size: {exemplar_size}, T: {temperature}, "
                               f"Batch Size: {batch_size}, Class Increment: {data_manager.class_increment}, Average Accuracy: {avg_accuracy:.2f}%")

if __name__ == "__main__":
    main()
