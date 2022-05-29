import random
import sqlite3
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from functools import partial
import torchvision
from typing import Iterator
import numpy as np
import torch.nn.functional as F
from typing import Tuple

DEVICE=torch.device('cuda')
BATCH_SIZE=64

BatchNorm2d = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels,
        channelsize
    ):
        super(SqueezeExcite, self).__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(channelsize),
            nn.Flatten(),
            nn.Linear(channels, channels//4),
            nn.LeakyReLU(0.3, True),
            nn.Linear(channels//4, channels),
            nn.Sigmoid(),
            nn.Unflatten(1, (channels, 1, 1))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class Block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        expanded_channels: int,
        channel_size,
        output_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super(Block, self).__init__()
        expander = nn.Conv2d(in_channels=input_channels, out_channels=expanded_channels, kernel_size=1)
        depthwise = nn.Conv2d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=kernel_size, stride=2, padding=padding, groups=expanded_channels)
        unexpander = nn.Conv2d(in_channels=expanded_channels, out_channels=output_channels, kernel_size=1)

        self.block = nn.Sequential(
            expander,
            BatchNorm2d(num_features=expanded_channels),
            nn.LeakyReLU(0.3, True),
            depthwise,
            BatchNorm2d(num_features=expanded_channels),
            nn.LeakyReLU(0.3, True),
            SqueezeExcite(expanded_channels, channel_size//2),
            unexpander,
            BatchNorm2d(num_features=output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpeedClassifier(nn.Module):
    def __init__(self):
        super(SpeedClassifier, self).__init__()
        C = 64

        #self.layer_1 = Block(3, 3*C, 16, C-1, 3, 1)

        self.block = nn.Sequential(
            # 3@20x16
            Block(3, 3*C, 16, C, 3, 1), # 64@10x8
            Block(C, 3*C, 8, C, 3, 1), # 64@5x4
            Block(C, 6*C, 4, 2*C, 3, 1), # 128@3x2
            nn.Conv2d(2*C, 4*C, (3, 2)), # 256@1x1
            nn.Flatten(),
            nn.Dropout(0.25, True),
            nn.LeakyReLU(0.3, True),
            nn.Linear(4*C, 4*C),
            nn.LeakyReLU(0.3, True),
            nn.Linear(4*C, 10),
            ##nn.Sigmoid()
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = 64
        # print(f"Size: {x.size()}")
        # x = Block(3, 3*C, 16, C, 3, 1).cuda()(x)
        # print(f"Size2: {x.size()}")
        # x = Block(C, 3*C, 8, C, 3, 1).cuda()(x)
        # print(f"Size3: {x.size()}")
        # x = Block(C, 6*C, 4, 2*C, 3, 1).cuda()(x)
        # print(f"Size4: {x.size()}")
        # x = nn.Conv2d(2*C, 4*C, (3, 2)).cuda()(x)
        # print(f"Size5: {x.size()}")

        return self.block(
            x
        )
        # return self.block(
        #     torch.cat(
        #         (
        #             self.layer_1(x),
        #             side,
        #         ),
        #         dim=1
        #     )
        # )


class FrameIter(IterableDataset):
    def __init__(self, db: sqlite3.Connection):
        super(FrameIter, self).__init__()
        self.db = db
        self.transformations = nn.Sequential(
            torchvision.transforms.Resize(
                size=(20, 48)
            ),
            torchvision.transforms.RandomEqualize(),
            #torchvision.transforms.RandomAutocontrast(),
            #torchvision.transforms.RandomAdjustSharpness(2),
            #torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            torchvision.transforms.ColorJitter(saturation=1.0, hue=.5)
        )
        self.normalize = torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )

    def get_iterator(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        def getiterator():
            while True:
                [label_index] = random.sample(range(1, 4), 1)
                [label] = random.sample(range(4 if label_index == 1 else 10), 1)
                for (imgbuf,) in self.db.execute(f'SELECT image FROM speed_labeled WHERE label{label_index} = ? ORDER BY RANDOM() LIMIT 1;', (label,)):
                    yield label_index, label, imgbuf

        return iter((t[0], t[1], self.parse_image(t[2], t[0])) for t in getiterator())

    def parse_image(self, buf, label_index):
        buf = np.asarray(bytearray(buf), dtype='uint8')
        img = self.normalize(
            self.transformations(
                torch.from_numpy(
                    np.transpose(
                        cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1],
                        (2, 0, 1)
                    ).copy()
                )
            ).float().__div__(255.0)
        )

        #print(f"Pre-downscale size: {img.size()}")

        if label_index == 1:
            img = img[:, :, 0:16]
        elif label_index == 2:
            img = img[:, :, 16:32]
        else:
            img = img[:, :, 32:48]

        #print(f"After-downscale size: {img.size()}\n")

        return img

    def __iter__(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        return self.get_iterator()


with sqlite3.connect('../frames.sqlite3') as db:
    framedataset = FrameIter(db)
    framedataloader = DataLoader(
        framedataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    model = SpeedClassifier().to(device=DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        eps=1e-3,
        amsgrad=True,
        weight_decay=3e-1
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.7)
    #criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 2):
        for batch in range(1, 1001):
            _, labels, imgs = next(iter(framedataloader))
            labels = labels.to(device=DEVICE, dtype=torch.int64)
            imgs = imgs.to(device=DEVICE)

            optimizer.zero_grad()
            predicted_labels = model(imgs)
            # predicted_labels1 = model(imgs, 1)
            # predicted_labels2 = model(imgs, -1)
            #predicted_labels1 = predicted_labels[:BATCH_SIZE]
            #predicted_labels2 = predicted_labels[BATCH_SIZE:]
            #loss = (criterion(predicted_labels1, labels1) + criterion(predicted_labels1, labels2))
            # l1 = F.nll_loss(predicted_labels1, labels1, reduction='none', weight=loss_weight)
            # l2 = F.nll_loss(predicted_labels2, labels2, reduction='none', weight=loss_weight)
            # loss = torch.where(l1 > torch.tensor([0.01], dtype=torch.float, device=DEVICE), l1, torch.zeros_like(l1, device=DEVICE)).mean() + torch.where(l2 > torch.tensor([0.01], dtype=torch.float, device=DEVICE), l2, torch.zeros_like(l2, device=DEVICE)).mean()
            loss = F.nll_loss(predicted_labels, labels)
            # loss = F.nll_loss(predicted_labels1, labels1) + \
            #     F.nll_loss(predicted_labels2, labels2)
            print("batch:", batch, "\nepoch:", epoch, "\nloss:", loss.item())

            _, predicted = torch.max(predicted_labels, 1)

            print("good predictions", torch.sum(predicted == labels))

            loss.backward()
            optimizer.step()
        scheduler.step()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        "speedclassifier1.pytorch")
