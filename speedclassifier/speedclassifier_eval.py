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
import sys

DEVICE=torch.device('cuda')
CPU_DEVICE = torch.device('cpu')

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



print("selected camera:", sys.argv[1])

cap = cv2.VideoCapture(int(sys.argv[1]))

classifier = SpeedClassifier().to(device=DEVICE)
classifier.load_state_dict(
    torch.load('speedclassifier1.pytorch')['model']
)
classifier = classifier.train(False)


classifier = torch.jit.script(
    classifier#,
    #example_inputs=[(torch.ones((2, 3, 20, 16), device=DEVICE, dtype=torch.float),)]
)

normalize = torchvision.transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
)

#TABLE_NAME = 'auto_labeled14'

classifier = classifier.to(DEVICE)

def norm(frame_part: np.ndarray):
    img = normalize(
        torch.from_numpy(
            np.transpose(
                frame_part[:, :, ::-1],
                (2, 0, 1)
            ).copy()
        ).float().__div__(255.0)
    )[None,...]
    return img

# with sqlite3.connect('frames.sqlite3') as db:
#     db.execute(f'CREATE TABLE IF NOT EXISTS {TABLE_NAME} (label INT, image BLOB NOT NULL);')
if True:
    # loop over
    while True:
        # read frames from stream
        #frame = stream.read()
        ret, vidframe = cap.read()
        if not ret:
            break

        # check for frame if Nonetype
        # if frame is None:
        #     break

        # {do something with the frame here}
        frame = vidframe[380:400, 515:563]

        cv2.imshow('Frame', frame)

        frame = norm(frame)

        img1, img2, img3 = frame[:, :, :, 0:16], frame[:, :, :, 16:32], frame[:, :, :, 32:48]

        # part1, part2, part3 = frame[:, :, 0:16], frame[:, :, 16:32], frame[:, :, 32:48]
        # print(part1.shape, part2.shape, part3.shape)
        # img1, img2, img3 = norm(part1), norm(part2), norm(part3)
        # print(img1.shape, img2.shape, img3.shape)

        input = torch.cat((img1, img2, img3), dim=0).to(DEVICE)

        #print(input.size())
        output = classifier(input)
        [_, values] = torch.max(output, dim=1)#.to(device=CPU_DEVICE)
        values = values.to(device=CPU_DEVICE)
        [p1, p2, p3] = values
        p1, p2, p3 = p1.item(), p2.item(), p3.item()

        print(p1 * 100 + p2 * 10 + p3)

        #label = -1 if r == 10 else ((0 if l == 10 else l)*10 + r)
        
        # db.execute(
        #     f'INSERT INTO {TABLE_NAME} (image, label) VALUES(?, ?)',
        #     (
        #         bytes(cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1]),
        #         label
        #     )
        # )

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break