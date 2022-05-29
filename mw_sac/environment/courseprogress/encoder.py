from functools import partial
import torch.nn as nn
import torch
import numpy as np
import torchvision

LATENT_SIZE = 16
DEVICE=torch.device('cuda')

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
            nn.LeakyReLU(0.3, False),
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
        padding: int = 1,
        activation1: nn.Module = partial(nn.LeakyReLU, negative_slope=0.3, inplace=False),
        activation2: nn.Module = partial(nn.LeakyReLU, negative_slope=0.3, inplace=False),
        se: bool = True
    ):
        super(Block, self).__init__()
        expander = nn.Conv2d(in_channels=input_channels, out_channels=expanded_channels, kernel_size=1)
        depthwise = nn.Conv2d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=kernel_size, stride=2, padding=padding, groups=expanded_channels)
        unexpander = nn.Conv2d(in_channels=expanded_channels, out_channels=output_channels, kernel_size=1)

        self.block = nn.Sequential(
            expander,
            BatchNorm2d(num_features=expanded_channels),
            #nn.LeakyReLU(0.3, True),
            activation1(),
            depthwise,
            BatchNorm2d(num_features=expanded_channels),
            #nn.LeakyReLU(0.3, True),
            activation2(),
            SqueezeExcite(expanded_channels, channel_size//2) if se else nn.Identity(),
            unexpander,
            BatchNorm2d(num_features=output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BlockRes(nn.Module):
    def __init__(
        self,
        input_channels: int,
        expanded_channels: int,
        channel_size,
        output_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        activation1: nn.Module = partial(nn.LeakyReLU, negative_slope=0.3, inplace=False),
        activation2: nn.Module = partial(nn.LeakyReLU, negative_slope=0.3, inplace=False),
        se: bool = True
    ):
        super(BlockRes, self).__init__()
        expander = nn.Conv2d(in_channels=input_channels, out_channels=expanded_channels, kernel_size=1)
        depthwise = nn.Conv2d(in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=expanded_channels)
        unexpander = nn.Conv2d(in_channels=expanded_channels, out_channels=output_channels, kernel_size=1)

        self.block = nn.Sequential(
            expander,
            BatchNorm2d(num_features=expanded_channels),
            #nn.LeakyReLU(0.3, True),
            activation1(),
            depthwise,
            BatchNorm2d(num_features=expanded_channels),
            #nn.LeakyReLU(0.3, True),
            activation2(),
            SqueezeExcite(expanded_channels, channel_size) if se else nn.Identity(),
            unexpander,
            BatchNorm2d(num_features=output_channels),
        )
        self.shortcut = nn.Identity() if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)



class Swish(nn.Module):
  def __init__(self, size):
    super(Swish, self).__init__()
    self.β = nn.parameter.Parameter(
        torch.zeros((1, *size), device=DEVICE, dtype=torch.float),
        requires_grad=True
    )

  def forward(self, x: torch.Tensor):
    return x * (x * self.β).sigmoid()

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.checkpoint_segments = 16

        s_96 = partial(Swish, size=(96, 1, 1))
        s_240 = partial(Swish, size=(240, 1, 1))
        s_120 = partial(Swish, size=(120, 1, 1))
        s_144 = partial(Swish, size=(144, 1, 1))
        s_288 = partial(Swish, size=(288, 1, 1))
        s_576 = partial(Swish, size=(576, 1, 1))

        # Encoder inspired by mobilenetv3-small
        self.encoder = nn.Sequential(
            # 3 * 480 * 640
            # Downscale the image to 224x224 using bilinear algorithm
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True), # 3*224*224
            nn.Conv2d(3, 16, 3, 2, 1), # 16*112*112
            ##
            Block(16, 16, 112, 16), # 16*56*56
            Block(16, 72, 56, 24, se=False), # 24*28*28
            BlockRes(24, 88, 28, 24, se=False), # 24*28*28
            Block(24, 96, 28, 40, 5, 2, s_96, s_96), # 40*14*14
            BlockRes(40, 240, 14, 40, 5, 2, s_240, s_240), # 40*14*14
            BlockRes(40, 240, 14, 40, 5, 2, s_240, s_240), # 40*14*14
            BlockRes(40, 120, 14, 48, 5, 2, s_120, s_120), # 48*14*14
            BlockRes(48, 144, 14, 48, 5, 2, s_144, s_144), # 48*14*14
            Block(48, 288, 14, 96, 5, 2, s_288, s_288), # 96*7*7
            BlockRes(96, 576, 7, 96, 5, 2, s_576, s_576), # 96*7*7
            BlockRes(96, 576, 7, 96, 5, 2, s_576, s_576), # 96*7*7
            nn.Conv2d(96, 576, 1), # 576*7*7
            Swish(size=(576, 1, 1)),
            #nn.Conv2d(576, LATENT_SIZE*2, 7), # (LATENT_SIZE * 2)*1*1
            nn.Conv2d(576, LATENT_SIZE*2, 1), # (LATENT_SIZE * 2)*7*7
            nn.Conv2d(LATENT_SIZE*2, LATENT_SIZE*2, 7, groups=LATENT_SIZE*2), # (LATENT_SIZE * 2)*1*1
            nn.Flatten(1), # (LATENT_SIZE * 2)
        )
        #self.encoder = torch.utils.checkpoint.checkpoint_sequential(self.encoder, chunks)

        # 640 = 2 * 2 * 2 * 2 * 2 * 4 * 5
        # 480 = 2 * 2 * 2 * 2 * 2 * 3 * 5

        decoder_multiplier = 4

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (LATENT_SIZE, 1, 1)),
            nn.ConvTranspose2d(LATENT_SIZE, decoder_multiplier*256, 5, 1, 0, bias=False), # 1024 * 5 * 5
            BatchNorm2d(num_features=decoder_multiplier*256),
            nn.LeakyReLU(0.3, False),
            nn.ConvTranspose2d(decoder_multiplier*256, decoder_multiplier*128, (3, 4), (3, 4), 0, bias=False), # 512 * 15 * 20
            BatchNorm2d(num_features=decoder_multiplier*128),
            nn.LeakyReLU(0.3, False),
            nn.ConvTranspose2d(decoder_multiplier*128, decoder_multiplier*64, 4, 2, 1, bias=False), # 256 * 30 * 40
            BatchNorm2d(num_features=decoder_multiplier*64),
            nn.LeakyReLU(0.3, False),
            nn.ConvTranspose2d(decoder_multiplier*64, decoder_multiplier*32, 4, 2, 1, bias=False), # 128 * 60 * 80
            BatchNorm2d(num_features=decoder_multiplier*32),
            nn.LeakyReLU(0.3, False),
            nn.ConvTranspose2d(decoder_multiplier*32, decoder_multiplier*16, 4, 2, 1, bias=False), # 64 * 120 * 160
            BatchNorm2d(num_features=decoder_multiplier*16),
            nn.LeakyReLU(0.3, False),
            nn.ConvTranspose2d(decoder_multiplier*16, decoder_multiplier*8, 4, 2, 1, bias=False), # 32 * 240 * 320
            BatchNorm2d(num_features=decoder_multiplier*8),
            nn.LeakyReLU(0.3, False),
            nn.ConvTranspose2d(decoder_multiplier*8, 3, 4, 2, 1, bias=False), # 3 * 480 * 640
            nn.Tanh()
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            #eps = std.data.new(std.size()).normal_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
      #print(f'x size: {x.size()}')
      #encoded = torch.utils.checkpoint.checkpoint(self.encoder, x, preserve_rng_state=False)#, use_reentrant=False)
      encoded = self.encoder(x)
      #print(f'Encoded size: {encoded.size()}')
      mu_logvar = encoded.view(-1, 2, LATENT_SIZE)
      mu = mu_logvar[:, 0, :]
      logvar = mu_logvar[:, 1, :]
      return mu, logvar

    def decode(self, z):
      #return torch.utils.checkpoint.checkpoint(self.decoder, z, preserve_rng_state=False)#, use_reentrant=False)
      return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, n_samples):
      z = torch.randn((n_samples, LATENT_SIZE)).to(DEVICE)
      return self.decode(z)

vae = VAE()

with open('checkpoint51.pt', 'rb') as f:
    vae_savestate = torch.load(f)
    vae.load_state_dict(vae_savestate['state'])

encoder = vae.encoder
del vae
encoder = encoder.to(DEVICE)
encoder = encoder.train(False)
encoder = torch.jit.script(
    encoder,
    example_inputs=[(torch.ones((2, 3, 480, 640), device=DEVICE, dtype=torch.float),)]
)

normalize = torchvision.transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
)

transformations = nn.Sequential(
    torchvision.transforms.Resize(
        size=(480, 640)
    )
)

def frame_to_torch(frame: np.array):
    img = normalize(
        transformations(
            torch.from_numpy(
                np.transpose(
                    frame[:, :, ::-1],
                    (2, 0, 1)
                ).copy()
            )
        ).float().__div__(255.0)
    )
    return img

def encode(frame: np.array):
    with torch.no_grad():
        encoded = encoder(frame_to_torch(frame).to(DEVICE)[None, :, :, :])
        mu_logvar = encoded.view(-1, 2, LATENT_SIZE)
        mu = mu_logvar[:, 0, :]
        return mu