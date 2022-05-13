import torch
from torch import nn
from unet import UNet


class UnetEncoderSSLRLP(nn.Module):
    def __init__(
        self,
        patch_dim_x=28,
        patch_dim_z=32,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        reduction_factor=64, # for dense connections from conv layer
        dropout=0,
        n_patches=2
    ):
        super().__init__()

        self.patch_dim_x = patch_dim_x
        self.patch_dim_z = patch_dim_z
        self.num_encoding_blocks = num_encoding_blocks
        self.normalization = normalization
        self.dropout = dropout
        self.n_patches = n_patches

        self.encoder = UNet(
            in_channels=1,
            out_classes=2,
            dimensions=3,
            num_encoding_blocks=num_encoding_blocks,
            out_channels_first_layer=out_channels_first_layer,
            normalization=normalization,
            upsampling_type='linear',
            padding=True,
            activation='PReLU',
            dropout=dropout
        ).encoder

        dx = patch_dim_x // 2**(num_encoding_blocks - 1)
        dz = patch_dim_z // 2**(num_encoding_blocks - 1)
        n_in = num_encoding_blocks * out_channels_first_layer * 2 * dx**2 * dz
        n_out = n_in // reduction_factor

        self.fc1 = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.PReLU(),
            nn.BatchNorm1d(n_out)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.n_patches * n_out, n_out // 2),
            nn.PReLU(),
            nn.Linear(n_out // 2, 8)
        )

    def forward_once(self, x):
        skip_connections, output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output = torch.cat((output1, output2), dim=1)
        output = self.fc(output)

        return output


class UnetEncoderSSLJPS(nn.Module):
    def __init__(
        self,
        patch_dim_x=28,
        patch_dim_z=32,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        reduction_factor=64,  # for dense connections from conv layer
        dropout=0,
        n_patches=9,
        n_permutations=100
    ):
        super().__init__()

        self.patch_dim_x = patch_dim_x
        self.patch_dim_z = patch_dim_z
        self.num_encoding_blocks = num_encoding_blocks
        self.normalization = normalization
        self.dropout = dropout
        self.n_patches = n_patches
        self.n_permutations = n_permutations

        self.encoder = UNet(
            in_channels=1,
            out_classes=2,
            dimensions=3,
            num_encoding_blocks=num_encoding_blocks,
            out_channels_first_layer=out_channels_first_layer,
            normalization=normalization,
            upsampling_type='linear',
            padding=True,
            activation='PReLU',
            dropout=dropout
        ).encoder

        dx = patch_dim_x // 2**(num_encoding_blocks - 1)
        dz = patch_dim_z // 2**(num_encoding_blocks - 1)
        n_in = num_encoding_blocks * out_channels_first_layer * 2 * dx**2 * dz
        n_out = n_in // reduction_factor

        self.fc1 = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.PReLU(),
            nn.BatchNorm1d(n_out)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.n_patches * n_out, n_out // 2),
            nn.PReLU(),
            nn.Linear(n_out // 2, self.n_permutations)
        )

    def forward(self, x):
        B, T, C, H, W, Z = x.size()
        x = x.transpose(0, 1)

        encoded_data = []

        for patch in x:
            skip_connections, output = self.encoder(patch)
            output = output.view(output.shape[0], -1)
            output = self.fc1(output)
            encoded_data.append(output)

        x = torch.cat(encoded_data, dim=1)
        x = self.fc(x.view(B, -1))

        return x

    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
