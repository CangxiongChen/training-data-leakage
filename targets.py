# Various CNN architectures against which we can test gradient leakage attacks
import torch
from torch import nn

class CustomConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size,stride,padding,dilation, groups,
                         bias, padding_mode)

    def forward(self, input):
        result = super(CustomConv2d,self).forward(input)
        return result


class CustomLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features,out_features,bias)

    def forward(self, input):
        result = super(CustomLinear,self).forward(input)
        return result


class CustomTanh(nn.Tanh):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        result = super(CustomTanh,self).forward(input)
        result.retain_grad()
        return result


class cnn2_c0(torch.nn.Module):
    """
    A 2-layer convolutional network. For testing.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=1,
                                                       kernel_size=3,stride=2,padding=0).to(device),
                                          CustomLinear(in_features=225, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c1(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6,
                                                       kernel_size=3,stride=1,padding=0).to(device),
                                          CustomLinear(in_features=5400, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c2(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6,
                                        kernel_size=4,stride=2,padding=0).to(device),
                                        CustomLinear(in_features=1350, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c4(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=1,
                                                       kernel_size=3,stride=1,padding=0).to(device),
                                          CustomLinear(in_features=900, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c5(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=3,
                                                       kernel_size=3,stride=1,padding=1).to(device),
                                          CustomLinear(in_features=3072, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c11(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6,
                                                       kernel_size=3,stride=1,padding=0).to(device),
                                          CustomLinear(in_features=5400, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c21(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6,
                                                       kernel_size=4,stride=2,padding=0).to(device),
                                          CustomLinear(in_features=1350, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c41(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=1,
                                                       kernel_size=3,stride=1,padding=0).to(device),
                                          CustomLinear(in_features=900, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn2_c6(torch.nn.Module):
    """
    A 2-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=12,
                                                       kernel_size=5,stride=2,padding=0).to(device),
                                          CustomLinear(in_features=2352, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c1(torch.nn.Module):
    """
    A 3-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c8(torch.nn.Module):
    """
    Same as cnn3_c1 apart from dims in intermediate layers.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=9, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=9, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c2(torch.nn.Module):
    """
    A 3-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=3, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=147, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(), nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c3(torch.nn.Module):
    """
    A 3-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=9, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=7056, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c9(torch.nn.Module):
    """
    Similar to cnn3_c3 with slower growth of dims of representations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=5, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=5, out_channels=7, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=5488, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c4(torch.nn.Module):
    """
    A 3-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=1, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=1, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=4704, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c5(torch.nn.Module):
    """
    for testing purpose
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=12, \
                                                       kernel_size=5, stride=1, padding=1).to(device),
                                          CustomConv2d(in_channels=12, out_channels=12, \
                                                       kernel_size=5, stride=2, padding=2).to(device),
                                          CustomLinear(in_features=768, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c6(torch.nn.Module):
    """
    cnn3_c5 with the last activation changed to Tanh.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.LeakyReLU(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c7(torch.nn.Module):
    """
    cnn3_c6 with the first activation changed to Tanh.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.LeakyReLU(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn4_c1(torch.nn.Module):
    """
    A 4-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1,padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=5, \
                                                       kernel_size=4, stride=2,padding=0).to(device),
                                          CustomConv2d(in_channels=5, out_channels=3, \
                                                       kernel_size=4, stride=1,padding=0).to(device),
                                          CustomLinear(in_features=363, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(), nn.Tanh(), nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn4_c3(torch.nn.Module):
    """
    A 4-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=5, stride=1,padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=16, \
                                                       kernel_size=5, stride=2,padding=0).to(device),
                                          CustomConv2d(in_channels=16, out_channels=16, \
                                                       kernel_size=5, stride=1,padding=2).to(device),
                                          CustomLinear(in_features=2304, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn4_c2(torch.nn.Module):
    """
    A 4-layer convolutional network.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=16, \
                                                       kernel_size=5, stride=1,padding=0).to(device),
                                          CustomConv2d(in_channels=16, out_channels=6, \
                                                       kernel_size=5, stride=2,padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=32, \
                                                       kernel_size=5, stride=1,padding=2).to(device),
                                          CustomLinear(in_features=4608, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(), nn.Tanh(), nn.Tanh(),nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c11(torch.nn.Module):
    """
    Similar to cnn3_c1 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c13(torch.nn.Module):
    """
    Similar to cnn3_c1 but with different activations: nn.Tanh(),nn.Tanh(),nn.Sigmoid()
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Tanh(),nn.Tanh(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c12(torch.nn.Module):
    """
    Similar to cnn3_c1 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Softplus(),nn.Softplus(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c14(torch.nn.Module):
    """
    Similar to cnn3_c1 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Identity(),nn.Identity(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c15(torch.nn.Module):
    """
    Similar to cnn3_c1 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=588, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Identity(),nn.Identity(),nn.Tanh()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c21(torch.nn.Module):
    """
    Similar to cnn3_c2 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=3, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=147, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c22(torch.nn.Module):
    """
    Similar to cnn3_c2 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=4, stride=2, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=3, \
                                                       kernel_size=3, stride=2, padding=0).to(device),
                                          CustomLinear(in_features=147, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Softplus(),nn.Softplus(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c31(torch.nn.Module):
    """
    Similar to cnn3_c3 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=9, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=7056, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c32(torch.nn.Module):
    """
    Similar to cnn3_c3 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=6, out_channels=9, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=7056, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Softplus(),nn.Softplus(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c41(torch.nn.Module):
    """
    Similar to cnn3_c4 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=1, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=1, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=4704, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn3_c42(torch.nn.Module):
    """
    Similar to cnn3_c4 but with different activations.
    """
    def __init__(self, device):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=3, out_channels=1, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomConv2d(in_channels=1, out_channels=6, \
                                                       kernel_size=3, stride=1, padding=0).to(device),
                                          CustomLinear(in_features=4704, out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.Softplus(),nn.Softplus(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)

            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class ResNetblock(torch.nn.Module):
    """
    A ResNet block, omitting batchnorm as we are dealing with one input at a time.
    """

    def __init__(self, input_h, num_filters, k_size, in_channels, strides, padding, device='cpu'):
        super().__init__()
        self.module_list = nn.ModuleList([CustomConv2d(in_channels=in_channels, out_channels=num_filters,
                                                       kernel_size=k_size,stride=strides,padding=padding).to(device),
                                          CustomConv2d(in_channels=num_filters, out_channels=num_filters,
                                                       kernel_size=k_size,stride=strides,padding=padding).to(device),
                                          CustomLinear(in_features=(input_h-2*k_size+4*padding+2)**2*num_filters,
                                                       out_features=10).to(device)])

        self.activation_list = nn.ModuleList([nn.LeakyReLU(),nn.LeakyReLU(),nn.Sigmoid()])

    def forward(self, input):
        x = input
        x_shape = []
        for i, (layer, act) in enumerate(zip(self.module_list, self.activation_list)):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            if i == 1:
                x += input
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class cnn6d(nn.Module):
    """
    cnn6d model used in the RGAP paper.
    For the last layer:
        1. output dimension for the last layer has been changed to 10 instead of 1.
        2. Bias is set to True instead of False.

    """

    def __init__(self, device):
        super().__init__()
        act = nn.LeakyReLU(negative_slope=0.2)
        self.module_list = nn.ModuleList([nn.Conv2d(3, 12, kernel_size=4, padding=2, stride=2, bias=False).to(device),
                                         nn.Conv2d(12, 20, kernel_size=3, padding=1, stride=2, bias=False).to(device),
                                         nn.Conv2d(20, 36, kernel_size=3, padding=1, stride=1, bias=False).to(device),
                                         nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False).to(device),
                                         nn.Conv2d(36, 64, kernel_size=3, padding=1, stride=2, bias=False).to(device),
                                         nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False).to(device),
                                         nn.Linear(3200, 10, bias=True).to(device)])
        self.activation_list = nn.ModuleList([act, act, act, act, act, act, nn.Identity()])

    def forward(self, input):
        x = input
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            print("shape of x before a layer: {}".format(x.shape))
            x = layer(x)
            print("shape of x after a layer: {}".format(x.shape))
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class LeNet(nn.Module):
    """
    LeNet for CIFAR-10 implementation adapted from
    https://github.com/mit-han-lab/dlg/blob/d21007fa1540ba2303ebc034976aa331814727c7/models/vision.py#L15
    """
    def __init__(self, device):
        super(LeNet, self).__init__()
        self.module_list = nn.ModuleList([nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2).to(device),
                                          nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2).to(device),
                                          nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1).to(device),
                                          nn.Linear(768, 10, bias=True).to(device)])
        self.activation_list = nn.ModuleList([nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU(), nn.Identity()])

    def forward(self, x):
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


class LeNet2(nn.Module):
    """
    LeNet for CIFAR-10 implementation adapted from
    https://github.com/mit-han-lab/dlg/blob/d21007fa1540ba2303ebc034976aa331814727c7/models/vision.py#L15
    """
    def __init__(self, device):
        super(LeNet2, self).__init__()
        self.module_list = nn.ModuleList([CustomConv2d(3, 12, kernel_size=3, padding=1, stride=2).to(device),
                                          CustomConv2d(12, 12, kernel_size=5, padding=2, stride=2).to(device),
                                          CustomConv2d(12, 12, kernel_size=5, padding=2, stride=1).to(device),
                                          CustomLinear(768, 10, bias=True).to(device)])
        self.activation_list = nn.ModuleList([nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.Identity()])

    def forward(self, x):
        x_shape = []
        for layer, act in zip(self.module_list, self.activation_list):
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x,1)
            x = layer(x)
            x = act(x)
            x_shape.append(x.shape)
        return x, x_shape


def gen_matrix_with_rank(rank, num_rows, num_cols, seed):
    """generate a random matrix of shape num_rows, num_cols with rank r"""
    import numpy as np
    from numpy.linalg import svd
    from scipy.stats import ortho_group
    m = num_rows
    n = num_cols
    A = ortho_group.rvs(m,random_state=seed)
    B = ortho_group.rvs(n, random_state=seed)
    diag = np.zeros(shape=(min(m,n),))
    diag[0:rank] = 1
    sigma = np.zeros(shape=(m,n))
    np.fill_diagonal(sigma,diag)
    res = (A.dot(sigma)).dot(B)

    # rescaling by a factor of 10
    res = res*0.1

    return res


def weight_init(shape, rank, same, seed):
    """
    shape: shape of the weight (c_out, c_in, h, h)
    rank: rank for the weight matrix in a single channel
    same: if True, each channel will have the same weight matrix; otherwise, random for each channel
    return a weight matrix with each channel-wise piece having rank 'rank'
    """
    import numpy as np
    res = np.zeros(shape=shape)

    if same:
        weight_per_channel = gen_matrix_with_rank(rank=rank, num_rows=shape[2], num_cols=shape[2], seed=seed)
        for i in range(shape[0]):
            for j in range(shape[1]):
                res[i,j,:,:] = weight_per_channel
    elif not same:
        for i in range(shape[0]):
            for j in range(shape[1]):
                weight_per_channel = gen_matrix_with_rank(rank=rank, num_rows=shape[2], num_cols=shape[2], seed=seed)
                res[i,j,:,:] = weight_per_channel

    return res


