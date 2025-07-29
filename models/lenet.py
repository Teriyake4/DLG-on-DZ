import torch.nn as nn
import torch

class LeNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, input_size=(1, 28, 28)):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),                                                             # 1

            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),       # 2

            act(),                                                             # 3

            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),       # 4

            act(),                                                             # 5
        )
        # Dynamically calculate the hidden size
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, *input_size)  # e.g., [1, 3, 32, 32]
        #     dummy_output = self.body(dummy_input)
        #     hidden = dummy_output.numel()
        hidden = 588
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )
        # Save flattened size and modules
        self.hidden = hidden
        self.splited_modules = list(self.body) + list(self.fc)
        self.length = len(self.splited_modules)

    def forward(self, x, starting_id=0, return_interval=False):
        if return_interval:
            res = []

        for i in range(starting_id, self.length):
            x = self.splited_modules[i](x)

            if i == self.length - 2:  # Before the Linear layer
                x = x.view(x.size(0), -1)

            if return_interval:
                res.append(x.clone())

        return res if return_interval else x


def lenet(channel=1, hidden=768, num_classes=10):
    return LeNet(channel=1, num_classes=num_classes)

def param_name_to_module_id_lenet(name = 'depth'):
    if name.startswith('body.0'):
        return 0
    elif name.startswith('body.2'):
        return 1
    elif name.startswith('body.4'):
        return 2
    elif name.startswith('fc.0'):
        return 3
    else:
        raise NotImplementedError(f"Unknown parameter name: {name}")