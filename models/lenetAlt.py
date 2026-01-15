import torch.nn as nn

class LeNetAlt(nn.Module):
    """LeNet-5 Network Architecture"""
    
    def __init__(self, num_classes=10, channels=1, input_size=(28, 28)):
        super(LeNetAlt, self).__init__()
        # First conv layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5, padding=2)
        # Average pooling layer: 2x2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Second conv layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # Average pooling layer: 2x2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Formula: (Input // 4) - 2. For 28x28, this results in 5x5.
        h = (input_size[0] // 4) - 2
        w = (input_size[1] // 4) - 2
        # Fully connected layer 1
        self.fc1 = nn.Linear(16 * h * w, 120)

        # Fully connected layer 2: 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # Fully connected layer 3 (output layer): 84 -> 10
        self.fc3 = nn.Linear(84, num_classes)
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Original LeNet uses tanh, but ReLU can also be used

        self.layer_seq = [
            self.conv1, self.tanh, self.pool1, 
            self.conv2, self.tanh, self.pool2, 
            self.fc1, self.tanh, self.fc2, self.tanh, self.fc3
        ]
        
    def forward(self, x, starting_id=0, return_interval=False):
        results = []
        # Iterate starting from the specific ID
        for i in range(starting_id, len(self.layer_seq)):
            # Check if we need to flatten before FC1 (which is at index 6 in layer_seq)
            # Sequence: [conv1(0), tanh(1), pool1(2), conv2(3), tanh(4), pool2(5), fc1(6)...]
            if i == 6:
                x = x.view(x.size(0), -1)
            x = self.layer_seq[i](x)
            if return_interval:
                results.append(x.clone())
        return results if return_interval else x
    
def param_name_to_module_id_lenet_alt(name='depth'):
    if name.startswith('conv1'):
        return 0
    elif name.startswith('conv2'):
        return 1
    elif name.startswith('fc1'):
        return 2
    elif name.startswith('fc2'):
        return 3
    elif name.startswith('fc3'):
        return 4
    elif name == 'depth':
        return 5
    else:
        raise NotImplementedError(f"Unknown parameter name: {name}")