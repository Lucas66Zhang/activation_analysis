from torch import nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet5_obs(nn.Module):
    def __init__(self):
        super(LeNet5_obs, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        output = {}
        x = self.relu(self.conv1(x))
        output["conv1"] = x
        x = self.maxpool1(x)
        output["maxpool1"] = x
        x = self.relu(self.conv2(x))
        output["conv2"] = x
        x = self.maxpool2(x)
        output["maxpool2"] = x
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        output["fc1"] = x
        x = self.relu(self.fc2(x))
        output["fc2"] = x
        x = self.fc3(x)
        output["fc3"] = x
        return output