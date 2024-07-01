class GeneratorLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, output_dim)
        self.bn1 = BatchNorm(256)
        self.bn2 = BatchNorm(512)
        self.bn3 = BatchNorm(1024)
        self.bn4 = BatchNorm(2048)

    def forward(self, noise):
        x = F.relu(self.bn1(self.fc1(noise)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = torch.tanh(self.fc5(x))
        x = x.view(-1, 3, 32, 32)
        return x