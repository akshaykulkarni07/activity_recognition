class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # defining layers
        self.maxpool = nn.MaxPool1d(3, stride = 2)
        self.conv1 = nn.Conv1d(3, 5, 3)
        self.conv2 = nn.Conv1d(5, 10, 5)
        self.conv3 = nn.Conv1d(10, 10, 5)
        self.conv4 = nn.Conv1d(10, 10, 3)
        self.conv5 = nn.Conv1d(10, 10, 3)
        self.fc = nn.Linear(26 * 10, 6)
        
        nn.init.xavier_uniform_(self.conv1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv3.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv4.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv5.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc.weight, gain = nn.init.calculate_gain('sigmoid'))
        
    def forward(self, signal):
        signal = signal.view(-1, 3, 80)
        out = self.maxpool(F.relu(self.conv1(signal)))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = out.view(-1, 26 * 10)
        out = self.fc(out)
        out = F.log_softmax(out, dim = 1)
        return out