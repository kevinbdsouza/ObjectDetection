input_size = 4096
hidden_state_size = 512
num_classes = 80

# First we construct a class for the model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
            
    def forward(self, x):
        out = self.fc1(x)
        out = self.sig1(out)
        out = self.fc2(out)
        return out
    
cfy_model = Model(input_size, hidden_state_size, num_classes)
cfy_model.train()
cfy_model