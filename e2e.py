vgg_model = models.vgg16(pretrained=True).cuda()
vgg_features = nn.Sequential(*list(vgg_model.features.children())[:])
vgg_new = nn.Sequential(*list(vgg_model.classifier.children())[:-2])

class E2E(nn.Module):
    def __init__(self):
        super(E2E, self).__init__()
        self.feature = vgg_features 
        self.fc1 = vgg_new
        self.fc2 = cfy_model
            
    def forward(self, x):
        x1= self.feature(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.fc1(x1)
        out = self.fc2(x2)
        return out
    
comb_model = E2E()

comb_model.cuda()
comb_model.train()
comb_model