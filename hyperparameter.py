# The output data is prepared by representing each output as a binary vector of categories
import torch.utils.data as utils

dataset_size = 45000
num_epochs = 1
log_interval = 500

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
    
#transform image 
img_size = 224
loader = transforms.Compose([
  transforms.Scale(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
]) 

#load image 
def load_image(filename):
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    return image_tensor.cuda()


train_class = []
val_class = []

#binary target train vector 
for train_id in train_ids[:45000]: 
    temp_class = np.zeros(80)
    for i,category in enumerate(train_id_to_categories[train_id]):
        temp_class[category_to_idx[category]] = 1
    
    train_class.append(temp_class)

#binary target val vector 
for val_id in val_ids[:100]: 
    temp_class = np.zeros(80)
    for i,category in enumerate(val_id_to_categories[val_id]):
        temp_class[category_to_idx[category]] = 1
    
    val_class.append(temp_class)
    
tensor_x_val = torch.stack([load_image(val_id_to_file[k]) for k in val_ids[:50]])    
tensor_y_val = torch.stack([torch.Tensor(i) for i in val_class[:50]])

#train fn
def train(comb_model,learning_rate,batch_size,num_epochs):
    
    criterion = nn.MultiLabelSoftMarginLoss() 
    optimizer = torch.optim.Adam(comb_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        loss_vector_train = [] 
        loss_vector_val = [] 
        for b in range(int(dataset_size/batch_size)):
            
            tensor_x = torch.stack([load_image(train_id_to_file[k]) for k in train_ids[batch_size*b:batch_size*(b+1)]])
            tensor_y = torch.stack([torch.Tensor(i) for i in train_class[batch_size*b:batch_size*(b+1)]])
            
            # Convert torch tensor to Variable
            img = Variable(tensor_x)
            labels = Variable(torch.FloatTensor(tensor_y)).cuda()
            
            comb_model.train()
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = comb_model(img)
            loss_train = criterion(outputs, labels)
            loss_train.backward()
            optimizer.step()
        
            loss_vector_train.append(loss_train.data[0])
            
            comb_model.eval()
            out_val = comb_model(Variable(tensor_x_val))
            labels_val = Variable(torch.FloatTensor(tensor_y_val)).cuda()
            loss_val = criterion(out_val, labels_val)
            
            loss_vector_val.append(loss_val.data[0])
            
            if (b % log_interval == 0): 
                print('batch %d %.6f %.6f' %(b,loss_train.data[0],loss_val.data[0]))
        
        #print(len(loss_vector))
        np.save(open('outputs/loss_vector_train'+str(learning_rate)+str(batch_size), 'wb+'), loss_vector_train)     
        np.save(open('outputs/loss_vector_val'+str(learning_rate)+str(batch_size), 'wb+'), loss_vector_val)     
    
# Finally train the model

learning_rate_vec = [0.001,0.0001,0.00001]
batch_size_vec = [25,40,50]
for lr in learning_rate_vec:
    for bs in batch_size_vec:
        comb_model = E2E()
        comb_model.cuda()
        train(comb_model,lr, bs, num_epochs)
        torch.save(comb_model.state_dict(), './comb_model'+str(lr)+str(bs)+'.pth')



print('training done')