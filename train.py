# The output data is prepared by representing each output as a binary vector of categories
import torch.utils.data as utils

learning_rate=0.001
batch_size = 100
num_epochs = 5
log_interval = 500

train_class = []
train_features = np.load(open('outputs/training_vectors', 'rb'))

#get 80 dimensional binary vectors 
for train_id in train_ids[:82783]: 
    temp_class = np.zeros(80)
    for i,category in enumerate(train_id_to_categories[train_id]):
        temp_class[category_to_idx[category]] = 1 
    
    train_class.append(temp_class)
    
#specify criterion and optimizer
criterion = nn.MultiLabelSoftMarginLoss() 
optimizer = torch.optim.Adam(cfy_model.parameters(), lr=learning_rate) 

#make custom dataset 
tensor_x = torch.stack([torch.Tensor(i) for i in train_features])
tensor_y = torch.stack([torch.Tensor(i) for i in train_class])
my_dataset = utils.TensorDataset(tensor_x,tensor_y)
train_loader = utils.DataLoader(dataset=my_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


#train function
def train(model, learning_rate, batch_size, num_epochs):
        
    for epoch in range(num_epochs):
        for batch_idx, (features, labels) in enumerate(train_loader):
                    
            # Convert torch tensor to Variable
            features = Variable(torch.FloatTensor(features))
            labels = Variable(torch.FloatTensor(labels))
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = cfy_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
                       
        print('epoch %d %.6f' %(epoch,loss.data[0]))
            
#train the model
train(cfy_model,learning_rate, batch_size, num_epochs)

print('training done')