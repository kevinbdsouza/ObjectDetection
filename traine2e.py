# The output data is prepared by representing each output as a binary vector of categories
import torch.utils.data as utils

dataset_size = 82750
learning_rate=0.0001
batch_size = 50
num_epochs = 2
log_interval = 100

train_class = []
train_features = np.load(open('outputs/training_vectors', 'rb'))

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

#binary target vector 
for train_id in train_ids[:82750]: 
    temp_class = np.zeros(80)
    for i,category in enumerate(train_id_to_categories[train_id]):
        temp_class[category_to_idx[category]] = 1
    
    train_class.append(temp_class)


             
#train fn
def train(comb_model,learning_rate,batch_size,num_epochs):
    
    criterion = nn.MultiLabelSoftMarginLoss() 
    optimizer = torch.optim.Adam(comb_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss_vector = []        
        for b in range(int(dataset_size/batch_size)):
            
            tensor_x = torch.stack([load_image(train_id_to_file[k]) for k in train_ids[batch_size*b:batch_size*(b+1)]])
            tensor_y = torch.stack([torch.Tensor(i) for i in train_class[batch_size*b:batch_size*(b+1)]])
            
                      
            # Convert torch tensor to Variable
            img = Variable(tensor_x)
            labels = Variable(torch.FloatTensor(tensor_y)).cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = comb_model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            loss_vector.append(loss.data[0])
            
            if (b % log_interval == 0): 
                print('batch %d %.6f' %(b,loss.data[0]))
        
        
        np.save(open('outputs/loss_vector'+str(epoch)+str(learning_rate)+str(batch_size), 'wb+'), loss_vector)     
        
    
# Finally train the model
train(comb_model,learning_rate, batch_size, num_epochs)

torch.save(comb_model.state_dict(), './comb_model.pth')

print('training done')
