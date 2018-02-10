# Remove the final layer of the classifier, and indicate to PyTorch that the model is being used for inference
# rather than training (most importantly, this disables dropout).

vgg_model = models.vgg16(pretrained=True).cuda()
vgg_features = nn.Sequential(*list(vgg_model.features.children())[:])
vgg_new = nn.Sequential(*list(vgg_model.classifier.children())[:-2])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = vgg_features
        self.fc = vgg_new 
    
    def forward(self, x):
        x1= self.feature(x)
        x1 = x1.view(x1.size(0), -1)
        out = self.fc(x1)
        return out
  

#create model 
vgg_fc2 = Model()
vgg_fc2.eval()

# display
vgg_fc2

# First we vectorize all of the features of training images and write the results to a file.

training_vectors = []

for image_id in train_ids[:82783]:
    # Load/preprocess the image.
    img = load_image(train_id_to_file[image_id])
    
    #run and extract features
    r = vgg_fc2(img).cpu().data.numpy()[0]
    training_vectors.append(r)
        
    
np.save(open('outputs/training_vectors', 'wb+'), training_vectors)
   
  

# Next we vectorize all of the features of validation images and write the results to a file.
    
validation_vectors = []

for image_id in val_ids[:100]:
    # Load/preprocess the image.
    img = load_image(val_id_to_file[image_id])
    
    #run and extract features
    r = vgg_fc2(img).cpu().data.numpy()[0]
    validation_vectors.append(r)

    
np.save(open('outputs/validation_vectors', 'wb+'), validation_vectors)    
    

  