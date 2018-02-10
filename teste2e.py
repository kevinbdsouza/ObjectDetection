sigmoid = nn.Sigmoid()

comb_model.load_state_dict(torch.load('./comb_model.pth'))

#comb_model.cuda()
comb_model.eval()

#image preprocessing 
img_size = 224
loader = transforms.Compose([
  transforms.Scale(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
]) 
def load_image(filename):
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor).unsqueeze(0)
    return image_var.cuda()


for image_id in val_ids[:10]:
    
    # Display the image.
    display.display(display.Image(val_id_to_file[image_id]))
        
    # Print all of the category labels for this image.
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))

    # Run the val img through the new model.
    img = load_image(val_id_to_file[image_id])
    out_val = sigmoid(comb_model(img))
       
    # Get the top 5 labels, and their corresponding likelihoods.
    probs, indices = out_val.topk(5)

    print('\n')
    probs = probs.data.cpu().numpy()[0]
    indices = indices.data.cpu().numpy()[0]
    
    # Iterate and print out the predictions.
    for p in range(0,len(probs)):
        print("%s - %s" %(category_idx_to_name[indices.data[p]],probs[p]))
