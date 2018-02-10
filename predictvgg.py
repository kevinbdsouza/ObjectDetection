vgg_model = models.vgg16(pretrained=True).cuda()
vgg_model.eval()

# Let's see what the model looks like.
vgg_model

softmax = nn.Softmax()
for image_id in val_ids[:10]:
    # Display the image.
    display.display(display.Image(val_id_to_file[image_id]))
        
    # Print all of the category labels for this image.
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))

    # Load/preprocess the image.
    img = load_image(val_id_to_file[image_id])

    # Run the image through the model and softmax.
    label_likelihoods = softmax(vgg_model(img)).squeeze()

    # Get the top 5 labels, and their corresponding likelihoods.
    probs, indices = label_likelihoods.topk(5)
    
    #check why labels not matching 
    print('\n')
    # Iterate and print out the predictions.
    for p in range(0,len(probs)):
        print("%s - %s" %(imagenet_categories[indices.data[p]],probs.data[p]))
        
    

    