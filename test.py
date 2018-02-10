#Now repeat step two using the two layer classifier.
sigmoid = nn.Sigmoid()

cfy_model.eval()

val_features = np.load(open('outputs/validation_vectors', 'rb'))

for image_id in val_ids[:10]:
    
    # Display the image.
    display.display(display.Image(val_id_to_file[image_id]))
        
    # Print all of the category labels for this image.
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))

    # Run the val_vec through the new model.
    out_val = sigmoid(cfy_model(Variable(torch.FloatTensor(val_features[i]))))
       
    # Get the top 5 labels, and their corresponding likelihoods.
    probs, indices = out_val.topk(5)

    print('\n')
    # Iterate and print out the predictions.
    for p in range(0,len(probs)):
        print("%s - %s" %(category_idx_to_name[indices.data[p]],probs.data[p]))
        
        