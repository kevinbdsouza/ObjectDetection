import math

f_val = np.load(open('outputs/validation_vectors', 'rb'))
f_train = np.load(open('outputs/training_vectors', 'rb'))

i = 0
for val_id in val_ids[:10]:
    j = 0    
    val_vec = f_val[i]
    min_id = 0
    min_dist = math.inf
    for train_id in train_ids[:82783]:
        train_vec = f_train[j]
        
        dist = np.square(np.linalg.norm(val_vec-train_vec))
        if (dist < min_dist):
            min_dist = dist
            min_id = train_id
            
        j = j + 1    
     
    display.display(display.Image(val_id_to_file[val_id]))
    display.display(display.Image(train_id_to_file[min_id]))
    i = i + 1
    
    