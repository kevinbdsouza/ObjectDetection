#plotting 


learning_rate_vec = [0.001,0.0001,0.00001]
batch_size = [10,25,50]

for lr in learning_rate_vec:
    for bs in batch_size_vec:

        loss_vector_train = np.load(open('outputs/loss_vector_train'+str(lr)+str(bs), 'rb'))
        loss_vector_val = np.load(open('outputs/loss_vector_val'+str(lr)+str(bs), 'rb'))
        batches = np.arange(1,len(loss_vector_train)+1,1)
        batches = batches*bs
        plt.xlabel('No. of images')
        plt.ylabel('Loss')
        plt.title('Loss vs No. of batches for lr ='+str(lr)+' and bs ='+str(bs))
        lineT, =  plt.plot(batches, loss_vector_train, c='r', label="Training loss", linewidth=2.0)
        lineV, =  plt.plot(batches, loss_vector_val, c='b', label="Validation loss", linewidth=2.0)
        plt.legend(handles=[lineT, lineV])
        plt.show()


        
        
        