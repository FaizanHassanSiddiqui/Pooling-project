import numpy as np
import torch

def validation(model, dataloaders, criterion, use_cuda):
    
    valid_loss = 0.0
    accuracy = 0.0
    
    for batch_idx, (features, labels) in enumerate(dataloaders['valid']):
        
        if use_cuda:
            features, labels = features.cuda(), labels.cuda()

        #features.resize_(images.shape[0], 784)
        
        
        
        
        labels = labels.view(-1,1)
        output = model.forward(features)
        loss = criterion(output, labels)
        
        valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        
        
        equality = labels.data== torch.round(output)
        accuracy += equality.type(torch.FloatTensor).mean() 
        
    
    return valid_loss, accuracy






def train(epochs, dataloaders, model, optimizer, criterion, save_path, use_cuda = False):
    
    print_every = 6
    steps = 0
    
    valid_loss_min = np.Inf

    for epoch in range(1, epochs+1):
        
        train_loss = 0.0
        
       
        model.train()

        for batch_idx, (features, labels) in enumerate(dataloaders['train']):
            
            steps += 1
            
            if use_cuda:
                features, labels = features.cuda(), labels.cuda()

            optimizer.zero_grad()
            output = model.forward(features)
            labels = labels.view(-1,1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()                  # updating the weights

            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))     # getting the loss out
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx + 1}, loss: {train_loss}')

            

        model.eval()

                # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            valid_loss, valid_accuracy = validation(model, dataloaders, criterion, use_cuda)
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss,
            valid_accuracy
            ))
                
                
        if valid_loss < valid_loss_min:

            
            checkpoint = {'input_size': 4,
              'output_size': 1,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

            torch.save(checkpoint, save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss  


    return model
