import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import time
from mymodel import Network
from train import train


SAVE_PATH = "checkpoint.pth"
LEARNING_RATE = 0.001
EPOCHS = 1000


# engine = create_engine('data base link hiddena')


class MyData(Dataset):
    
    def __init__(self):
        df = pd.read_sql_table("weather_ml", engine)
        self.y = torch.FloatTensor((df['pooling'].replace({'true': 1, 'false': 0})).values)
        
        self.x = torch.FloatTensor(df.drop(['id', 'pooling', 'date'], axis = 1).values)
        
        self.n_samples = df.shape[0]
        
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]
    
    def __len__(self):
        
        return self.n_samples
count=0
temp_count=0
while (True):

    try:
        time.sleep(15)
        dataset = MyData()
        
        count=len(dataset)       
        if(count>temp_count):
            temp_count=count
        
            validation_split = 0.2
            shuffle_dataset = True
            random_seed = 35
            batch_size = 167

            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            if shuffle_dataset :
                np.random.seed(random_seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size=batch_size, 
                                                    sampler=train_sampler)
            validation_loader = DataLoader(dataset, batch_size=len(val_indices),
                                                            sampler=valid_sampler)

            dataloaders = {'train': train_loader ,
                        
                        'valid': validation_loader}


            model = Network(4, 1, [6,])

            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)


            use_cuda = torch.cuda.is_available()

            trained_model = train(EPOCHS, dataloaders, model, optimizer, criterion, SAVE_PATH, use_cuda)
        else:
            print("db count is same")
    except KeyboardInterrupt:
        print("Program is exiting as keyboard interruption is made")
        break        
    except:
        print("Error Occurred please check your connection-> program will keep running")