
import numpy as np
import pandas as pd
import torch

from mymodel import Network

input_file = pd.read_csv("prediction_data.csv", header = None )

trained_model_path = "checkpoint.pth"

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint(trained_model_path)


data1 = np.array(input_file)

assert data1.ndim == 2
data1 = torch.FloatTensor(data1)
prediction = model.forward(data1)
prediction = prediction.detach().numpy()

prediction = np.where(prediction>0.5, True, False)

for pred in prediction.ravel():
    if pred:
        print("goes to pool!")
    else:
        print("doesn't go to pool.")