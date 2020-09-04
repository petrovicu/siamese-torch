import torch

source = '/home/wingman2/models/model-inter-901.pt'
dest = '/home/wingman2/models/siamese-model.pt'

model_state = torch.load(source)
new_model_state = {}

for key in model_state.keys():
    new_model_state[key[7:]] = model_state[key]

torch.save(new_model_state, dest)