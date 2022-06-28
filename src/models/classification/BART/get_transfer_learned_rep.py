# this code uses the chemprop package (https://github.com/chemprop/chemprop)
# it extracts a representation from a pre-trained DMPNN, evaluated on a new dataset, and writes to .csv
#
# the representation is the final layer of the feed-forward network of the DMPNN
#
# inputs:
# -   trained chemprop model (located in save_dir)
# -   new data to evaluate on the trained model (located in dat_dir)
# output:
# -   representation (written to write_dir)

from argparse import Namespace
import torch
import torch.nn as nn
from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
import csv
from chemprop.train.predict import predict

# directories
save_dir = "emolecules/checkpoints/fold_0/model_0/model.pt"
dat_dir = "data/smiles.csv"
write_dir = "data/chemprop_embeddings.csv"

# load scaler and arguments the model was trained with
scaler, features_scaler = load_scalers(save_dir)
train_args = load_args(save_dir)

# get test data and check smiles are ok
test_data = get_data(dat_dir)

print('Validating SMILES')
valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
full_data = test_data
test_data = MoleculeDataset([test_data[i] for i in valid_indices])

smiles, features = test_data.smiles(), test_data.features()

# load pre-trained model
model = load_checkpoint(save_dir)

# copy pre-trained model
new_model = model

# remove final layer of new_model (i.e. the final layer of the feed-forward neural network)
new_model.ffn = nn.Sequential(*list(new_model.ffn.children())[:-1])

# create data loader
test_data_loader = MoleculeDataLoader(dataset=test_data)

# get predictions from new_model for test smiles
preds = predict(model=new_model, data_loader=test_data_loader)

# write predictions to a .csv file

with open(write_dir, 'w') as f:
    writer = csv.writer(f)

    header = []

    header.append("smiles")

    for i in range(len(preds[1])):
        header.append("chemprop_" + str(i))

    writer.writerow(header)

    for i in range(len(preds)):
        row = []

        row.append(smiles[i][0])
        row.extend(preds[i])

        writer.writerow(row)



