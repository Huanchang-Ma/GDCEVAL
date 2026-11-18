### Original code from MoFlow (under MIT License) https://github.com/calvin-zcx/moflow
import os
import sys
# Assuming the parent directory of the 'utils_train' folder is one level up from the current directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd
import argparse
import time
from utils_reGen.data_frame_parser import DataFrameParser
from utils_reGen.numpytupledataset import NumpyTupleDataset
from utils_reGen.smile_to_graph import GGNNPreprocessor


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='ogbg-molclintox', choices=['ZINC250k', 'QM9','ogbg-molfreesolv','ogbg-molbace','ogbg-molbbbp', 'ogbg-molclintox'])
args = parser.parse_args()

start_time = time.time()
data_name = args.dataset

if data_name == 'ZINC250k':
    max_atoms = 38
    path = 'data/zinc250k.csv'
    smiles_col = 'smiles'
    label_idx = 1
elif data_name == 'QM9':
    max_atoms = 9
    path = 'data/qm9.csv'
    smiles_col = 'SMILES1'
    label_idx = 2
elif data_name == 'ogbg-molfreesolv':
    max_atoms = 24
    path = 'data/ogbg-molfreesolv.csv'
    smiles_col = 'smiles'
    label_idx = 1
elif data_name == 'ogbg-molbace':
    max_atoms = 97
    path = 'data/ogbg-molbace.csv'
    smiles_col = 'smiles'
    label_idx = 0
elif data_name == 'ogbg-molbbbp':
    max_atoms = 132
    path = 'data/ogbg-molbbbp.csv'
    smiles_col = 'smiles'
    label_idx = 0
elif data_name == 'ogbg-molhiv':
    max_atoms = 222
    path = 'data/ogbg-molhiv.csv'
    smiles_col = 'smiles'
    label_idx = 0
elif data_name == 'ogbg-molclintox':
    max_atoms = 136
    path = 'data/ogbg-molclintox.csv'
    smiles_col = 'smiles'


else:
    raise ValueError(f"[ERROR] Unexpected value data_name={data_name}")

preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)

print(f'Preprocessing {data_name} data')
if data_name=='ogbg-molbace' or data_name=='ogbg-molbbbp' or data_name=='ogbg-molhiv':
    df = pd.read_csv(path)
    labels = df.keys().tolist()[label_idx]
    #print(labels)
    #assert False
elif data_name == 'ogbg-molclintox':
    df = pd.read_csv(path)
    label0 = df.keys().tolist()[0] # FDA_APPROVED
    label1 = df.keys().tolist()[1] # CT_TOX
    labels = [label0, label1]
    # labels = label0


# Caution: Not reasonable but used in chain_chemistry\datasets\zinc.py:
# 'smiles' column contains '\n', need to remove it.
# Here we do not remove \n, because it represents atom N with single bond
#labels = df.keys().tolist()
parser = DataFrameParser(preprocessor, labels=labels, smiles_col=smiles_col)
result = parser.parse(df, return_smiles=True)

dataset = result['dataset']
smiles = result['smiles']

NumpyTupleDataset.save(f'{data_name.lower()}_kekulized_xin.npz', dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
