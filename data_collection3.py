import pandas as pd
import requests
import pubchempy as pcp
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import nemo.collections.nlp as nemo_nlp

# Step 1: Fetch receptor data from ChEMBL API
def fetch_receptors(query):
    url = f"https://www.ebi.ac.uk/chembl/api/data/target?search={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('targets', [])
    else:
        print(f"Error fetching data for {query}: {response.status_code}")
        return []

serotonin_receptors = fetch_receptors('serotonin receptor')
gaba_receptors = fetch_receptors('GABA receptor')

# Step 2: Fetch bioactivity data for receptors
def fetch_bioactivity_data(receptors, target='IC50'):
    data = []
    for receptor in receptors:
        receptor_id = receptor.get('target_chembl_id')
        if receptor_id:
            url = f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id={receptor_id}&standard_type={target}"
            response = requests.get(url)
            if response.status_code == 200:
                activities = response.json().get('activities', [])
                data.extend(activities)
            else:
                print(f"Error fetching bioactivity data for {receptor_id}: {response.status_code}")
    return pd.DataFrame(data)

serotonin_df = fetch_bioactivity_data(serotonin_receptors)
gaba_df = fetch_bioactivity_data(gaba_receptors)
bioactivity_data = pd.concat([serotonin_df, gaba_df]).drop_duplicates()
bioactivity_data.to_csv('bioactivity_data.csv', index=False)

# Step 3: Supplement bioactivity data with compound properties from PubChem
compound_ids = [2244, 5288826, 2222]
compound_data = []
for cid in compound_ids:
    compound = pcp.Compound.from_cid(cid)
    compound_data.append({
        'cid': cid,
        'molecular_weight': compound.molecular_weight,
        'logp': compound.xlogp,
        'h_bond_donor_count': compound.h_bond_donor_count,
        'h_bond_acceptor_count': compound.h_bond_acceptor_count
    })
compound_df = pd.DataFrame(compound_data)
combined_data = pd.merge(bioactivity_data, compound_df, left_on='molecule_chembl_id', right_on='cid', how='inner')
combined_data.to_csv('combined_bioactivity_data.csv', index=False)

# Step 4: Preprocess the data
data = pd.read_csv('combined_bioactivity_data.csv')
data = data.drop(columns=['molecule_chembl_id', 'cid']).dropna()

scaler = StandardScaler()
data[['molecular_weight', 'logp', 'h_bond_donor_count', 'h_bond_acceptor_count']] = scaler.fit_transform(
    data[['molecular_weight', 'logp', 'h_bond_donor_count', 'h_bond_acceptor_count']]
)

if 'bioactivity_class' in data.columns:
    encoder = LabelEncoder()
    data['bioactivity_class'] = encoder.fit_transform(data['bioactivity_class'])

data.to_csv('preprocessed_bioactivity_data.csv', index=False)

# Step 5: Define a PyTorch Dataset and DataLoader
class BioactivityDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data.iloc[idx, :-1].values  # All columns except target
        target = self.data.iloc[idx, -1]  # Last column as target
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

dataset = BioactivityDataset('preprocessed_bioactivity_data.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 6: Initialize and train a NeMo model
# Note: Model type and parameters depend on the specific use case (e.g., TextClassificationModel, etc.)
model = nemo_nlp.models.TextClassificationModel(...)  # Specify model details as per the task

# Start training
model.train(train_loader)
