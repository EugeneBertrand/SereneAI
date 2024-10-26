import pandas as pd
import requests
import pubchempy as pcp
from nemo.collections.nlp.models import TokenClassificationModel  # Adjust model type as needed

# Function to fetch target data from ChEMBL API
def fetch_receptors(query):
    """
    Fetch receptor data from ChEMBL based on a search query.
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/target?search={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('targets', [])
    else:
        print(f"Error fetching data for {query}: {response.status_code}")
        return []

# Function to fetch bioactivity data for a given receptor
def fetch_bioactivity_data(receptors, target='IC50'):
    """
    Fetch bioactivity data for a list of receptors from ChEMBL API.
    """
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

# Fetch data for serotonin and GABA receptors
serotonin_receptors = fetch_receptors('serotonin receptor')
gaba_receptors = fetch_receptors('GABA receptor')
serotonin_df = fetch_bioactivity_data(serotonin_receptors)
gaba_df = fetch_bioactivity_data(gaba_receptors)

# Combine bioactivity data
bioactivity_data = pd.concat([serotonin_df, gaba_df]).drop_duplicates()

# Supplement with compound properties from PubChem
compound_ids = [2244, 5288826, 2222]  # Sample compound CIDs for demonstration
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

# Merge bioactivity data with compound properties
combined_data = pd.merge(bioactivity_data, compound_df, left_on='molecule_chembl_id', right_on='cid', how='inner')

# Filter and preprocess data for NeMo
nemo_data = combined_data[['molecule_chembl_id', 'target', 'standard_value', 'molecular_weight', 'logp']].copy()
nemo_data.dropna(inplace=True)  # Drop rows with missing values

# Save the preprocessed data for NeMo
nemo_data.to_csv('nemo_ready_data.csv', index=False)

# Load the data into NeMo for training (assuming a token classification model, adjust as needed)
model = TokenClassificationModel.from_pretrained("token_classification_model")
model.setup_training_data(train_data='nemo_ready_data.csv')

# Start model training
model.train()
