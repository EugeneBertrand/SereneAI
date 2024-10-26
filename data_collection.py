# data_collection.py
import pandas as pd  # Import pandas for data manipulation
import pubchempy as pcp  # Import PubChemPy to interact with PubChem
from chembl_webresource_client.new_client import new_client  # Import ChEMBL API client

# Initialize ChEMBL client to access the target data
chembl_client = new_client.target

# Search for serotonin and GABA receptors in the ChEMBL database
serotonin_receptors = chembl_client.search('serotonin receptor')
gaba_receptors = chembl_client.search('GABA receptor')

def fetch_bioactivity_data(receptors, target='IC50'):
    """
    Fetch bioactivity data for a given list of receptors.
    
    :param receptors: List of receptor objects from ChEMBL
    :param target: The type of bioactivity to filter (default is IC50)
    :return: DataFrame containing bioactivity information
    """
    data = []  # Initialize an empty list to store the data
    for receptor in receptors:  # Loop through each receptor
        # Fetch bioactivity data from ChEMBL API for the current receptor
        bioactivities = new_client.activity.filter(target_chembl_id=receptor['target_chembl_id']).filter(standard_type=target)
        for activity in bioactivities:  # Loop through each bioactivity entry
            data.append(activity)  # Append activity to the data list
    return pd.DataFrame(data)  # Convert the list to a DataFrame and return it

# Fetch bioactivity data for serotonin and GABA receptors
serotonin_df = fetch_bioactivity_data(serotonin_receptors)
gaba_df = fetch_bioactivity_data(gaba_receptors)

# Combine and save the bioactivity data to a CSV file
bioactivity_data = pd.concat([serotonin_df, gaba_df]).drop_duplicates()  # Combine dataframes and remove duplicates
bioactivity_data.to_csv('bioactivity_data.csv', index=False)  # Save to CSV without index

# Supplement with compound data from PubChem
compound_ids = [2244, 5288826, 2222]  # List of sample Compound IDs (CIDs) - Replace with your own CIDs
compound_data = []  # Initialize an empty list to store compound data
for cid in compound_ids:  # Loop through each Compound ID
    compound = pcp.Compound.from_cid(cid)  # Fetch compound information from PubChem
    compound_data.append({  # Append compound properties to the list
        'cid': cid,
        'molecular_weight': compound.molecular_weight,
        'logp': compound.xlogp,
        'h_bond_donor_count': compound.h_bond_donor_count,
        'h_bond_acceptor_count': compound.h_bond_acceptor_count
    })

# Convert the compound data list to a DataFrame
compound_df = pd.DataFrame(compound_data)
# Merge bioactivity data with compound properties based on 'cid'
combined_data = pd.merge(bioactivity_data, compound_df, left_on='molecule_chembl_id', right_on='cid', how='inner')
# Save the combined data to a CSV file
combined_data.to_csv('combined_bioactivity_data.csv', index=False)  # Save to CSV without index
