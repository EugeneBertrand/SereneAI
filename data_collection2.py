import pandas as pd  # For data manipulation
import requests  # For making HTTP requests to the ChEMBL API
import pubchempy as pcp  # For interacting with PubChem

# Function to fetch target data from ChEMBL API
def fetch_receptors(query):
    """
    Fetch receptor data from ChEMBL based on a search query.
    
    :param query: The receptor type to search for (e.g., "serotonin receptor")
    :return: List of receptor objects
    """
    url = f"https://www.ebi.ac.uk/chembl/api/data/target?search={query}"
    response = requests.get(url)  # Send a GET request
    if response.status_code == 200:  # Check if the request was successful
        return response.json().get('targets', [])
    else:
        print(f"Error fetching data for {query}: {response.status_code}")
        return []

# Fetch serotonin and GABA receptors
serotonin_receptors = fetch_receptors('serotonin receptor')
gaba_receptors = fetch_receptors('GABA receptor')

# Function to fetch bioactivity data for a given receptor
def fetch_bioactivity_data(receptors, target='IC50'):
    """
    Fetch bioactivity data for a list of receptors from ChEMBL API.
    
    :param receptors: List of receptor dictionaries
    :param target: Type of bioactivity (default is IC50)
    :return: DataFrame with bioactivity data
    """
    data = []  # Initialize list for bioactivity data
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
    return pd.DataFrame(data)  # Convert list to DataFrame

# Fetch bioactivity data for serotonin and GABA receptors
serotonin_df = fetch_bioactivity_data(serotonin_receptors)
gaba_df = fetch_bioactivity_data(gaba_receptors)

# Combine bioactivity data into a single DataFrame
bioactivity_data = pd.concat([serotonin_df, gaba_df]).drop_duplicates()
bioactivity_data.to_csv('bioactivity_data.csv', index=False)  # Save to CSV without index

# Supplement data with compound properties from PubChem
compound_ids = [2244, 5288826, 2222]  # Sample compound CIDs for demonstration
compound_data = []
for cid in compound_ids:
    compound = pcp.Compound.from_cid(cid)  # Fetch compound details
    compound_data.append({
        'cid': cid,
        'molecular_weight': compound.molecular_weight,
        'logp': compound.xlogp,
        'h_bond_donor_count': compound.h_bond_donor_count,
        'h_bond_acceptor_count': compound.h_bond_acceptor_count
    })

# Convert compound data to DataFrame
compound_df = pd.DataFrame(compound_data)
# Merge bioactivity data with compound properties
combined_data = pd.merge(bioactivity_data, compound_df, left_on='molecule_chembl_id', right_on='cid', how='inner')

# Save the combined data to CSV
combined_data.to_csv('combined_bioactivity_data.csv', index=False)
