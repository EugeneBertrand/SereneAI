import pandas as pd
from chembl_webresource_client.new_client import new_client

# Initialize ChEMBL client
chembl_client = new_client.target

# Search for serotonin-related targets in the ChEMBL database
serotonin_targets = chembl_client.search('serotonin')

# Convert serotonin target data to a DataFrame
def fetch_target_data(targets):
    """
    Convert target data into a DataFrame.
    """
    data = []
    for target in targets:
        data.append({
            'target_chembl_id': target.get('target_chembl_id'),
            'pref_name': target.get('pref_name'),
            'organism': target.get('organism'),
            'target_type': target.get('target_type'),
            'description': target.get('description')
        })
    return pd.DataFrame(data)

# Fetch data and save to CSV
serotonin_df = fetch_target_data(serotonin_targets)
serotonin_df.to_csv('serotonin_targets.csv', index=False)

# HTML Template Function
def generate_html(data_file='serotonin_targets.csv', html_file='serotonin_targets.html'):
    """
    Generates an HTML page to display data from the CSV file.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Serotonin Targets - ChEMBL Data</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f7f9;
                color: #333;
                display: flex;
                justify-content: center;
                padding: 20px;
            }}
            .container {{
                max-width: 900px;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }}
            h1 {{
                text-align: center;
                color: #333;
            }}
            .target {{
                background: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 15px;
            }}
            .target h2 {{
                color: #0073e6;
            }}
            .label {{
                font-weight: bold;
                color: #555;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Serotonin-Related Targets in ChEMBL</h1>
    """

    # Read the CSV file to include data in HTML
    df = pd.read_csv(data_file)
    for _, row in df.iterrows():
        html_content += f"""
        <div class="target">
            <h2>{row['pref_name'] or "Unnamed Target"}</h2>
            <p><span class="label">ChEMBL ID:</span> {row['target_chembl_id']}</p>
            <p><span class="label">Organism:</span> {row['organism']}</p>
            <p><span class="label">Target Type:</span> {row['target_type']}</p>
            <p><span class="label">Description:</span> {row['description'] or "N/A"}</p>
        </div>
        """
    
    # Closing HTML tags
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save the HTML content to a file
    with open(html_file, 'w') as file:
        file.write(html_content)
    print(f"HTML file generated: {html_file}")

# Generate HTML after saving CSV
generate_html()
