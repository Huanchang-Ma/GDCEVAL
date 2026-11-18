import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os


def draw_molecules_from_csv(file_path, save_path='mol_vis_test'):
    """
    Read SMILES from a CSV file and draw the corresponding molecules.

    Parameters:
        file_path (str): Path to the CSV file.
        save_path (str): Directory to save the molecule images.
    """
    # Create the save directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if 'smiles' column exists
    if 'smiles' not in df.columns:
        raise ValueError("The CSV file must contain a 'smiles' column.")

    # Iterate through the SMILES column
    for idx, smiles in enumerate(df['smiles']):
        try:
            # Convert SMILES to a molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES at index {idx}: {smiles}")
                continue

            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)

            # Draw the molecule
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(f"{save_path}/molecule_{idx}.png")

            print(f"Saved molecule image for index {idx}.")

        except Exception as e:
            print(f"Error processing SMILES at index {idx}: {smiles}. Error: {e}")


# Example usage
# csv_file_path = '/home/exp2/Diffusion_Model/reGen_Molbace/reGen/data/ogbg-molbace.csv'
csv_file_path = '/home/mhc/Diffusion_Model/reGen_Molbace/reGen/data/ogbg-molfreesolv.csv'
draw_molecules_from_csv(csv_file_path)