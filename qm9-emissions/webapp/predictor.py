import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys
import learn
import joblib
import json


def predict(smiles: str) -> str:
    try:
        with open("dbs/pipeline.sav", "rb") as f:
            pipeline = joblib.load(f)
        with open("dbs/features.json", "r") as f:
            features = json.load(f)
            features = features["features"]
    except:
        raise "Error try running predictor.train() first"

    molecule_data_frame = pd.DataFrame()
    molecule_data_frame["mol"] = pd.Series(Chem.MolFromSmiles(smiles))
    mol = molecule_data_frame["mol"][0]

    descriptors_names = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_names)
    descriptors = pd.DataFrame()
    descriptors = descriptors.append([calculator.CalcDescriptors(mol)])
    maccs_keys = pd.Series(np.asarray(MACCSkeys.GenMACCSKeys(mol)))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_vect = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_vect)
    morgan_fp = pd.Series(fp_vect)

    cols = list(molecule_data_frame.columns) + descriptors_names + [f"MACCS_key{x}" for x in
                                                                    range(maccs_keys.shape[0])] + [f"MorganFP_bit_{x}"
                                                                                                   for x in range(1024)]
    molecule_data_frame = pd.concat(
        [
            molecule_data_frame,
            descriptors.reset_index(drop=True),
            pd.DataFrame(maccs_keys).T,
            pd.DataFrame(morgan_fp).T,
        ],
        axis=1
    )
    molecule_data_frame.columns = cols
    qm9 = pd.read_csv("dbs/qm9.csv")
    qm9["mol"] = qm9["smiles"].apply(Chem.MolFromSmiles)
    molecule_data_frame["qm9_pattern_indexes"] = molecule_data_frame["mol"].apply(learn.find_pattern_indexes,
                                                                                    patterns_dataframe=qm9)
    qfeatures = list(qm9.columns.drop(["mol", "mol_id", "smiles"]))
    molecule_data_frame[qfeatures] = 0
    molecule_data_frame = molecule_data_frame.apply(learn.get_features, database=qm9, features=qfeatures, axis=1)
    molecule_data_frame.drop(columns="qm9_pattern_indexes", inplace=True)

    return f"""Predicted wavelength: {pipeline.predict(molecule_data_frame[features])[0]:.1f} nm"""
