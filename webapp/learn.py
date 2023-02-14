import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json
import joblib


def check_conf(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in "CONF":
            continue
        else:
            return False
    return True


def find_pattern_indexes(molecule: Chem.Mol, patterns_dataframe: pd.DataFrame):
    ret = []
    for index, row in patterns_dataframe.iterrows():
        if len(list(molecule.GetSubstructMatch(row["mol"]))) > 0:
            ret.append(index)
    return ret


def get_features(row, database: pd.DataFrame, features) -> None:
    for index in row["qm9_pattern_indexes"]:
        count = find_pattern_count(row["mol"], database["mol"][index])
        for feature in features:
            value = count * database[feature][index]
            row[feature] += value
    return row


def find_pattern_count(molecule: Chem.Mol, pattern: Chem.Mol) -> int:
    result = molecule.GetSubstructMatches(pattern)
    return len(result)


def train():
    chromophore_database = pd.read_csv("dbs/DB for chromophore_Sci_Data_rev02.csv")
    qm9 = pd.read_csv("dbs/qm9.csv")
    chromophore_database = chromophore_database[["Chromophore", "Solvent", "Emission max (nm)"]]
    chromophore_database = chromophore_database.dropna(subset=["Emission max (nm)"])
    mask = chromophore_database["Chromophore"] == chromophore_database["Solvent"]
    chromophore_database = chromophore_database[mask]
    chromophore_database["mol"] = chromophore_database["Chromophore"].apply(Chem.MolFromSmiles)

    chromophore_database["drop"] = chromophore_database["mol"].apply(check_conf)

    chromophore_database = chromophore_database[chromophore_database["drop"]]
    chromophore_database.drop(["drop", "Solvent"], inplace=True, axis=1)

    descriptors_names = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_names)
    descriptors = pd.DataFrame()
    maccs_keys = pd.DataFrame()
    morgan_fp = pd.DataFrame()
    for mol in chromophore_database["mol"]:
        descriptors = descriptors.append([calculator.CalcDescriptors(mol)])
        maccs_keys = maccs_keys.append(pd.Series(np.asarray(MACCSkeys.GenMACCSKeys(mol))), ignore_index=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_vect = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_vect)
        morgan_fp = morgan_fp.append(pd.Series(fp_vect), ignore_index=True)

    cols = list(chromophore_database.columns) + descriptors_names + [f"MACCS_key{x}" for x in
                                                                     range(maccs_keys.shape[1])] + [f"MorganFP_bit_{x}"
                                                                                                    for x in
                                                                                                    range(1024)]
    chromophore_database = pd.concat(
        [
            chromophore_database.reset_index(drop=True),
            descriptors.reset_index(drop=True),
            maccs_keys.reset_index(drop=True),
            morgan_fp.reset_index(drop=True),
        ], axis=1)
    chromophore_database.columns = cols

    trueinds = []
    for ind, val in (chromophore_database.iloc[:, 3:].std() == 0).items():
        if val:
            trueinds.append(ind)

    qm9["mol"] = qm9["smiles"].apply(Chem.MolFromSmiles)
    chromophore_database["qm9_pattern_indexes"] = chromophore_database["mol"].apply(find_pattern_indexes,
                                                                                    patterns_dataframe=qm9)
    features = list(qm9.columns.drop(["mol", "mol_id", "smiles"]))
    chromophore_database[features] = 0
    chromophore_database = chromophore_database.apply(get_features, database=qm9, features=features, axis=1)
    chromophore_database.drop(columns="qm9_pattern_indexes", inplace=True)

    X_train = chromophore_database.iloc[:, 3:].drop(columns=trueinds)
    features = X_train.columns

    descriptors = [x for x in features if x in descriptors_names]

    transformer = ColumnTransformer([("scaler", StandardScaler(), descriptors)], remainder="passthrough")

    X_train = transformer.fit_transform(X_train)

    X_train = pd.DataFrame(X_train, columns=features)

    y_train = chromophore_database["Emission max (nm)"]

    gbr = GradientBoostingRegressor(learning_rate=0.05,
                                    max_depth=4,
                                    n_estimators=1000, )

    pipeline = Pipeline([
        ("transformer", transformer),
        ("predictor", gbr)
    ]
    )

    pipeline.fit(X_train, y_train)

    with open("dbs/pipeline.sav", "wb") as f:
        joblib.dump(pipeline, f)

    with open("dbs/features.json", "w") as f:
        json.dump(
            {
                "features": features,
            },
            f
        )
