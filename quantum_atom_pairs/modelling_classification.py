from sklearn.dummy import DummyClassifier
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
import json
import apairs_descriptor_calculator

alldata = {"data": {},
           "metrics": {
               "sumQAP": {},
               "spQAP": {},
               "hQAP": {},
               "baseline": {},
           }
           }


# Read data

## BBBP data

def read_bbbp_data():
    read_data = pd.read_csv("BBBP/ci300124c_si_001.txt", sep="\t", encoding="latin-1")[["smiles", "p_np"]]
    read_data.columns = ["smiles", "target"]
    read_data.replace({"n": 0, "p": 1}, inplace=True)
    return read_data.copy()


## mutagenicity

def read_ames_data():
    read_data = pd.read_csv("ames_mutagenicity/Ames_smi/Ames.csv")["Canonical_Smiles,Activity".split(",")]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


## hERG

def read_herg_data():
    read_data = pd.read_excel("hERG/supplementary file 1.xlsx", sheet_name="Classification based on IC50", header=1)[
        ["SMILES", "class"]]
    read_data.columns = ["smiles", "target"]
    read_data.replace({-1: 0}, inplace=True)
    return read_data


## ClinTox

def read_clintox_data():
    read_data = pd.read_csv("clinTox/ClinTox_smi/ClinTox.csv")[["smiles", "CT_TOX"]]
    read_data.columns = ["smiles", "target"]
    return read_data


## HIV

def read_hiv_data():
    read_data = pd.read_csv("moleculenet/HIV.csv")["smiles,HIV_active".split(",")]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


# alldata["data"]["hiv"] = read_hiv_data

alldata["data"]["bbbp"] = read_bbbp_data

alldata["data"]["ames"] = read_ames_data

alldata["data"]["herg"] = read_herg_data

alldata["data"]["clintox"] = read_clintox_data

# modelling


for name, dat in alldata["data"].items():
    data = dat()
    data = data.dropna()
    data = apairs_descriptor_calculator.reduce_to_applicability_domain(data["smiles"], data["target"])

    data_sum = apairs_descriptor_calculator.generate_apairs_descriptors(data["smiles"], data["target"], desc_type="sQAP")
    data_boomed = apairs_descriptor_calculator.generate_apairs_descriptors(data["smiles"], data["target"],
                                                                           desc_type="spQAP")
    data_hist = apairs_descriptor_calculator.generate_apairs_hist_descriptors(data["smiles"], data["target"])

    data_sum.dropna(inplace=True)
    data_boomed.dropna(inplace=True)
    data_hist.dropna(inplace=True)

    target = data_sum["target"]

    data_sum = data_sum.iloc[:, 5:]
    data_boomed = data_boomed.iloc[:, 5:]
    data_hist = data_hist.iloc[:, 5:]

    data_sum = data_sum.loc[:, data_sum.std() > 0.05]
    data_boomed = data_boomed.loc[:, data_boomed.std() > 0.05]
    data_hist = data_hist.loc[:, data_hist.std() > 0.05]

    rfr = RandomForestClassifier(n_estimators=500, max_features=0.3, random_state=1)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    baseline = DummyClassifier(strategy="stratified")

    alldata["metrics"]["sumQAP"][name] = cross_validate(rfr, data_sum, target, cv=cv,
                                                     scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["spQAP"][name] = cross_validate(rfr, data_boomed, target, cv=cv,
                                                        scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["hQAP"][name] = cross_validate(rfr, data_hist, target, cv=cv,
                                                      scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["baseline"][name] = cross_validate(baseline, data_hist, target, cv=cv,
                                                          scoring=["balanced_accuracy", "f1", "roc_auc"])

    print(f"{name} finished")

metrics = {k: {} for k in alldata["data"].keys()}
for apairs, dbs in alldata["metrics"].items():
    for db_name, scors in dbs.items():
        metrics[db_name][apairs] = {key: list(val) for key, val in scors.items()}


with open("classification_apairs.json", "w") as f:
    json.dump(metrics, f)


alldata = {"data": {},
           "metrics": {
               "MorganFP": {},
               "AtomPairs": {},
               "TopoTorsions": {},
               "RDK": {},
               "descriptors": {},
           }
           }


def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule

        missingVal is used if the descriptor cannot be calculated
    '''
    from rdkit.Chem import Descriptors
    res = {}
    for nm, fn in Chem.Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


# Read data

## BBBP data

def read_bbbp_data():
    read_data = pd.read_csv("BBBP/ci300124c_si_001.txt", sep="\t", encoding="latin-1")[["smiles", "p_np"]]
    read_data.columns = ["smiles", "target"]
    read_data.replace({"n": 0, "p": 1}, inplace=True)
    return read_data.copy()


## mutagenicity

def read_ames_data():
    read_data = pd.read_csv("ames_mutagenicity/Ames_smi/Ames.csv")["Canonical_Smiles,Activity".split(",")]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


## hERG

def read_herg_data():
    read_data = pd.read_excel("hERG/supplementary file 1.xlsx", sheet_name="Classification based on IC50", header=1)[
        ["SMILES", "class"]]
    read_data.columns = ["smiles", "target"]
    read_data.replace({-1: 0}, inplace=True)
    return read_data


## ClinTox

def read_clintox_data():
    read_data = pd.read_csv("clinTox/ClinTox_smi/ClinTox.csv")[["smiles", "CT_TOX"]]
    read_data.columns = ["smiles", "target"]
    return read_data


## HIV

def read_hiv_data():
    read_data = pd.read_csv("moleculenet/HIV.csv")["smiles,HIV_active".split(",")]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


# alldata["data"]["hiv"] = read_hiv_data

alldata["data"]["bbbp"] = read_bbbp_data

alldata["data"]["ames"] = read_ames_data

alldata["data"]["herg"] = read_herg_data

alldata["data"]["clintox"] = read_clintox_data

# modelling


for name, dat in alldata["data"].items():
    data = dat()
    data = data.dropna()
    data = apairs_descriptor_calculator.reduce_to_applicability_domain(data["smiles"], data["target"])
    target = data["target"]

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=4096)
    apgen = rdFingerprintGenerator.GetAtomPairGenerator(maxDistance=4, fpSize=1089)
    ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=4096)
    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=4, fpSize=4096)

    mfp = [mfpgen.GetFingerprintAsNumPy(m) for m in data["mol"]]
    ap = [apgen.GetFingerprintAsNumPy(m) for m in data["mol"]]
    tt = [ttgen.GetFingerprintAsNumPy(m) for m in data["mol"]]
    rdk = [rdkgen.GetFingerprintAsNumPy(m) for m in data["mol"]]
    desc = descriptors = [descr_dict for descr_dict in data["mol"].apply(getMolDescriptors)]

    mfp = pd.DataFrame(mfp, columns=[f"MFP{x}" for x in range(4096)])
    ap = pd.DataFrame(ap, columns=[f"AP{x}" for x in range(1089)])
    tt = pd.DataFrame(tt, columns=[f"TT{x}" for x in range(4096)])
    rdk = pd.DataFrame(rdk, columns=[f"RDK{x}" for x in range(4096)])
    desc = pd.DataFrame(desc)
    desc.dropna(axis=1, inplace=True)

    mfp = mfp.loc[:, mfp.std() > 0.05]
    ap = ap.loc[:, ap.std() > 0.05]
    tt = tt.loc[:, tt.std() > 0.05]
    rdk = rdk.loc[:, rdk.std() > 0.05]
    desc = desc.loc[:, desc.std() > 0.05]

    rfr = RandomForestClassifier(n_estimators=500, max_features=0.3, random_state=1)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("predictor", rfr)
        ]
    )

    alldata["metrics"]["MorganFP"][name] = cross_validate(rfr, mfp, target, cv=cv,
                                                          scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["AtomPairs"][name] = cross_validate(rfr, ap, target, cv=cv,
                                                           scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["TopoTorsions"][name] = cross_validate(rfr, tt, target, cv=cv,
                                                              scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["RDK"][name] = cross_validate(rfr, rdk, target, cv=cv,
                                                     scoring=["balanced_accuracy", "f1", "roc_auc"])
    alldata["metrics"]["descriptors"][name] = cross_validate(pipeline, desc, target, cv=cv,
                                                             scoring=["balanced_accuracy", "f1", "roc_auc"])

    print(f"{name} finished")

metrics = {k: {} for k in alldata["data"].keys()}
for apairs, dbs in alldata["metrics"].items():
    for db_name, scors in dbs.items():
        metrics[db_name][apairs] = {key: list(val) for key, val in scors.items()}


with open("classification_traditional.json", "w") as f:
    json.dump(metrics, f)
