import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.dummy import DummyRegressor
import json
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import apairs_descriptor_calculator

alldata = {"data": {},
           "metrics": {
               "sumQAP": {},
               "spQAP": {},
               "hQAP": {},
               "baseline": {},
           }
           }


def read_logP_data():
    def read_exp_filename(filename):
        with open(pathlib.Path(f"./logP/logP_smi/training-8199/exps/{filename}.exp"), "r") as f:
            return float(f.readlines()[0])

    read_data = pd.read_csv("logP/logP_smi/train.csv")
    read_data["logP"] = read_data["filename"].apply(lambda x: read_exp_filename(str(x)))
    read_data = read_data[["smiles", "logP"]].copy()
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


def get_solubility_data():
    read_data = pd.read_csv("moleculenet/delaney-processed.csv")[
        ["smiles", "measured log solubility in mols per litre"]]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


def get_lipophilicity_data():
    read_data = pd.read_csv("moleculenet/Lipophilicity.csv")[["smiles", "exp"]]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


def get_IE_data():
    read_data = pd.read_csv("ionization_energy/IonEner-Pred/datasets/full_set/nist_organic_full_set.csv")[
        ["smiles", "IE"]]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


def get_bace_data():
    read_data = pd.read_csv("moleculenet/bace.csv")[["mol", "pIC50"]]
    read_data.columns = ["smiles", "target"]
    return read_data.copy()


def get_mp_data():
    suppl = Chem.SDMolSupplier('melting_point/13321_2018_263_MOESM1_ESM/OPERA_MP/TR_MP_6486.sdf')
    molsdict = {"smiles": [], "target": []}
    for mol in suppl:
        molsdict["smiles"].append(Chem.MolToSmiles(mol))
        molsdict["target"].append(mol.GetProp("MP"))
    suppl = Chem.SDMolSupplier('melting_point/13321_2018_263_MOESM1_ESM/OPERA_MP/TST_MP_2167.sdf')
    for mol in suppl:
        molsdict["smiles"].append(Chem.MolToSmiles(mol))
        molsdict["target"].append(mol.GetProp("MP"))

    allaP_mp = pd.read_excel("melting_point/11224_2021_1778_MOESM1_ESM.xlsx", sheet_name="Table S1").iloc[:, [1, 3]]
    molsdict["smiles"].extend(list(allaP_mp.iloc[:, 0]))
    molsdict["target"].extend(list(allaP_mp.iloc[:, 1]))

    mp_data = pd.DataFrame(molsdict).drop_duplicates(subset="smiles")
    return mp_data.copy()


alldata["data"]["logP"] = read_logP_data

alldata["data"]["pIC50"] = get_bace_data

alldata["data"]["solubility"] = get_solubility_data

alldata["data"]["lipophilicity"] = get_lipophilicity_data

alldata["data"]["ionization"] = get_IE_data

alldata["data"]["melting_point"] = get_mp_data

for name, dat in alldata["data"].items():
    data = dat()
    data = apairs_descriptor_calculator.reduce_to_applicability_domain(data["smiles"], data["target"])
    data = data.dropna()

    data_sum = apairs_descriptor_calculator.generate_apairs_descriptors(data["smiles"], data["target"],
                                                                        desc_type="sQAP")
    data_boomed = apairs_descriptor_calculator.generate_apairs_descriptors(data["smiles"], data["target"],
                                                                           desc_type="spQAP")
    data_hist = apairs_descriptor_calculator.generate_apairs_hist_descriptors(data["smiles"], data["target"])

    target = data_sum["target"]

    data_sum = data_sum.iloc[:, 5:]
    data_boomed = data_boomed.iloc[:, 5:]
    data_hist = data_hist.iloc[:, 5:]

    data_sum = data_sum.loc[:, data_sum.std() > 0.05]
    data_boomed = data_boomed.loc[:, data_boomed.std() > 0.05]
    data_hist = data_hist.loc[:, data_hist.std() > 0.05]

    rfr = RandomForestRegressor(n_estimators=500, max_features=0.3, random_state=1)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    baseline = DummyRegressor()

    alldata["metrics"]["baseline"][name] = cross_validate(baseline, data_sum, target, cv=cv,
                                                          scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                                   "r2"])
    alldata["metrics"]["sumQAP"][name] = cross_validate(rfr, data_sum, target, cv=cv,
                                                      scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                               "r2"])
    alldata["metrics"]["spQAP"][name] = cross_validate(rfr, data_boomed, target, cv=cv,
                                                       scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                                "r2"])
    alldata["metrics"]["hQAP"][name] = cross_validate(rfr, data_hist, target, cv=cv,
                                                      scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                               "r2"])

    print(f"{name} finished")

metrics = {k: {} for k in alldata["data"].keys()}
for apairs, dbs in alldata["metrics"].items():
    for db_name, scors in dbs.items():
        metrics[db_name][apairs] = {key: list(val) for key, val in scors.items()}

with open("regression_apairs.json", "w") as f:
    json.dump(metrics, f)

# Fingerprints

alldata = {
    "data": {},
    "metrics": {
        "MorganFP": {},
        "AtomPairs": {},
        "TopoTorsions": {},
        "RDK": {},
        "descriptors": {},
    }
}

alldata["data"]["logP"] = read_logP_data

alldata["data"]["pIC50"] = get_bace_data

alldata["data"]["solubility"] = get_solubility_data

alldata["data"]["lipophilicity"] = get_lipophilicity_data

alldata["data"]["ionization"] = get_IE_data

alldata["data"]["melting_point"] = get_mp_data


def getMolDescriptors(mol, missingVal=None):
    """calculate the full list of descriptors for a molecule

        missingVal is used if the descriptor cannot be calculated
    """
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


for name, dat in alldata["data"].items():
    data = dat()
    data = apairs_descriptor_calculator.reduce_to_applicability_domain(data["smiles"], data["target"])
    data = data.dropna()
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

    rfr = RandomForestRegressor(n_estimators=500, max_features=0.3, random_state=1)
    pipeline = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("predictor", rfr)
        ]
    )

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)

    alldata["metrics"]["MorganFP"][name] = cross_validate(rfr, mfp, target, cv=cv,
                                                          scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                                   "r2"])
    alldata["metrics"]["AtomPairs"][name] = cross_validate(rfr, ap, target, cv=cv,
                                                           scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                                    "r2"])
    alldata["metrics"]["TopoTorsions"][name] = cross_validate(rfr, tt, target, cv=cv,
                                                              scoring=["neg_mean_absolute_error",
                                                                       "neg_mean_squared_error", "r2"])
    alldata["metrics"]["RDK"][name] = cross_validate(rfr, rdk, target, cv=cv,
                                                     scoring=["neg_mean_absolute_error", "neg_mean_squared_error",
                                                              "r2"])
    alldata["metrics"]["descriptors"][name] = cross_validate(pipeline, desc, target, cv=cv,
                                                             scoring=["neg_mean_absolute_error",
                                                                      "neg_mean_squared_error", "r2"])

    print(f"{name} finished")

metrics = {k: {} for k in alldata["data"].keys()}
for apairs, dbs in alldata["metrics"].items():
    for db_name, scors in dbs.items():
        metrics[db_name][apairs] = {key: list(val) for key, val in scors.items()}

with open("regression_traditional.json", "w") as f:
    json.dump(metrics, f)
