import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.SaltRemover import SaltRemover
import itertools
import statistics


def rem_salts_ret_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        remover = SaltRemover(defnData="[Cl,Br,I]")
        stripped = remover.StripMol(mol)
        return stripped
    except:
        return None


def generate_apairs_descriptors(smiles, target=None, max_distance=4, desc_type="sum"):
    if target is None:
        database = pd.DataFrame(smiles, columns=["smiles"])
    else:
        database = pd.DataFrame({"smiles": smiles, "target": target})

    mols = database["smiles"].apply(rem_salts_ret_mol)
    mols.dropna(inplace=True)
    fpgen = rdFingerprintGenerator.GetAtomPairGenerator(1, max_distance)
    apairs = mols.apply(lambda x: fpgen.GetSparseCountFingerprint(x).GetNonzeroElements())

    qpair_db = pd.read_csv("apair_db.csv")
    qpair_db.index = qpair_db["Unnamed: 0"]
    qpair_db.drop(columns="Unnamed: 0", inplace=True)

    apairs_applicability = apairs.apply(lambda x: set(x.keys()).issubset(set(qpair_db.index)))

    cols = list(database.columns) + "mol,apairs,apairs_applicability".split(",")
    database = pd.concat([database, mols, apairs, apairs_applicability], axis=1)
    database.columns = cols
    database.dropna(inplace=True)

    qprops = list(qpair_db.columns)

    if desc_type == "sum":
        def generate(apairs: dict, prop: str):
            qprop = []
            num_apairs = len(apairs.keys())
            if num_apairs < 1:
                return None
            for apair, num in apairs.items():
                qprop.append((qpair_db[prop][apair]**2 * num) / num_apairs)
            return statistics.mean(qprop)

        desc = {qprop: 0 for qprop in qprops}

        for qprop in qprops:
            desc[qprop] += database[database["apairs_applicability"]]["apairs"].apply(generate, prop=qprop)

        cols = list(database.columns) + qprops
        database = pd.concat([database, pd.DataFrame(desc)], axis=1)
        database.columns = cols
        return database

    elif desc_type == "atomic":
        def generate(mol, apairs):
            atoms_list = list(mol.GetAtoms())
            for atom in atoms_list:
                atom.atom_pairs = set()
            for at1, at2 in itertools.product(atoms_list, repeat=2):
                atom_aprs = []
                for dist in range(1, max_distance + 1):
                    pair = Pairs.pyScorePair(at1, at2, dist)
                    atom_aprs.append(pair)
                atom_aprs = [pair for pair in atom_aprs if pair in apairs.keys()]
                at1.atom_pairs |= set(atom_aprs)
            ret = {qprop: 0 for qprop in qprops}
            for atom in atoms_list:
                for index, qprop in itertools.product(atom.atom_pairs, qprops):
                    ret[qprop] += qpair_db[qprop][index]**2
            return ret

        descs = database[database["apairs_applicability"]].apply(lambda x: generate(x["mol"], x["apairs"]), axis=1)
        database = pd.concat([database, pd.DataFrame(list(descs), index=descs.index)], axis=1)
        return database

    elif desc_type == "boomed":
        def generate(apairs):
            unique_apairs = qpair_db.index
            apairs_desc = {f"{x}_{qprop}": 0 for x, qprop in itertools.product(unique_apairs, qprops)}
            for k, v in apairs_desc.items():
                key, qprop = int(k.split("_")[0]), k.split("_")[1]
                if key in apairs.keys():
                    apairs_desc[k] += apairs[key] * qpair_db[qprop][key]**2
            return apairs_desc

        apairs_desc = apairs.apply(generate)

        apairs_desc = pd.DataFrame(list(apairs_desc), index=apairs_desc.index)
        database = pd.concat([database, apairs_desc], axis=1)
        database = database[database["apairs_applicability"]]
        return database

    else:
        raise "Error"


def generate_apairs_hist_descriptors(smiles, target=None, max_distance=4):
    if target is None:
        database = pd.DataFrame(smiles, columns=["smiles"])
    else:
        database = pd.DataFrame({"smiles": smiles, "target": target})

    mols = database["smiles"].apply(rem_salts_ret_mol)
    mols.dropna(inplace=True)
    fpgen = rdFingerprintGenerator.GetAtomPairGenerator(1, max_distance)
    apairs = mols.apply(lambda x: fpgen.GetSparseCountFingerprint(x).GetNonzeroElements())

    qpair_db = pd.read_csv("apair_hist_db.csv", index_col=0)

    apairs_applicability = apairs.apply(lambda x: set(x.keys()).issubset(set(qpair_db.index)))

    cols = list(database.columns) + "mol,apairs,apairs_applicability".split(",")
    database = pd.concat([database, mols, apairs, apairs_applicability], axis=1)
    database.columns = cols
    database.dropna(inplace=True)

    def generate(apairs):
        apairs = dict(apairs)
        try:
            dc = pd.concat([qpair_db.loc[k, :] * v for k, v in apairs.items()], axis=1).T.sum()
            return dc
        except:
            return np.empty((len(qpair_db.columns)))

    apairs_desc = database[database["apairs_applicability"]]["apairs"].apply(generate)

    database = pd.concat([database, apairs_desc], axis=1)
    return database.dropna()


def reduce_to_applicability_domain(smiles, target, max_distance=4):
    if target is None:
        database = pd.DataFrame(smiles, columns=["smiles"])
    else:
        database = pd.DataFrame({"smiles": smiles, "target": target})

    mols = database["smiles"].apply(rem_salts_ret_mol)
    mols.dropna(inplace=True)
    fpgen = rdFingerprintGenerator.GetAtomPairGenerator(1, max_distance)
    apairs = mols.apply(lambda x: fpgen.GetSparseCountFingerprint(x).GetNonzeroElements())

    qpair_db = pd.read_csv("apair_hist_db.csv", index_col=0)

    def check_applicability(apairs_dict):
        apairs_set = set(apairs_dict.keys())
        if len(apairs_set) > 0:
            return apairs_set.issubset(set(qpair_db.index))
        else:
            return False

    apairs_applicability = apairs.apply(check_applicability)

    cols = list(database.columns) + "mol,apairs,apairs_applicability".split(",")
    database = pd.concat([database, mols, apairs, apairs_applicability], axis=1)
    database.columns = cols
    database.dropna(inplace=True)
    return database[database["apairs_applicability"]]
