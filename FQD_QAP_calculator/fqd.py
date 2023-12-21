from rdkit import Chem
import numpy as np
import pandas as pd
from typing import Union, List


def _prepare_fragments_db(quant_db) -> None:
        """
        Convert self._quant_db to pandas DataFrame of fragments database.
        The method does the following:
        1. read csv (self._quant_db must be either name of default or a path)
        2. create mol column containing Chem.Mol instances
        3. eventually add column with numbers of atoms of fragments
        """
        if quant_db == "qm9-extended-plus":
            _quant_db = pd.read_csv("qm_ext_plus.csv")
            _quant_db["mol"] = _quant_db["smiles"].apply(Chem.MolFromSmiles)
            _quant_db["num_atoms"] = _quant_db["mol"].apply(lambda x: x.GetNumAtoms())
            return _quant_db


QUANTUM_DB = _prepare_fragments_db("qm9-extended-plus")
FEATURES = "mu,alpha,homo,lumo,gap,zpve,u0,u298,h298,g298,cv".split(",")


class FQD():
    """
    A class intended to generate fragments quantum descriptors invented in the
    following paper:


    """

    def __init__(
                self,
                smiles: Union[str, List],
                features: Union[str, List[str]] = "all",
                atoms_range: Union[int, range] = 9,
            ):
        self.smiles = self._check_smiles(smiles)
        self.features = self._check_features(features)
        self.atoms_range = self._check_atoms_range(atoms_range)
        self.atoms_range = list(self.atoms_range)
        self._indices = None
        self._quant_db = QUANTUM_DB
        self._fqd_df = None
        self._fqd_indices = None


    @classmethod
    def generate_features(
        cls,
        smiles: str,
        features: Union[str, List[str]] = "all",
        atoms_range: int = 9,
        fqd_type: str = "all"
        ) -> pd.DataFrame:
        """[summary]
        Args:
            smiles (str): SMILES string of a molecule
            features (Union[str, List[str]]): a list of quantum features that will be calculated. If not stated all quantum features will be calculated
            atoms_range (int): a number of highest number
            fqd_type (str): type of FQDs one of 'quant' (quantitative), 'qual' (qualitative) or 'all'
        """
        ret = cls(smiles, features, atoms_range)
        ret.fqd_indices_search()
        if fqd_type == "quant":
            ret.generate_fqd_quant()
            return ret._fqd_df_quant
        elif fqd_type == "qual":
            ret.generate_fqd()
            return ret._fqd_df
        else:
            ret.generate_fqd_quant()
            ret.generate_fqd()
            return ret._fqd_df_quant.join(ret._fqd_df.drop(columns="smiles"))


    def _check_atoms_range(self, atoms_range) -> range:
        if isinstance(atoms_range, range):
            if atoms_range.start >= 2 and atoms_range.stop <= 10:
                return atoms_range
            else:
                raise ValueError("the boundaries of atoms_range defined by range are minumum 2 and maximum 10")
        elif isinstance(atoms_range, int):
            if atoms_range in range(2, 10):
                return range(2, atoms_range+1)
            else:
                raise ValueError("atoms_range must be a number between 2 and 9")
        else:
            raise TypeError("atoms_range must be an int in range 2 to 9 or type range of maximum range(2, 10).")


    def _check_features(self, features) -> List[str]:
        if isinstance(features, list):
            for feat in features:
                if not isinstance(feat, str):
                    raise TypeError("features must be a list of strings")
                if feat not in FEATURES:
                    raise ValueError(f"features must be one of {', '.join(FEATURES)}")
            return features
        elif isinstance(features, str):
            if features == "all":
                return FEATURES
            elif features not in FEATURES:
                raise ValueError(f"features must be one of {', '.join(FEATURES)}")
            return [features]
        else:
            raise TypeError("features must be a string or list of strings")


    def _check_smiles(self, smiles) -> List[str]:
        if isinstance(smiles, list):
            for smi in smiles:
                if not isinstance(smi, str):
                    raise TypeError("smiles must be a list of strings")
            return smiles
        elif isinstance(smiles, str):
            return [smiles]
        else:
            raise TypeError("smiles must be either a list of strings or string")


    def substructures_search(self, return_df=False) -> Union[None, pd.DataFrame]:
        import warnings
        if self._check_indices_present():
            print("Substructures already present.")
            return 
        indices = []
        failed = []
        for smiles in self.smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except:
                warnings.warn(f"{smiles} RDKit SMILES conversion to mol failed", Warning)
                failed.append(smiles)
                continue
            indices.append(self._get_indices(mol, self._quant_db.iloc[3:, :]))
        self._indices = pd.DataFrame(
            {
                "smiles": self.smiles,
                "indices": indices
                }
            )
        if len(failed) > 0:
            self.failed = failed
            warnings.warn("Some SMILES were not converted. They are stored in instance.failed attribute")
        if return_df:
            return self._indices.copy()


    def fqd_indices_search(self, return_df=False):
        import warnings
        if not self._check_fqd_indices_present():
            print("FQD indices already present.")
            return
        failed = []
        return_dict = {f"FQD_idx_{num_atoms}": [] for num_atoms in self.atoms_range}
        for smiles in self.smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except:
                warnings.warn(f"{smiles} RDKit SMILES conversion to mol failed", Warning)
                failed.append(smiles)
                continue
            for num_atoms in self.atoms_range:
                return_dict[f"FQD_idx_{num_atoms}"].append(
                    self._get_indices(
                        mol,
                        self._quant_db[self._quant_db["num_atoms"]==num_atoms]
                    )
                )
        self._fqd_indices = pd.DataFrame(
            {
                "smiles": self.smiles,
                **return_dict
                }
            )
        if len(failed) > 0:
            self.failed = failed
            warnings.warn("Some SMILES were not converted. They are stored in instance.failed attribute")
        if return_df:
            return self._fqd_indices.copy()


    def _get_indices(self, mol, fragments_db) -> Union[List[int], None]:
        indices = []
        for index, row in fragments_db.iterrows():
            if mol.HasSubstructMatch(row["mol"]):
                indices.append(index)
        return indices


    def _get_feature(self, prop, indices):
        import statistics
        if isinstance(indices, int) and indices == 0:
            return 0
        if isinstance(indices, list) and len(indices)==0:
            return None
        if indices is None or indices == np.NaN:
            return np.NaN
        ret = []
        for i in indices:
            ret.append(self._quant_db[prop][i])
        return statistics.mean(ret)


    def generate_fqd(self, quantum_properties="default", ):
        import itertools
        if self._check_fqd_indices_present:
            self.fqd_indices_search()
        if quantum_properties == "default":
            quantum_properties = self.features
        return_dict = {f"FQD_{num_atoms}_{prop}": [] for num_atoms, prop in
        itertools.product(self.atoms_range, quantum_properties)}
        for _, row in self._fqd_indices.iterrows():
            for num_atoms, prop in itertools.product(self.atoms_range, quantum_properties):
                return_dict[f"FQD_{num_atoms}_{prop}"].append(
                    self._get_feature(
                        prop, 
                        row[f"FQD_idx_{num_atoms}"]
                        )
                    )
        self._fqd_df = pd.DataFrame(
            {
                "smiles": self.smiles,
                **return_dict
                }
            )


    def _get_feature_quant(self, prop, indices, mol):
        import statistics
        if isinstance(indices, int) and indices == 0:
            return 0 
        if isinstance(indices, list) and len(indices)==0:
            return None
        if indices is None or indices == np.NaN:
            return np.NaN
        ret = []
        for i in indices:
            n_occurences = len(list(mol.GetSubstructMatch(self._quant_db["mol"][i])))
            ret.append(self._quant_db[prop][i] * n_occurences)
        return statistics.mean(ret)


    def generate_fqd_quant(self, quantum_properties="default", ):
        import itertools
        if self._check_fqd_indices_present:
            self.fqd_indices_search()
        if quantum_properties == "default":
            quantum_properties = self.features
        return_dict = {f"quant_FQD_{num_atoms}_{prop}": [] for num_atoms, prop in itertools.product(self.atoms_range, quantum_properties)}
        for _, row in self._fqd_indices.iterrows():
            for num_atoms, prop in itertools.product(self.atoms_range, quantum_properties):
                return_dict[f"quant_FQD_{num_atoms}_{prop}"].append(
                    self._get_feature_quant(prop, row[f"FQD_idx_{num_atoms}"], Chem.MolFromSmiles(row["smiles"]))
                    )
        self._fqd_df_quant = pd.DataFrame(
            {
                "smiles": self.smiles,
                **return_dict
                }
            )


    def _check_fqd_indices_present(self):
        import warnings
        if self._fqd_indices is None:
            if self._indices is None:
                return True
            else:
                warnings.warn("""Indices of substructures are not gouped by
                number of atoms in substructure. Will doo it now.""")
                self._convert_indices_to_fqd_indices()
                return False
        else: 
            return False


    def _check_indices_present(self):
        import warnings
        if self._indices is None:
            if self._fqd_indices is None:
                # warnings.warn("No substructures indices found. Will search them now")
                return False
            else:
                warnings.warn("""Found indices grouped by number of atoms. Will convert them now.""")
                self._wrap_fqd_indices()
                return True
        else:
            return True


    def _convert_indices_to_fqd_indices(self):
        return_dict = {f"FQD_idx_{num_atoms}": [[]] for num_atoms in self.atoms_range}
        for _, row in self._indices.iterrows():
            for idx in row["indices"]:
                num_atoms = self._quant_db["num_atoms"][idx]
                return_dict[f"FQD_idx_{num_atoms}"][0].append(idx)
        self._fqd_indices = pd.DataFrame(
            {
                "smiles": self.smiles,
                **return_dict
                }
            )


    def _wrap_fqd_indices(self):
        import itertools
        indices = []
        columns = [f"FQD_idx_{num_atoms}" for num_atoms in self.atoms_range]
        for _, row in self._fqd_indices.iterrows():
            ind = []
            for col in columns:
                ind.append(row[col])
            indices.append(list(itertools.chain.from_iterable(ind)))
        self._indices = pd.DataFrame(
            {
                "smiles": self.smiles,
                "indices": indices
                }
            )



def generate_features(
    smiles: str,
    features: Union[str, List[str]] = "all",
    atoms_range: int = 9,
    fqd_type: str = "all"
    ) -> pd.DataFrame:
    """[summary]

    Args:
        smiles (str): SMILES string of a molecule
        features (Union[str, List[str]]): a list of quantum features that will be calculated. If not stated all quantum features will be calculated
        atoms_range (int): a number of highest number 
    """
    return FQD.generate_features(smiles, features, atoms_range, fqd_type)

class NoIndicesError(Exception):
    pass
