"""
Utility functions for the dept-similarity project.
"""

import re

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors

from dept_similarity.constants import ALLOWED_ELEMENTS

rdkit.RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings


def passed_cleanup(
    smiles,
    min_mass=120,
    max_mass=1200,
    allowed_elements: bool = True,
):
    """Check if a SMILES string can be parsed by RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mol_mass = Descriptors.MolWt(mol)

        unique_atoms = set(atom.GetSymbol() for atom in mol.GetAtoms())

        if allowed_elements and not unique_atoms.issubset(ALLOWED_ELEMENTS):
            return False

        # Check for neutral charge
        if Chem.GetFormalCharge(mol) != 0:
            return False

        # Check for reasonable molecular weight
        if min_mass <= mol_mass <= max_mass:
            return True

        return False

    except Exception:
        return False


def smiles_to_inchikey(smiles):
    """Convert a SMILES string to an InChIKey."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchikey = Chem.inchi.MolToInchiKey(mol)
        return inchikey
    except Exception:
        return None


"""Functions to allow filtering NMRShiftDB spectra"""


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def has_value(x):
    return x is not None and str(x).strip() != ""


def has_multiplicity_labels(spectrum_data: str) -> bool:
    """
    Detect S/D/T/Q multiplicity labels in pipe-delimited peak entries.

    Parses the structured `shift;intensity_with_mult;count` format so that
    incidental occurrences of these letters elsewhere in the string (e.g. in
    solvent names or metadata) do not cause false positives.
    """
    if not has_value(spectrum_data):
        return False

    valid_labels = {"s", "d", "t", "q"}

    for peak in spectrum_data.split("|"):
        fields = peak.strip().split(";")
        if len(fields) < 2:
            continue
        # Multiplicity is the trailing alpha characters of the second field
        # e.g. "0.0Q" → "q", "Q" → "q"
        mult = re.sub(r"[^a-zA-Z]", "", fields[1]).lower()
        if mult in valid_labels:
            return True

    return False


def count_assigned_peaks(spectrum_data: str) -> int:
    """
    Count assigned peaks in SPECTRUM_DATA.

    Example:
    '17.6;0.0Q;10|18.3;0.0T;0'
    """
    if not has_value(spectrum_data):
        return 0

    peaks = [p for p in spectrum_data.split("|") if p.strip()]

    return len(peaks)


def get_spectrum_score(entry: dict) -> int:
    """
    Score a spectrum entry for DEPT-like CH2 analysis.

    Higher score = better spectrum.
    """

    score = 0

    spectrum_type = str(entry.get("spectrum_type", "")).lower()
    spectrum_data = str(entry.get("spectrum", ""))
    dept_spectra = entry.get("signed_shifts", [])

    # -----------------------------------------
    # Prefer regular assigned 13C spectra
    # -----------------------------------------
    if "13c" in spectrum_type:
        score += 100

    # -----------------------------------------
    # Multiplicity labels S/D/T/Q
    # -----------------------------------------
    if has_multiplicity_labels(spectrum_data):
        score += 50

    # -----------------------------------------
    # More DEPT-like peaks = better
    # -----------------------------------------
    # n_peaks = count_assigned_peaks(spectrum_data)
    # score += 10 * n_peaks
    score += 10 * len(dept_spectra)

    return score
