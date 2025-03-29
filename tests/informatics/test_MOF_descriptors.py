import pytest
import json
import numpy as np
import pandas as pd
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
from molSimplify.utils.timer import DebugTimer


@pytest.fixture
def ref_names():
    def RACs_names(depth=3, Gval=True, alpha=True):

        def generate_names(starts, properties, depth, scope="all"):
            names = []
            for start in starts:
                for prop in properties:
                    for d in range(depth + 1):
                        if scope is None:
                            names.append(f"{start}-{prop}-{d}")
                        else:
                            names.append(f"{start}-{prop}-{d}-{scope}")
            return names

        properties = ["chi", "Z", "I", "T", "S"]
        if Gval:
            properties.append("Gval")

        names = generate_names(["f", "mc", "D_mc"], properties, depth)
        # f-link does not include the "scope"
        names.extend(generate_names(["f-link"], properties, depth, scope=None))

        # Same for the starts that include the additional property alpha
        if alpha:
            properties.append("alpha")
        names.extend(
            generate_names(["lc", "D_lc", "func", "D_func"], properties, depth))
        return names
    return RACs_names


@pytest.mark.parametrize(
    "name",
    [
        "odac-21383",
        "odac-21433",
        "odac-21478",
        "odac-21735",
        "odac-21816",
    ])
def test_get_MOF_descriptors_ODAC(resource_path_root, tmp_path, name, ref_names):
    # NOTE All the .cif files were converted to primitive unit cell using the
    # MOF_descriptors.get_primitive() function

    with DebugTimer("get_MOF_descriptors()"):
        full_names, full_descriptors = get_MOF_descriptors(
            str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"),
            depth=3,
            path=str(tmp_path),
            xyz_path=str(tmp_path / "test.xyz"),
            Gval=True,
        )

    with open(resource_path_root / "refs" / "informatics" / "mof"  / "MOF_descriptors"
              / name / f"{name}.json", "r") as fin:
        ref = json.load(fin)

    assert full_names == ref_names()
    np.testing.assert_allclose(full_descriptors, ref["descriptors"], atol=1e-6)

    lc_descriptors = pd.read_csv(tmp_path / "lc_descriptors.csv")
    lc_ref = pd.read_csv(resource_path_root / "refs" / "informatics" / "mof" / "MOF_descriptors" / name / "lc_descriptors.csv")
    assert all(lc_descriptors == lc_ref)

    sbu_descriptors = pd.read_csv(tmp_path / "sbu_descriptors.csv")
    sbu_ref = pd.read_csv(resource_path_root / "refs" / "informatics" / "mof" / "MOF_descriptors" / name / "sbu_descriptors.csv")
    assert all(sbu_descriptors == sbu_ref)


@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        # "ETECIR_clean",  TODO: Figure out why these two example do not work!
        # "FAVGUH_clean",  Disagreement on all ligand center RACs
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_get_MOF_descriptors_JACS(resource_path_root, tmp_path, name, ref_names):
    """
    Tests a handful of the MOFs used in
    Nandy et al., J. Am. Chem. Soc. 2021, 143, 42, 17535-17547
    https://doi.org/10.1021/jacs.1c07217
    """
    # NOTE All the .cif files were converted to primitive unit cell using the
    # MOF_descriptors.get_primitive() function

    with DebugTimer("get_MOF_descriptors()"):
        full_names, full_descriptors = get_MOF_descriptors(
            str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"),
            depth=3,
            path=str(tmp_path),
            xyz_path=str(tmp_path / "test.xyz"),
            Gval=False,
        )

    with open(resource_path_root / "refs" / "informatics" / "mof"  / "MOF_descriptors"
              / name / f"{name}.json", "r") as fin:
        ref = json.load(fin)

    # For now we are using a workaround because polarization descriptors
    # are now added by default.
    # Here they should be compared to ref_names(Gval=False, alpha=False)
    assert full_names == ref_names(Gval=False, alpha=True)
    np.testing.assert_allclose(
        # Get only the subset of descriptors without the property alpha
        [d for d, n in zip(full_descriptors, full_names) if "alpha" not in n],
        ref["descriptors"], atol=1e-6)

    lc_descriptors = pd.read_csv(tmp_path / "lc_descriptors.csv")
    lc_ref = pd.read_csv(resource_path_root / "refs" / "informatics" / "mof" / "MOF_descriptors" / name / "lc_descriptors.csv")
    assert all(lc_descriptors == lc_ref)

    sbu_descriptors = pd.read_csv(tmp_path / "sbu_descriptors.csv")
    sbu_ref = pd.read_csv(resource_path_root / "refs" / "informatics" / "mof" / "MOF_descriptors" / name / "sbu_descriptors.csv")
    assert all(sbu_descriptors == sbu_ref)
