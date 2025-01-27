import pytest
from molSimplify.Informatics.MOF.MOF_functionalizer import functionalize_MOF
from molSimplify.Informatics.MOF.PBC_functions import readcif
import numpy as np
import os

@pytest.mark.parametrize(
    "num_func, func_group",
    [
        (1, "CH3"),
        (2, "CH3"),
        (1, "CN"),
        (2, "CN"),
        (1, "F"),
        (2, "F"),
        (1, "I"),
        (2, "I"),
        (1, "NO2"),
        (2, "NO2"),
        (1, "OH"),
        (2, "OH"),
    ])
def test_fg_addition(resource_path_root, tmpdir, num_func, func_group):
    starting_cif = str(resource_path_root / "inputs" / "cif_files" / "UiO-66.cif")
    destination_path = str(tmpdir / "functionalized_MOF")
    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)
        
    functionalize_MOF(
        starting_cif,
        destination_path,
        path_between_functionalizations=3,
        functionalization_limit=num_func,
        functional_group=func_group
        )

    # Check the structure with functional groups added
    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / f"functionalized_UiO-66_{func_group}_{num_func}.cif")
    cpar1, allatomtypes1, fcoords1 = readcif(str(destination_path / "cif" / f"functionalized_UiO-66_{func_group}_{num_func}.cif"))
    cpar2, allatomtypes2, fcoords2 = readcif(reference_cif_path)

    assert np.array_equal(cpar1, cpar2)
    assert allatomtypes1 == allatomtypes2
    assert np.array_equal(fcoords1, fcoords2)
