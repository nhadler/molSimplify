import pytest
import json
import numpy as np
import pandas as pd
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
from molSimplify.utils.timer import DebugTimer


@pytest.fixture
def RACs_names(depth=3):

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

    properties = ["chi", "Z", "I", "T", "S", "Gval"]

    names = generate_names(["f", "mc", "D_mc"], properties, depth)
    # f-lig does not include the "scope"
    names.extend(generate_names(["f-lig"], properties, depth, scope=None))

    # Same for the starts that include the additional property alpha
    properties.append("alpha")
    names.extend(
        generate_names(["lc", "D_lc", "func", "D_func"], properties, depth))
    return names


@pytest.mark.parametrize(
    "name",
    [
        "odac-21383",
        "odac-21433",
        "odac-21478",
        "odac-21735",
        "odac-21816",
    ])
def test_get_MOF_descriptors(resource_path_root, tmpdir, name, RACs_names):
    # NOTE All the .cif files were converted to primitive unit cell using the
    # MOF_descriptors.get_primitive() function

    with DebugTimer("get_MOF_descriptors()"):
        full_names, full_descriptors = get_MOF_descriptors(
            str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"),
            depth=3,
            path=str(tmpdir),
            xyzpath=str(tmpdir / "test.xyz"),
            Gval=True,
        )

    with open(resource_path_root / "refs" / "MOF_descriptors"
              / name / f"{name}.json", "r") as fin:
        ref = json.load(fin)

    assert full_names == RACs_names
    np.testing.assert_allclose(full_descriptors, ref["descriptors"], atol=1e-6)

    lc_descriptors = pd.read_csv(tmpdir / "lc_descriptors.csv")
    lc_ref = pd.read_csv(resource_path_root / "refs" / "MOF_descriptors" / name / "lc_descriptors.csv")
    assert all(lc_descriptors == lc_ref)

    sbu_descriptors = pd.read_csv(tmpdir / "sbu_descriptors.csv")
    sbu_ref = pd.read_csv(resource_path_root / "refs" / "MOF_descriptors" / name / "sbu_descriptors.csv")
    assert all(sbu_descriptors == sbu_ref)
