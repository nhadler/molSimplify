import pytest
import json
import numpy as np
from molSimplify.Informatics.MOF.MOF_descriptors import get_MOF_descriptors
from molSimplify.utils.timer import DebugTimer


@pytest.fixture
def ref_names(depth=3):

    def RACs_names(starts, properties, depth, scope="all"):
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

    names = RACs_names(["f", "mc", "D_mc"], properties, depth)
    # f-lig does not include the "scope"
    names.extend(RACs_names(["f-lig"], properties, depth, scope=None))

    # Same for the starts that include the additional property alpha
    properties.append("alpha")
    names.extend(
        RACs_names(["lc", "D_lc", "func", "D_func"], properties, depth))
    return names


@pytest.mark.parametrize(
    "name",
    ["odac-21383.cif",
     "odac-21433.cif",
     "odac-21478.cif",
     "odac-21735.cif",
     "odac-21816.cif"])
def test_get_MOF_descriptors(resource_path_root, tmpdir, name, ref_names):
    with DebugTimer("get_MOF_descriptors()"):
        full_names, full_descriptors = get_MOF_descriptors(
            str(resource_path_root / "inputs" / "cif_files" / name),
            depth=3,
            path=str(tmpdir),
            xyzpath=str(tmpdir / "test.xyz"),
            Gval=True,
        )

    with open(resource_path_root / "refs" / "MOF_descriptors"
              / name.replace("cif", "json"), "r") as fin:
        ref = json.load(fin)

    assert full_names == ref_names
    np.testing.assert_allclose(full_descriptors, ref["descriptors"], atol=1e-6)
