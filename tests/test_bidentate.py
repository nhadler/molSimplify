import helperFuncs as hp
from packaging import version
import openbabel


def test_bidentate(tmpdir):
    # There are two versions of this test depending on the openbabel version.
    # This is necessary because openbabel changed the numbering of atoms for v3.
    if version.parse(openbabel.__version__) > version.parse('3.0.0'):
        testName = "bidentate_v3"
    else:
        testName = "bidentate"
    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 1.5
    [passMultiFileCheck, pass_structures] = hp.runtestMulti(
        tmpdir, testName, threshMLBL, threshLG, threshOG)
    assert passMultiFileCheck
    for f, passNumAtoms, passMLBL, passLG, passOG, pass_report in pass_structures:
        print(f)
        assert passNumAtoms
        assert passMLBL
        assert passLG
        assert passOG
        assert pass_report
