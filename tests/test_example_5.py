import helperFuncs as hp
from packaging import version
import openbabel


def test_example_5(tmpdir):
    # There are two versions of this test depending on the openbabel version.
    # This is necessary because openbabel changed the numbering of atoms for v3.
    if version.parse(openbabel.__version__) > version.parse('3.0.0'):
        testName = "example_5_v3"
    else:
        testName = "example_5"
    threshMLBL = 0.1
    threshLG = 0.5
    threshOG = 1.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtest(
        tmpdir, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin


def test_example_5_No_FF(tmpdir):
    # There are two versions of this test depending on the openbabel version.
    # This is necessary because openbabel changed the numbering of atoms for v3.
    if version.parse(openbabel.__version__) > version.parse('3.0.0'):
        testName = "example_5_v3"
    else:
        testName = "example_5"
    threshMLBL = 0.1
    threshLG = 0.5
    threshOG = 1.0
    [passNumAtoms, passMLBL, passLG, passOG, pass_report, pass_qcin] = hp.runtestNoFF(
        tmpdir, testName, threshMLBL, threshLG, threshOG)
    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
    assert pass_report, pass_qcin
