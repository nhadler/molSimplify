import pytest
from molSimplify.__main__ import main


def test_main_empty(tmpdir, resource_path_root):
    main(args=[f"-rundir {tmpdir}"])
    with open(tmpdir / "fe_oct_2_water_6_s_5" /
              "fe_oct_2_water_6_s_5_conf_1" /
              "fe_oct_2_water_6_s_5_conf_1.report", "r") as fin:
        lines = fin.readlines()
    with open(resource_path_root / "refs" / "test_cli" /
              "fe_oct_2_water_6_s_5_conf_1.report", "r") as fin:
        lines_ref = fin.readlines()
    assert lines == lines_ref


@pytest.mark.skip("Test for help not working yet.")
def test_help(capsys):
    main(args=["--help",])
    captured = capsys.readouterr()
    print(captured.out)
    assert "Welcome to molSimplify. Only basic usage is described here." in captured.out
