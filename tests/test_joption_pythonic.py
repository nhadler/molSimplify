from molSimplify.Scripts.generator import startgen_pythonic
import helperFuncs as hp

def test_joption_pythonic(tmp_path, resource_path_root):
    out_dir = "cr_thd_2_cl_4_s_1/cr_thd_2_cl_4_s_1_conf_1/jobscript"
    input_dict_homo = {
        '-core': "cr",
        '-coord': str(4),
        '-oxstate': str(2),
        '-lig': str("cl"),
        '-geometry': "thd",
        '-ligocc': "4",
        '-rundir': str(tmp_path),
        '-runtyp': "minimize",
        '-keepHs': "yes",
        '-spin': str(1),
        '-jname': "{}_{}_{}_{}_hs_{}".format("cr", "thd", 2, "cl", 0),
        '-modules': "cuda,terachem",
        '-joption': "-fin terachem_input, -fin *.xyz, -fout scr/"
    }
    startgen_pythonic(input_dict_homo, write=True)
    with open(str(tmp_path) + "/" + out_dir, 'r') as f_in:
        data1 = f_in.readlines()
    with open(resource_path_root / "refs" / "joption_pythonic_jobscript", 'r') as f_in:
        data2 = f_in.readlines()
    for i, j in zip(data1, data2):
        assert i == j

def test_pythonic_metalloid_structure(tmp_path, resource_path_root):
    input_dict = {
        '-core': 'Sn',
        '-coord': '4',
        '-oxstate': '4',
        '-lig': 'nh3',
        '-ligocc': '4',
        '-geometry': 'square_planar'
    }
    _, _, this_diag = startgen_pythonic(input_dict)

    output_xyz = f'{tmp_path}/tin_complex.xyz'
    this_diag.mol.writexyz(output_xyz)

    ref_xyz = f'{resource_path_root}/refs/structgen/structgen_complex.xyz'

    threshMLBL = 0.1
    threshLG = 1.0
    threshOG = 2.0

    pass_xyz = hp.compareGeo(output_xyz, ref_xyz,
                          threshMLBL, threshLG, threshOG, transition_metals_only=False)
    [passNumAtoms, passMLBL, passLG, passOG] = pass_xyz

    assert passNumAtoms
    assert passMLBL
    assert passLG
    assert passOG
