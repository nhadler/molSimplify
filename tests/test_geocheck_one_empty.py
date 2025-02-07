import helperFuncs as hp


def test_example_1(tmp_path, resource_path_root):
    testName = "one_empty_good"
    thresh = 0.01
    passGeo = hp.runtestgeo(tmp_path, resource_path_root, testName, thresh, geo_type="one_empty")
    assert passGeo


def test_example_2(tmp_path, resource_path_root):
    testName = "one_empty_bad"
    thresh = 0.01
    passGeo = hp.runtestgeo(tmp_path, resource_path_root, testName, thresh, geo_type="one_empty")
    assert passGeo
