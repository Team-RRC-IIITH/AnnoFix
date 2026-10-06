from annofix.sam_backend import device_kind


def test_device_kind_ignores_the_device_index():
    assert device_kind("cuda") == device_kind("cuda:0") == device_kind("cuda:1") == "cuda"
    assert device_kind("cpu") == "cpu"
