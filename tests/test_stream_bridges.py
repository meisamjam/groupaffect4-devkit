import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(path: str, name: str):
    spec = spec_from_file_location(name, Path(path))
    mod = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vicon_safe_json_loads():
    mod = _load_module("tools/vicon_nexus_lsl_bridge.py", "vicon_bridge")
    assert mod._safe_json_loads(b'{"motion":{"x":1}}')["motion"]["x"] == 1
    assert mod._safe_json_loads(b"not-json") is None


def test_tobii_parse_vector():
    mod = _load_module("tools/tobii_glasses_lsl_bridge.py", "tobii_bridge")
    assert mod._parse_vector([1, "2"], 2) == [1.0, 2.0]
    assert mod._parse_vector([1], 2) is None
    assert mod._parse_vector("bad", 2) is None


def test_emotibit_hardware_id_extraction():
    mod = _load_module("src/affectai_capture/devices/emotibit.py", "emotibit_bridge")
    assert mod._extract_emotibit_hardware_id(["abc", "MD-V7-0001141"]) == "MD-V7-0001141"
    assert mod._extract_emotibit_hardware_id([1, 2, 3]) is None


def test_raw_to_bids_stream_name_detection():
    mod = _load_module("tools/raw_to_bids.py", "raw_to_bids")
    assert mod._is_tobii_stream_name("Tobii_P1_stream") is True
    assert mod._is_tobii_stream_name("TobiiGlasses_legacy") is True
    assert mod._is_tobii_stream_name("Other") is False
    assert mod._is_emotibit_stream_name("Emotibit_P2_stream") is True
    assert mod._is_emotibit_stream_name("EmotiBit_192_168_1_10_ppg_green") is True
    assert mod._is_emotibit_stream_name("Other") is False
