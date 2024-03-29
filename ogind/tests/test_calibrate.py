"""
Tests of calibrate.py module
"""

import pytest
import numpy as np
import os
import ogcore
import json
from ogind.calibrate import Calibration

CUR_PATH = os.path.abspath(os.path.dirname(__file__))


def test_calibrate():
    p = ogcore.Specifications()
    p.update_specifications(
        json.load(
            open(os.path.join(CUR_PATH, "..", "ogind_default_parameters.json"))
        )
    )
    _ = Calibration(p)


def test_read_tax_func_estimate_error():
    with pytest.raises(RuntimeError):
        p = ogcore.Specifications()
        p.tax_func_type = "linear"
        tax_func_path = os.path.join(
            CUR_PATH, "test_io_data", "TxFuncEst_policy.pkl"
        )
        c = Calibration(p)
        _, _ = c.read_tax_func_estimate(p, tax_func_path)


def test_read_tax_func_estimate():
    p = ogcore.Specifications()
    p.update_specifications(
        json.load(
            open(os.path.join(CUR_PATH, "..", "ogind_default_parameters.json"))
        )
    )
    p.BW = 11
    tax_func_path = os.path.join(
        CUR_PATH, "test_io_data", "TxFuncEst_policy.pkl"
    )
    c = Calibration(p)
    dict_params, _ = c.read_tax_func_estimate(p, tax_func_path)
    print("Dict keys = ", dict_params.keys())

    assert isinstance(dict_params["tfunc_etr_params_S"], np.ndarray)


def test_get_dict():
    p = ogcore.Specifications()
    p.update_specifications(
        json.load(
            open(os.path.join(CUR_PATH, "..", "ogind_default_parameters.json"))
        )
    )
    c = Calibration(p)
    c_dict = c.get_dict()

    assert isinstance(c_dict, dict)
