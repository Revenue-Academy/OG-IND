import multiprocessing
from distributed import Client, LocalCluster
import pytest
from pandas.testing import assert_frame_equal
import numpy as np
import os
from ogind.constants import DEFAULT_START_YEAR, TC_LAST_YEAR
from ogind import get_micro_data
from ogcore import utils
from taxcalc import GrowFactors

NUM_WORKERS = min(multiprocessing.cpu_count(), 7)


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()


def test_set_path():
    """
    Check that 'notapath.csv' is passed to taxcalc. An error
    containing 'notapath.csv' is sufficient proof for this
    """
    baseline = False
    start_year = 2016
    reform = {"II_em": {2017: 10000}}

    # In theory this path doesn't exist so there should be an IOError
    # But taxcalc checks if the path exists and if it doesn't, it tries
    # to read from an egg file. This raises a ValueError. At some point,
    # this could change. So I think it's best to catch both errors
    with pytest.raises((IOError, ValueError), match="notapath.csv"):
        get_micro_data.get_calculator(
            baseline,
            start_year,
            reform=reform,
            records_start_year=2021,
            data="notapath.csv",
        )


# def test_pit_path():
#     """
#     Check that setting `data` to None uses the pit.csv file from taxcalc
#     """
#     baseline = False
#     start_year = 2017
#     reform = {2020: {"_std_deduction": [10000]}}

#     with pytest.raises((IOError, ValueError), match="pit.csv"):
#         get_micro_data.get_calculator(
#             baseline,
#             start_year,
#             reform=reform,
#             records_start_year=2021,
#             data=None,
#         )


pit_reform_1 = {
    2020: {"_std_deduction": [50000]},
    2020: {"_rebate_thd": [500000]},
    2020: {"_rebate_ceiling": [12500]},
}


@pytest.mark.parametrize(
    "baseline,pit_reform",
    [(False, pit_reform_1), (False, {}), (True, pit_reform_1), (True, {})],
    ids=[
        "Reform, Policy change given",
        "Reform, No policy change given",
        "Baseline, Policy change given",
        "Baseline, No policy change given",
    ],
)
def test_get_calculator(baseline, pit_reform):
    calc = get_micro_data.get_calculator(
        baseline=baseline,
        calculator_start_year=2017,
        reform=pit_reform,
        data=None,
        gfactors=None,
        records_start_year=2017,
    )
    assert calc.current_year == 2017


def test_get_calculator_exception():
    pit_reform = {
    2020: {"_std_deduction": [50000]},
    2020: {"_rebate_thd": [500000]},
    2020: {"_rebate_ceiling": [12500]},
}
    with pytest.raises(Exception):
        assert get_micro_data.get_calculator(
            baseline=False,
            calculator_start_year=TC_LAST_YEAR + 1,
            reform=pit_reform,
            data=None,
            gfactors=GrowFactors(),
            records_start_year=2021,
        )


@pytest.mark.parametrize("baseline", [True, False], ids=["Baseline", "Reform"])
def test_get_data(baseline, dask_client):
    """
    Test of get_micro_data.get_data() function
    """
    test_data, _ = get_micro_data.get_data(
        baseline=baseline,
        start_year=2020,
        reform={},
        data=None,
        client=dask_client,
        num_workers=NUM_WORKERS,
    )

    assert test_data


def test_taxcalc_advance():
    """
    Test of the get_micro_data.taxcalc_advance() function
    """
    test_dict = get_micro_data.taxcalc_advance(True, 2022, {}, None, 2022)

    assert test_dict["year"][0] == 2022


@pytest.mark.local
def test_cap_inc_mtr():
    """
    Test of the get_micro_data.cap_inc_mtr() function

    """
    calc1 = get_micro_data.get_calculator(
        baseline=True, calculator_start_year=2022, reform={}, data=None
    )
    calc1.advance_to_year(2022)

    test_data = get_micro_data.cap_inc_mtr(calc1)
    print(test_data)

    assert np.any(test_data > 0.0)
