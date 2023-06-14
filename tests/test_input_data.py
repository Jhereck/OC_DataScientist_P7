import pytest
import pandas as pd


@pytest.fixture(scope="module")
def data_test():
    """Get customers processed test data to feed into the tests"""
    return pd.read_csv("./datas/application_test.csv", index_col=[0])


def test_test_duplicates(data_test):
    """Test if the test duplicated dataframe is empty --> no duplicates"""
    duplicates = data_test[data_test.duplicated()]
    assert duplicates.empty
