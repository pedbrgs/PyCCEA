import pytest
import logging
import numpy as np
import pandas as pd
from unittest.mock import patch
from pyccea.utils.datasets import DataLoader
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold


# Disable logging output during tests
logging.disable(logging.CRITICAL)


def test_init_valid_data_conf(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with a valid configuration."""
    DataLoader("dummy_dataset", complete_data_conf)


def test_init_asserts_missing_data_conf_sections(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with configurations missing required sections."""
    for primary_key in complete_data_conf.keys():
        with pytest.raises(AssertionError):
            incomplete_data_conf = complete_data_conf.copy()
            incomplete_data_conf.pop(primary_key)
            DataLoader("dummy_dataset", incomplete_data_conf)


def test_parse_general_parameters_missing_splitter_type(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing splitter_type."""
    complete_data_conf["general"].pop("splitter_type")
    with pytest.raises(AssertionError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_general_parameters_invalid_splitter_type(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with invalid splitter_type."""
    complete_data_conf["general"]["splitter_type"] = "invalid_type"
    with pytest.raises(NotImplementedError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_splitter_parameters_missing_kfolds(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing kfolds."""
    complete_data_conf["splitter_type"] = "k_fold"
    del complete_data_conf["splitter"]["kfolds"]
    with pytest.raises(AssertionError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_splitter_parameters_preset_true_and_test_ratio(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with preset and test ratio."""
    complete_data_conf["splitter"]["preset"] = True
    complete_data_conf["splitter"]["test_ratio"] = 0.2
    dl = DataLoader("dummy_dataset", complete_data_conf)
    assert dl.test_ratio is None


def test_parse_splitter_parameters_missing_test_ratio_and_preset_false(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing test ratio when preset is false."""
    complete_data_conf["splitter"]["preset"] = False
    with pytest.raises(ValueError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_splitter_parameters_test_ratio_out_of_bounds(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with test ratio out of bounds."""
    complete_data_conf["splitter"]["preset"] = False
    for test_ratio in [-0.5, 1.5]:
        complete_data_conf["splitter"]["test_ratio"] = test_ratio
        with pytest.raises(ValueError):
            DataLoader("dummy_dataset", complete_data_conf)


def test_parse_normalization_parameters_missing_method(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with missing normalization method."""
    complete_data_conf["normalization"]["normalize"] = True
    complete_data_conf["normalization"].pop("method")
    with pytest.raises(AssertionError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_normalization_parameters_invalid_method(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with invalid normalization method."""
    complete_data_conf["normalization"]["normalize"] = True
    complete_data_conf["normalization"]["method"] = "invalid_method"
    with pytest.raises(NotImplementedError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_parse_normalization_parameters_method_and_normalize_false(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with normalization method when normalize is false."""
    complete_data_conf["normalization"]["normalize"] = False
    complete_data_conf["normalization"]["method"] = "min_max"
    with pytest.raises(ValueError):
        DataLoader("dummy_dataset", complete_data_conf)


def test_get_ready_calls_all_methods(mocker, complete_data_conf: dict) -> None:
    """Test get_ready method when normalize is True."""
    loader = DataLoader("dummy", complete_data_conf)

    # Mock _load to assign synthetic data
    def _mock_load():
        loader.data = pd.DataFrame({
            0: [1, 2, 3, 4, 5],
            1: [6, 7, 8, 9, 0],
            2: [0, 1, 0, 1, 1],
            3: ["train", "train", "train", "test", "test"]
        })

    mocker.patch.object(loader, "_load", side_effect=_mock_load)

    # Mock the other methods just to track their calls
    mocker.patch.object(loader, "_preprocess")
    mocker.patch.object(loader, "_split")
    mocker.patch.object(loader, "_normalize_subsets")
    mocker.patch.object(loader, "_model_selection")

    # Act
    loader.get_ready()

    # Assert: all methods called once
    loader._load.assert_called_once()
    loader._preprocess.assert_called_once()
    loader._split.assert_called_once()
    loader._normalize_subsets.assert_called_once()
    loader._model_selection.assert_called_once()


def test_get_ready_without_normalize(mocker, complete_data_conf: dict) -> None:
    """Test get_ready method when normalize is False."""
    complete_data_conf["normalization"]["normalize"] = False
    del complete_data_conf["normalization"]["method"]
    loader = DataLoader("dummy", complete_data_conf)

    # Mock _load to assign synthetic data
    def _mock_load():
        loader.data = pd.DataFrame({
            0: [1, 2, 3, 4, 5],
            1: [6, 7, 8, 9, 0],
            2: [0, 1, 0, 1, 1],
            3: ["train", "train", "train", "test", "test"]
        })

    mocker.patch.object(loader, "_load", side_effect=_mock_load)

    # Mock the other methods just to track their calls
    mocker.patch.object(loader, "_preprocess")
    mocker.patch.object(loader, "_split")
    mocker.patch.object(loader, "_normalize_subsets")
    mocker.patch.object(loader, "_model_selection")

    # Act
    loader.get_ready()

    # Assert: normalization should NOT be called
    loader._load.assert_called_once()
    loader._preprocess.assert_called_once()
    loader._split.assert_called_once()
    loader._normalize_subsets.assert_not_called()
    loader._model_selection.assert_called_once()


def test_load_data_invalid_dataset_name(complete_data_conf: dict) -> None:
    """Test DataLoader initialization with an invalid dataset name."""
    with pytest.raises(ValueError):
        dl = DataLoader("invalid_dataset", complete_data_conf)
        dl._load()


def test_load_dataset(monkeypatch, complete_data_conf: dict) -> None:
    """Test _load reads PARQUET correctly."""

    loader = DataLoader("dummy_dataset", complete_data_conf)

    # Patch os.path.dirname and os.path.join to return a dummy path
    monkeypatch.setattr("os.path.dirname", lambda _: "/dummy_dir")
    monkeypatch.setattr("os.path.join", lambda *args: "/dummy_dir/datasets/dummy.parquet")

    # Patch DataLoader.DATASETS to simulate dataset file mapping
    dummy_dataset_info = {"dummy_dataset": {"file": "dummy.parquet", "task": "regression"}}
    with patch.object(DataLoader, "DATASETS", dummy_dataset_info):

        # Mock pd.read_parquet to return a dummy DataFrame
        dummy_df = pd.DataFrame({
            "A": [1, 2],
            "B": [3, 4]
        })

        with patch("pandas.read_parquet", return_value=dummy_df) as mock_read_parquet:
            loader._load()

            # Assert pd.read_parquet called
            mock_read_parquet.assert_called_once_with("/dummy_dir/datasets/dummy.parquet")

            # Assert the data attribute got assigned
            pd.testing.assert_frame_equal(loader.data, dummy_df)


def test_get_input_and_output(complete_data_conf: dict) -> None:
    """Test _get_input and _get_output extract correct columns."""
    loader = DataLoader("dummy", complete_data_conf)
    loader.data = pd.DataFrame({
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: ["train", "test"]
    })

    X = loader._get_input()
    y = loader._get_output()

    expected_X = pd.DataFrame({0: [1, 2], 1: [3, 4]})
    expected_y = pd.Series([5, 6], name=2)

    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y)


def test_preprocess(monkeypatch, complete_data_conf: dict) -> None:
    """Test _preprocess replaces ? with NaN, drops rows if required."""
    loader = DataLoader("dummy", complete_data_conf)
    loader.data = pd.DataFrame({
        0: [1, "?", 3],
        1: [4, 5, None],
        2: [0, 1, 0],
        3: ["train", "test", "train"]
    })

    monkeypatch.setitem(DataLoader.DATASETS, "dummy", {"task": "classification"})

    loader._preprocess()

    # Check NaN replaced and rows dropped
    assert not loader.data.isin(["?"]).any().any()
    assert loader.n_examples == 1
    assert loader.n_features == 2
    assert loader.n_classes == 1
    assert loader.classes == [0]


def test_split_preset(monkeypatch, complete_data_conf: dict) -> None:
    """Test _split with preset splits."""
    loader = DataLoader("dummy", complete_data_conf)
    loader.data = pd.DataFrame({
        0: [1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 0],
        2: [0, 1, 0, 1, 1],
        3: ["train", "train", "train", "test", "test"]
    })
    loader.X = loader._get_input()
    loader.y = loader._get_output()

    loader._split()

    assert loader.X_train.shape[0] == 3
    assert loader.X_test.shape[0] == 2


def test_split_without_preset(complete_data_conf: dict) -> None:
    """Test _split without preset splits."""
    complete_data_conf["splitter"]["preset"] = False
    complete_data_conf["splitter"]["test_ratio"] = 0.3
    loader = DataLoader("dummy", complete_data_conf)
    n_samples = 100
    loader.data = pd.DataFrame({
        0: np.arange(n_samples),
        1: np.arange(n_samples, 2*n_samples),
        2: [0, 1] * 50,
        3: ["train"] * 70 + ["test"] * 30
    })
    loader.X = loader._get_input()
    loader.y = loader._get_output()

    loader._split()

    # Check sizes
    expected_train_size = int(n_samples * (1 - loader.test_ratio))
    expected_test_size = int(n_samples * loader.test_ratio)

    assert loader.X_train.shape[0] == expected_train_size
    assert loader.X_test.shape[0] == expected_test_size
    assert loader.y_train.shape[0] == expected_train_size
    assert loader.y_test.shape[0] == expected_test_size

    # Check no data loss
    total_samples = loader.X_train.shape[0] + loader.X_test.shape[0]
    assert total_samples == n_samples

    # Check subset size attributes
    assert loader.train_size == expected_train_size
    assert loader.test_size == expected_test_size


def test_min_max_normalization(complete_data_conf: dict) -> None:
    """Test min-max normalization of subsets."""
    # Min max normalization
    complete_data_conf["normalization"]["method"] = "min_max"
    loader = DataLoader("dummy", complete_data_conf)
    loader.data = pd.DataFrame({
        0: [1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 0],
        2: [0, 1, 0, 1, 1],
        3: ["train", "train", "train", "test", "test"]
    })
    loader.X = loader._get_input()
    loader.y = loader._get_output()
    loader._split()

    loader._normalize_subsets()

    assert np.allclose(loader.X_train, np.array([[0., 0.], [0.5, 0.5], [1., 1.]]))


def test_standard_normalization(complete_data_conf: dict) -> None:
    # Standard normalization
    complete_data_conf["normalization"]["method"] = "standard"
    loader = DataLoader("dummy", complete_data_conf)
    loader.data = pd.DataFrame({
        0: [1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 0],
        2: [0, 1, 0, 1, 1],
        3: ["train", "train", "train", "test", "test"]
    })
    loader.X = loader._get_input()
    loader.y = loader._get_output()
    loader._split()
    loader._normalize_subsets()
    # For standard scaler: mean 0, std 1 in training set
    assert np.allclose(loader.X_train.mean(axis=0), np.zeros(loader.X_train.shape[1]))
    assert np.allclose(loader.X_train.std(axis=0), np.ones(loader.X_train.shape[1]))


def test_stratified_kfold_model_selection(monkeypatch, complete_data_conf: dict) -> None:
    """Test _model_selection with k_fold splitter."""
    complete_data_conf["splitter"]["kfolds"] = 2
    loader = DataLoader("dummy", complete_data_conf)
    monkeypatch.setitem(DataLoader.DATASETS, "dummy", {"task": "classification"})
    loader.X_train = np.array([[1], [2], [3], [4], [5], [6]])
    loader.y_train = np.array([0, 1, 0, 1, 1, 0])

    loader._model_selection()

    assert len(loader.train_folds) == loader.kfolds
    assert len(loader.val_folds) == loader.kfolds
    assert isinstance(loader.splitter, StratifiedKFold)


def test_kfold_model_selection(monkeypatch, complete_data_conf: dict) -> None:
    """Test _model_selection with k_fold splitter."""
    complete_data_conf["splitter"]["kfolds"] = 2
    loader = DataLoader("dummy", complete_data_conf)
    monkeypatch.setitem(DataLoader.DATASETS, "dummy", {"task": "regression"})
    loader.X_train = np.array([[1], [2], [3], [4], [5], [6]])
    loader.y_train = np.array([0, 1, 0, 1, 1, 0])

    loader._model_selection()

    assert len(loader.train_folds) == loader.kfolds
    assert len(loader.val_folds) == loader.kfolds
    assert isinstance(loader.splitter, KFold)


def test_leave_one_out_model_selection(complete_data_conf: dict) -> None:
    """Test _model_selection with leave_one_out splitter."""
    complete_data_conf["splitter"]["kfolds"] = 1
    complete_data_conf["splitter"]["stratified"] = False
    complete_data_conf["general"]["splitter_type"] = "leave_one_out"
    loader = DataLoader("dummy", complete_data_conf)
    loader.data = pd.DataFrame({
        0: [1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 0],
        2: [0, 1, 0, 1, 1],
        3: ["train", "train", "train", "test", "test"]
    })
    loader.X = loader._get_input()
    loader.y = loader._get_output()
    loader._split()
    loader.X_train = np.array([[1], [2], [3], [4], [5], [6]])
    loader.y_train = np.array([0, 1, 0, 1, 1, 0])

    loader._model_selection()

    assert len(loader.train_folds) == len(loader.X_train)
    assert len(loader.val_folds) == len(loader.X_train)
    assert isinstance(loader.splitter, LeaveOneOut)