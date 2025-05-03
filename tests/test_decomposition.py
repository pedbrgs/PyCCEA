import pytest
import numpy as np
from pyccea.decomposition import (
    FeatureGrouping,
    DummyFeatureGrouping,
    RandomFeatureGrouping,
    RankingFeatureGrouping,
    SequentialFeatureGrouping,
    ClusteringFeatureGrouping
)


def test_init_with_both_n_subcomps_and_subcomp_sizes() -> None:
    """Test error is raised when both n_subcomps and subcomp_sizes are provided."""
    with pytest.raises(AssertionError, match="Provide only one of the parameters"):
        FeatureGrouping(n_subcomps=3, subcomp_sizes=[2, 3])


def test_get_subcomponents_with_subcomp_sizes() -> None:
    """Test the decomposition of features into subcomponents when subcomp_sizes is provided."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    feature_grouping = FeatureGrouping(subcomp_sizes=[2, 4])
    subcomponents = feature_grouping._get_subcomponents(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[1, 2], [7, 8]]))
    assert np.array_equal(subcomponents[1], np.array([[3, 4, 5, 6], [9, 10, 11, 12]]))


def test_get_subcomponents_with_n_subcomps() -> None:
    """Test the decomposition of features into subcomponents when n_subcomps is provided."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    feature_grouping = FeatureGrouping(n_subcomps=3)
    subcomponents = feature_grouping._get_subcomponents(X)
    assert len(subcomponents) == 3
    assert np.array_equal(subcomponents[0], np.array([[1, 2], [7, 8]]))
    assert np.array_equal(subcomponents[1], np.array([[3, 4], [9, 10]]))
    assert np.array_equal(subcomponents[2], np.array([[5, 6], [11, 12]]))


def test_get_subcomponents_invalid_subcomp_sizes_shape() -> None:
    """Test error is raised when subcomp_sizes do not match the number of features in X."""
    X = np.random.rand(10, 5)  # 5 features
    feature_grouping = FeatureGrouping(subcomp_sizes=[2, 2])  # Sum is 4, mismatch with 5

    with pytest.raises(AssertionError, match="The sum of subcomponent sizes"):
        feature_grouping._get_subcomponents(X)


def test_dummy_decomposition() -> None:
    """Test the decomposition of features using DummyFeatureGrouping."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    feature_grouping = DummyFeatureGrouping(subcomp_sizes=[4, 2], feature_idxs=np.array([5, 0, 2, 3, 4, 1]))
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[6, 1, 3, 4], [12, 7, 9, 10]]))
    assert np.array_equal(subcomponents[1], np.array([[5, 2], [11, 8]]))
    assert np.array_equal(feature_idxs, np.array([5, 0, 2, 3, 4, 1]))


def test_random_decomposition() -> None:
    """Test the decomposition of features using RandomFeatureGrouping."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    feature_grouping = RandomFeatureGrouping(n_subcomps=2, seed=42)
    # The seed ensures that the random shuffle is reproducible
    # feature_idxs = np.array([0, 1, 5, 2, 4, 3]])
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[1, 2, 6], [7, 8, 12]]))
    assert np.array_equal(subcomponents[1], np.array([[3, 5, 4], [9, 11, 10]]))
    assert len(feature_idxs) == X.shape[1]


def test_sequential_decomposition() -> None:
    """Test the sequential decomposition of features using SequentialFeatureGrouping."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    feature_grouping = SequentialFeatureGrouping(subcomp_sizes=[2, 3, 1])
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 3
    assert np.array_equal(subcomponents[0], np.array([[1, 2], [7, 8]]))
    assert np.array_equal(subcomponents[1], np.array([[3, 4, 5], [9, 10, 11]]))
    assert np.array_equal(subcomponents[2], np.array([[6], [12]]))
    assert np.array_equal(feature_idxs, np.array([0, 1, 2, 3, 4, 5]))


def test_ranking_elitist_decomposition() -> None:
    """Test the decomposition of features using RankingFeatureGrouping when method is 'elitist'."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    scores = np.array([0.1, 0.5, 0.2, 0.4, 0.3, 0.6])
    feature_grouping = RankingFeatureGrouping(n_subcomps=2, scores=scores, method="elitist", ascending=False)
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[6, 2, 4], [12, 8, 10]]))
    assert np.array_equal(subcomponents[1], np.array([[5, 3, 1], [11, 9, 7]]))
    assert np.array_equal(feature_idxs, np.array([5, 1, 3, 4, 2, 0]))


def test_ranking_ascending_decomposition() -> None:
    """Test the decomposition of features using RankingFeatureGrouping when ascending is True."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    scores = np.array([0.1, 0.5, 0.2, 0.4, 0.3, 0.6])
    feature_grouping = RankingFeatureGrouping(subcomp_sizes=[3, 3], scores=scores, method="elitist", ascending=True)
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[1, 3, 5], [7, 9, 11]]))
    assert np.array_equal(subcomponents[1], np.array([[4, 2, 6], [10, 8, 12]]))
    assert np.array_equal(feature_idxs, np.array([0, 2, 4, 3, 1, 5]))


def test_ranking_distributed_decomposition() -> None:
    """Test the decomposition of features using RankingFeatureGrouping when method is 'distributed'."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    scores = np.array([0.1, 0.5, 0.2, 0.4, 0.3, 0.6])
    feature_grouping = RankingFeatureGrouping(n_subcomps=2, scores=scores, method="distributed", ascending=False)
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[6, 4, 3], [12, 10, 9]]))
    assert np.array_equal(subcomponents[1], np.array([[2, 5, 1], [8, 11, 7]]))
    assert np.array_equal(feature_idxs, np.array([5, 3, 2, 1, 4, 0]))


def test_ranking_invalid_method() -> None:
    """Test error is raised when an invalid method is provided."""
    scores = np.array([0.1, 0.5, 0.2, 0.4, 0.3, 0.6])
    with pytest.raises(AssertionError):
        RankingFeatureGrouping(n_subcomps=2, scores=scores, method="invalid_method")


def test_clustering_decomposition() -> None:
    """Test the decomposition of features using ClusteringFeatureGrouping."""
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12]])
    clusters = np.array([0, 0, 1, 1, 1, 0])
    feature_grouping = ClusteringFeatureGrouping(n_subcomps=2, clusters=clusters)
    subcomponents, feature_idxs = feature_grouping.decompose(X)
    assert len(subcomponents) == 2
    assert np.array_equal(subcomponents[0], np.array([[1, 2, 6], [7, 8, 12]]))
    assert np.array_equal(subcomponents[1], np.array([[3, 4, 5], [9, 10, 11]]))
    assert np.array_equal(feature_idxs, np.array([0, 1, 5, 2, 3, 4]))
    assert len(feature_idxs) == X.shape[1]
