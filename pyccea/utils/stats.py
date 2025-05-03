import logging
import numpy as np
from typing import Tuple
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene


# Initialize logger with info level
logging.basicConfig(level=logging.INFO)
# Reset handlers
logging.getLogger().handlers = []
# Add a custom handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(handler)


def statistical_comparison_between_independent_samples(
        x: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
        alternative: str = "two-sided"
) -> Tuple[str, float, float, bool]:
    """Perform a statistical comparison between two independent samples.

    This function compares two independent samples `x` and `y` using a
    statistical test (e.g., t-test) to determine if there is a significant
    difference between them. The test is two-sided by default, but can be
    configured to perform one-sided tests. It returns the test statistic
    name, the computed p-value, the test statistic, and whether the null
    hypothesis is rejected at the given significance level `alpha`.

    Parameters
    ----------
    x : np.ndarray
        The first independent sample.
    y : np.ndarray
        The second independent sample.
    alpha : float, optional
        The significance level for the hypothesis test (default is 0.05).
    alternative : str, optional
        Defines the alternative hypothesis. Options are 'two-sided', 
        'greater', or 'less' (default is 'two-sided').

    Returns
    -------
    test_name : str
        The name of the statistical test used (e.g., 't-test', 'mann-whitney-u').
    p_comparison : float
        The computed p-value of the test.
    statistic : float
        The test statistic from the hypothesis test.
    reject_null : bool
        True if the null hypothesis is rejected at the given significance level,
        False otherwise.

    Raises
    ------
    ValueError
        If the input arrays `x` and `y` have different lengths or are not 
        one-dimensional.
    ValueError
        If the `alternative` argument is not one of 'two-sided', 'greater', 
        or 'less'.
    """

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative hypothesis must be 'two-sided', 'greater', or 'less'.")

    logging.info(f"Sample sizes: x: {len(x)}, y: {len(y)}")

    if len(x) < 3:
        raise ValueError(f"1st sample has only {len(x)} observation(s).")
    if len(y) < 3:
        raise ValueError(f"2nd sample has only {len(y)} observation(s).")
    logging.info("")
    if len(x) >= 3 and len(y) >= 3:

        # Test for normality using Shapiro-Wilk test
        # Test for homogeneity of variances (homocedasticity) using Levene's test
        # The null-hypothesis of the Shapiro Wilk test is that the population is normally distributed.
        # Thus, if the p value is less than the chosen significance level, then the null hypothesis is
        # rejected and there is evidence that the data tested are not normally distributed.
        logging.info("Applying Shapiro-Wilk test to validate normality...")
        _, p_x_normality = shapiro(np.array(x))
        _, p_y_normality = shapiro(np.array(y))
        logging.info(f"p-value for sample x: {p_x_normality}.\n")
        logging.info(f"p-value for sample y: {p_y_normality}.\n")

        # The Levene test tests the null hypothesis that all input samples are from populations with equal variances.
        # Thus, if the p value is less than the chosen significance level, then the null hypothesis is rejected and
        # there is evidence that the samples are not from populations with equal variances.
        logging.info("Applying Levene's test to validate homocedasticity...")
        _, p_homocedasticity = levene(x, y)
        logging.info(f"p-value of Levene's test: {p_homocedasticity}.\n")

        # Decide which test to use based on normality and homocedasticity
        if (p_x_normality > alpha) and (p_y_normality > alpha) and (p_homocedasticity > alpha):
            logging.info("Applying t-test independent samples for comparison...")
            test_name = "t-test"
            statistic, p_comparison = ttest_ind(x, y, alternative=alternative)
        else:
            logging.info("Applying Mann-Whitney U rank test for comparison...")
            test_name = "mann-whitney-u"
            statistic, p_comparison = mannwhitneyu(x, y, alternative=alternative)

        # P-value small indicates statistical difference
        if p_comparison <= alpha:
            confidence_level = int((1-alpha)*100)
            alternative = 'different from' if alternative == 'two-sided' else f'{alternative} than'
            logging.info(f"There are statistical differences between the two samples with a confidence level of {confidence_level}%.")
            logging.info(f"The 1st sample is statistical {alternative} the 2nd sample.")
            reject_null = True
        else:
            logging.info("There are NOT statistical differences between the two samples.")
            reject_null = False
        logging.info(f"p-value of {p_comparison}.")

        return (test_name, statistic, p_comparison, reject_null)
