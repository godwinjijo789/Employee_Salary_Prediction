from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrelationResult:
    method: str
    coefficient: float
    p_value: float
    sample_size: int
    confidence_interval: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class GroupComparisonResult:
    method: str
    statistic: float
    p_value: float
    sample_size: int
    group_count: int
    mean_difference: Optional[float] = None
    difference_confidence_interval: Optional[Tuple[float, float]] = None


def analyze_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str,
    confidence_level: float = 0.95,
) -> CorrelationResult:
    if method not in ("Pearson", "Spearman"):
        raise ValueError("Choose Pearson or Spearman correlation.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    pairs = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 3:
        raise ValueError("At least three paired numeric observations are required.")
    if pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        raise ValueError("Both variables must vary to calculate a correlation.")

    if method == "Pearson":
        test = stats.pearsonr(pairs["x"], pairs["y"])
        interval = test.confidence_interval(confidence_level=confidence_level)
        confidence_interval = (float(interval.low), float(interval.high))
        coefficient = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        test = stats.spearmanr(pairs["x"], pairs["y"])
        confidence_interval = None
        coefficient = float(test.statistic)
        p_value = float(test.pvalue)

    if not np.isfinite(coefficient) or not np.isfinite(p_value):
        raise ValueError("The selected correlation could not be estimated from these values.")

    return CorrelationResult(
        method=method,
        coefficient=coefficient,
        p_value=p_value,
        sample_size=len(pairs),
        confidence_interval=confidence_interval,
    )


def compare_salary_groups(
    data: pd.DataFrame,
    group_column: str,
    value_column: str = "Salary",
    confidence_level: float = 0.95,
) -> Tuple[pd.DataFrame, GroupComparisonResult]:
    if group_column not in data.columns or value_column not in data.columns:
        raise ValueError("The selected grouping and salary columns must be present.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    values = pd.DataFrame({
        "group": data[group_column].astype("string"),
        "salary": pd.to_numeric(data[value_column], errors="coerce"),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    values["group"] = values["group"].astype(str)
    groups = [
        (label, frame["salary"].to_numpy(dtype=float))
        for label, frame in values.groupby("group", sort=True, observed=True)
    ]

    if len(groups) < 2:
        raise ValueError("At least two groups with valid salary data are required.")
    too_small = [label for label, sample in groups if len(sample) < 2]
    if too_small:
        raise ValueError(
            "Each group needs at least two salary observations; insufficient groups: "
            + ", ".join(too_small)
            + "."
        )
    if values["salary"].nunique() < 2:
        raise ValueError("Salary must vary to run a group comparison.")

    alpha = 1 - confidence_level
    summary_rows = []
    samples = []
    for label, sample in groups:
        count = len(sample)
        mean = float(np.mean(sample))
        standard_deviation = float(np.std(sample, ddof=1))
        margin = float(
            stats.t.ppf(1 - alpha / 2, count - 1)
            * standard_deviation
            / np.sqrt(count)
        )
        summary_rows.append({
            "Group": label,
            "Records": count,
            "Mean salary (₹)": mean,
            "Median salary (₹)": float(np.median(sample)),
            "SD (₹)": standard_deviation,
            f"{confidence_level:.0%} CI lower (₹)": mean - margin,
            f"{confidence_level:.0%} CI upper (₹)": mean + margin,
        })
        samples.append(sample)

    mean_difference = None
    difference_interval = None
    if len(samples) == 2:
        first, second = samples
        test = stats.ttest_ind(first, second, equal_var=False)
        first_term = np.var(first, ddof=1) / len(first)
        second_term = np.var(second, ddof=1) / len(second)
        standard_error_squared = first_term + second_term
        degrees_denominator = (
            first_term**2 / (len(first) - 1)
            + second_term**2 / (len(second) - 1)
        )
        if standard_error_squared <= 0 or degrees_denominator <= 0:
            raise ValueError("Welch's test is undefined because there is no within-group variation.")
        degrees_of_freedom = standard_error_squared**2 / degrees_denominator
        mean_difference = float(np.mean(first) - np.mean(second))
        margin = float(
            stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
            * np.sqrt(standard_error_squared)
        )
        difference_interval = (
            mean_difference - margin,
            mean_difference + margin,
        )
        method = "Welch's two-sample t-test"
    else:
        test = stats.kruskal(*samples)
        method = "Kruskal-Wallis test"

    statistic = float(test.statistic)
    p_value = float(test.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("The group comparison could not be estimated from these values.")

    summary = pd.DataFrame(summary_rows)
    result = GroupComparisonResult(
        method=method,
        statistic=statistic,
        p_value=p_value,
        sample_size=len(values),
        group_count=len(groups),
        mean_difference=mean_difference,
        difference_confidence_interval=difference_interval,
    )
    return summary, result
