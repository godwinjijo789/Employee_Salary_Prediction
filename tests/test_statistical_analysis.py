import unittest

import pandas as pd

from statistical_analysis import analyze_correlation, compare_salary_groups


class CorrelationAnalysisTests(unittest.TestCase):
    def test_pearson_reports_correlation_and_confidence_interval(self):
        result = analyze_correlation(
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series([2, 4, 6, 8, 10]),
            "Pearson",
        )

        self.assertAlmostEqual(result.coefficient, 1.0)
        self.assertEqual(result.sample_size, 5)
        self.assertEqual(result.p_value, 0.0)
        self.assertIsNotNone(result.confidence_interval)

    def test_spearman_detects_monotonic_association(self):
        result = analyze_correlation(
            pd.Series([1, 2, 3, 4]),
            pd.Series([1, 3, 2, 4]),
            "Spearman",
        )

        self.assertGreater(result.coefficient, 0)
        self.assertIsNone(result.confidence_interval)

    def test_correlation_rejects_constant_variables(self):
        with self.assertRaisesRegex(ValueError, "must vary"):
            analyze_correlation(
                pd.Series([1, 1, 1]),
                pd.Series([2, 3, 4]),
                "Pearson",
            )


class SalaryGroupComparisonTests(unittest.TestCase):
    def test_two_groups_use_welch_test_and_difference_interval(self):
        data = pd.DataFrame({
            "Gender": ["A"] * 5 + ["B"] * 5,
            "Salary": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
        })

        summary, result = compare_salary_groups(data, "Gender")

        self.assertEqual(result.method, "Welch's two-sample t-test")
        self.assertEqual(result.sample_size, 10)
        self.assertLess(result.p_value, 0.05)
        self.assertAlmostEqual(result.mean_difference, -10.0)
        self.assertEqual(len(summary), 2)
        self.assertIsNotNone(result.difference_confidence_interval)

    def test_three_groups_use_kruskal_wallis(self):
        data = pd.DataFrame({
            "Department": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "Salary": [1, 2, 3, 10, 11, 12, 20, 21, 22],
        })

        _, result = compare_salary_groups(data, "Department")

        self.assertEqual(result.method, "Kruskal-Wallis test")
        self.assertEqual(result.group_count, 3)
        self.assertLess(result.p_value, 0.05)
        self.assertIsNone(result.mean_difference)

    def test_group_test_requires_two_groups(self):
        data = pd.DataFrame({
            "Department": ["A", "A", "A"],
            "Salary": [10, 20, 30],
        })

        with self.assertRaisesRegex(ValueError, "At least two groups"):
            compare_salary_groups(data, "Department")


if __name__ == "__main__":
    unittest.main()
