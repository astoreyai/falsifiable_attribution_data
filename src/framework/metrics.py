"""Statistical metrics for falsification experiments.

This module implements statistical analysis tools for evaluating
attribution methods using the falsification framework:
- Separation margin (d-prime statistic)
- Effect size (Cohen's d)
- Statistical significance testing (chi-square, t-test)
- Confidence intervals
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def compute_separation_margin(
    d_high: np.ndarray,
    d_low: np.ndarray,
    use_dprime: bool = True
) -> float:
    """
    Compute separation margin between high and low attribution distances.

    Two metrics available:
    1. d-prime (signal detection theory):
       d' = (μ_high - μ_low) / sqrt((σ²_high + σ²_low) / 2)

    2. Simple difference:
       margin = μ_high - μ_low

    The d-prime metric is preferred as it accounts for variance,
    providing a normalized measure of separation.

    Parameters
    ----------
    d_high : np.ndarray
        Geodesic distances for high-attribution perturbations
    d_low : np.ndarray
        Geodesic distances for low-attribution perturbations
    use_dprime : bool, optional
        If True, use d-prime metric (default)
        If False, use simple difference

    Returns
    -------
    separation : float
        Separation margin
        - Positive = high-attr produces larger changes (valid attribution)
        - Zero = no separation
        - Negative = low-attr produces larger changes (falsified)

    Examples
    --------
    >>> # Valid attribution: high regions produce larger changes
    >>> d_high = np.array([0.8, 0.9, 0.85, 0.95])
    >>> d_low = np.array([0.3, 0.2, 0.25, 0.35])
    >>> compute_separation_margin(d_high, d_low)
    4.123  # Large positive d-prime

    >>> # Falsified attribution: no separation
    >>> d_high = np.array([0.5, 0.6, 0.55])
    >>> d_low = np.array([0.5, 0.6, 0.55])
    >>> compute_separation_margin(d_high, d_low)
    0.0  # No separation
    """
    if len(d_high) == 0 or len(d_low) == 0:
        raise ValueError("Input arrays cannot be empty")

    mu_high = np.mean(d_high)
    mu_low = np.mean(d_low)

    if use_dprime:
        # d-prime statistic
        var_high = np.var(d_high, ddof=1) if len(d_high) > 1 else 0.0
        var_low = np.var(d_low, ddof=1) if len(d_low) > 1 else 0.0

        # Pooled standard deviation
        pooled_std = np.sqrt((var_high + var_low) / 2.0)

        if pooled_std == 0:
            # No variance - use simple difference
            return float(mu_high - mu_low)

        dprime = (mu_high - mu_low) / pooled_std
        return float(dprime)
    else:
        # Simple difference
        return float(mu_high - mu_low)


def compute_effect_size(
    fr1: float,
    fr2: float,
    n1: int,
    n2: int,
    method: str = 'cohens_h'
) -> float:
    """
    Compute effect size for comparing two falsification rates.

    Cohen's h for proportions:
        h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

    where p1 and p2 are proportions (FR/100).

    Interpretation:
    - |h| < 0.2: small effect
    - |h| ≈ 0.5: medium effect
    - |h| > 0.8: large effect

    Parameters
    ----------
    fr1 : float
        Falsification rate for method 1 (percentage, 0-100)
    fr2 : float
        Falsification rate for method 2 (percentage, 0-100)
    n1 : int
        Sample size for method 1
    n2 : int
        Sample size for method 2
    method : str, optional
        Effect size method (default 'cohens_h')
        Options: 'cohens_h', 'cohens_d'

    Returns
    -------
    effect_size : float
        Effect size measure
        Positive = method 1 has higher FR (worse)
        Negative = method 2 has higher FR (method 1 better)

    Raises
    ------
    ValueError
        If FR values not in [0, 100]
        If sample sizes <= 0

    Examples
    --------
    >>> # Method 1 (Grad-CAM): FR = 45%
    >>> # Method 2 (Random): FR = 60%
    >>> effect_size = compute_effect_size(45, 60, n1=100, n2=100)
    >>> effect_size
    -0.30  # Medium effect, Method 1 better
    """
    if not (0 <= fr1 <= 100) or not (0 <= fr2 <= 100):
        raise ValueError(f"FR values must be in [0, 100], got {fr1}, {fr2}")

    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive, got {n1}, {n2}")

    # Convert percentages to proportions
    p1 = fr1 / 100.0
    p2 = fr2 / 100.0

    if method == 'cohens_h':
        # Cohen's h for proportions
        phi1 = 2 * np.arcsin(np.sqrt(p1))
        phi2 = 2 * np.arcsin(np.sqrt(p2))
        h = phi1 - phi2
        return float(h)

    elif method == 'cohens_d':
        # Cohen's d (approximation for proportions)
        # Use binomial variance: p(1-p)
        var1 = p1 * (1 - p1)
        var2 = p2 * (1 - p2)

        # Pooled standard deviation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)

        if pooled_std == 0:
            return 0.0

        d = (p1 - p2) / pooled_std
        return float(d)

    else:
        raise ValueError(f"Unknown method: {method}")


def statistical_significance_test(
    fr1: float,
    fr2: float,
    n1: int,
    n2: int,
    test: str = 'chi_square',
    alpha: float = 0.05
) -> Dict:
    """
    Test statistical significance of difference between two falsification rates.

    Available tests:
    1. Chi-square test (default): Tests independence of FR and method
    2. Two-proportion z-test: Tests difference in proportions
    3. Fisher's exact test: Exact test for small samples

    Parameters
    ----------
    fr1 : float
        Falsification rate for method 1 (percentage)
    fr2 : float
        Falsification rate for method 2 (percentage)
    n1 : int
        Sample size for method 1
    n2 : int
        Sample size for method 2
    test : str, optional
        Test to use: 'chi_square', 'z_test', 'fisher' (default 'chi_square')
    alpha : float, optional
        Significance level (default 0.05)

    Returns
    -------
    result : dict
        {
            'statistic': float,      # Test statistic
            'p_value': float,        # p-value
            'is_significant': bool,  # p < alpha
            'test_name': str,        # Name of test used
            'effect_size': float     # Cohen's h effect size
        }

    Examples
    --------
    >>> # Test if Grad-CAM (FR=45%) is significantly better than Random (FR=60%)
    >>> result = statistical_significance_test(45, 60, n1=100, n2=100)
    >>> result['is_significant']
    True
    >>> result['p_value']
    0.023  # p < 0.05, significant difference
    """
    # Convert to proportions and counts
    p1 = fr1 / 100.0
    p2 = fr2 / 100.0
    count1 = int(p1 * n1)
    count2 = int(p2 * n2)

    if test == 'chi_square':
        # Contingency table:
        #              Falsified   Not Falsified
        # Method 1:    count1      n1-count1
        # Method 2:    count2      n2-count2
        contingency = np.array([
            [count1, n1 - count1],
            [count2, n2 - count2]
        ])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        statistic = chi2
        test_name = 'Chi-square test'

    elif test == 'z_test':
        # Two-proportion z-test
        # Pooled proportion
        p_pool = (count1 + count2) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se == 0:
            # No variance
            statistic = 0.0
            p_value = 1.0
        else:
            # Z statistic
            z = (p1 - p2) / se
            statistic = z

            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        test_name = 'Two-proportion z-test'

    elif test == 'fisher':
        # Fisher's exact test
        contingency = np.array([
            [count1, n1 - count1],
            [count2, n2 - count2]
        ])

        oddsratio, p_value = stats.fisher_exact(contingency)
        statistic = oddsratio
        test_name = "Fisher's exact test"

    else:
        raise ValueError(f"Unknown test: {test}")

    # Compute effect size
    effect_size = compute_effect_size(fr1, fr2, n1, n2)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': bool(p_value < alpha),
        'test_name': test_name,
        'effect_size': float(effect_size),
        'alpha': alpha
    }


def compute_confidence_interval(
    fr: float,
    n: int,
    confidence: float = 0.95,
    method: str = 'wilson'
) -> Tuple[float, float]:
    """
    Compute confidence interval for falsification rate.

    Available methods:
    1. Wilson score interval (default): Better for extreme proportions
    2. Normal approximation: Simple but less accurate at extremes
    3. Clopper-Pearson: Exact binomial CI (conservative)

    Parameters
    ----------
    fr : float
        Falsification rate (percentage, 0-100)
    n : int
        Sample size
    confidence : float, optional
        Confidence level (default 0.95 = 95% CI)
    method : str, optional
        Method: 'wilson', 'normal', 'clopper_pearson' (default 'wilson')

    Returns
    -------
    ci_lower, ci_upper : tuple of float
        Lower and upper bounds of confidence interval (as percentages)

    Examples
    --------
    >>> # Grad-CAM: FR = 45% on 100 samples
    >>> ci = compute_confidence_interval(45, 100)
    >>> ci
    (35.2, 55.1)  # 95% CI: [35.2%, 55.1%]
    """
    if not (0 <= fr <= 100):
        raise ValueError(f"FR must be in [0, 100], got {fr}")

    if n <= 0:
        raise ValueError(f"Sample size must be positive, got {n}")

    # Convert to proportion
    p = fr / 100.0

    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    if method == 'wilson':
        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denominator

        lower = center - margin
        upper = center + margin

    elif method == 'normal':
        # Normal approximation
        se = np.sqrt(p * (1-p) / n)
        margin = z * se

        lower = p - margin
        upper = p + margin

    elif method == 'clopper_pearson':
        # Exact binomial CI
        k = int(p * n)

        if k == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf((1 - confidence) / 2, k, n - k + 1)

        if k == n:
            upper = 1.0
        else:
            upper = stats.beta.ppf((1 + confidence) / 2, k + 1, n - k)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Clip to [0, 1] and convert back to percentage
    lower = max(0, min(1, lower)) * 100
    upper = max(0, min(1, upper)) * 100

    return float(lower), float(upper)


def summarize_comparison(
    method1_name: str,
    method2_name: str,
    fr1: float,
    fr2: float,
    n1: int,
    n2: int,
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive statistical comparison of two attribution methods.

    Parameters
    ----------
    method1_name : str
        Name of first method (e.g., "Grad-CAM")
    method2_name : str
        Name of second method (e.g., "Random")
    fr1 : float
        Falsification rate for method 1 (percentage)
    fr2 : float
        Falsification rate for method 2 (percentage)
    n1 : int
        Sample size for method 1
    n2 : int
        Sample size for method 2
    alpha : float, optional
        Significance level (default 0.05)

    Returns
    -------
    summary : dict
        Complete statistical summary including:
        - Falsification rates and CIs
        - Effect size
        - Statistical test results
        - Interpretation

    Examples
    --------
    >>> summary = summarize_comparison(
    ...     "Grad-CAM", "Random",
    ...     fr1=45, fr2=60, n1=100, n2=100
    ... )
    >>> print(summary['interpretation'])
    "Grad-CAM significantly better than Random (p=0.023, h=-0.30)"
    """
    # Confidence intervals
    ci1 = compute_confidence_interval(fr1, n1)
    ci2 = compute_confidence_interval(fr2, n2)

    # Statistical test
    test_result = statistical_significance_test(fr1, fr2, n1, n2, alpha=alpha)

    # Effect size
    effect_size = test_result['effect_size']

    # Interpretation
    if test_result['is_significant']:
        if fr1 < fr2:
            winner = method1_name
            interpretation = f"{method1_name} significantly better than {method2_name}"
        else:
            winner = method2_name
            interpretation = f"{method2_name} significantly better than {method1_name}"

        interpretation += f" (p={test_result['p_value']:.3f}, h={effect_size:.2f})"
    else:
        winner = "none"
        interpretation = f"No significant difference between {method1_name} and {method2_name}"
        interpretation += f" (p={test_result['p_value']:.3f})"

    # Effect size interpretation
    abs_h = abs(effect_size)
    if abs_h < 0.2:
        effect_interp = "small"
    elif abs_h < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    return {
        'method1': {
            'name': method1_name,
            'fr': fr1,
            'n': n1,
            'ci': ci1,
            'ci_str': f"[{ci1[0]:.1f}%, {ci1[1]:.1f}%]"
        },
        'method2': {
            'name': method2_name,
            'fr': fr2,
            'n': n2,
            'ci': ci2,
            'ci_str': f"[{ci2[0]:.1f}%, {ci2[1]:.1f}%]"
        },
        'statistical_test': test_result,
        'effect_size': effect_size,
        'effect_interpretation': effect_interp,
        'winner': winner,
        'interpretation': interpretation
    }


def format_result_table(
    results: Dict[str, Dict],
    n_samples: int = 200
) -> str:
    """
    Format results as LaTeX table.

    Args:
        results: Dictionary mapping method names to results
        n_samples: Number of samples used

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Falsification Rate Comparison (Experiment 6.1)}")
    lines.append("\\label{tab:exp_6_1_results}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & FR (\\%) & $d'$ & $p$-value & Significant \\\\ ")
    lines.append("\\midrule")
    
    for method, res in results.items():
        fr = res.get('falsification_rate', 0.0)
        sep = res.get('separation_margin', 0.0)
        p_val = res.get('p_value', 1.0)
        sig = "Yes" if p_val < 0.05 else "No"
        
        lines.append(f"{method} & {fr:.1f} & {sep:.2f} & {p_val:.3f} & {sig} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\\\[0.5em] {{\\footnotesize $n={n_samples}$ pairs, $\\alpha=0.05$}}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)
