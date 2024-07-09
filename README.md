
# Adaptive EVI Estimation with Probability Weighted Moments (PWM)
# EXAMPLE USEAGE 

```python
# Create an instance of AdaptiveEVI
evi_estimator = AdaptiveEVI(data, subsample_size=0.8, bootstrap_samples=50, estimator='hill')

# Generate the final EVI estimate
evi_estimate = evi_estimator.generate_evi()

print(f"Estimated Extreme Value Index: {evi_estimate}")
print(evi_estimator.k_0_star)

```
## Introduction

This repository provides a Python implementation of an adaptive method for estimating the Extreme Value Index (EVI) using Probability Weighted Moments (PWM). The EVI is a crucial parameter in extreme value theory, playing a key role in understanding and predicting the behavior of extreme events such as natural disasters, financial market crashes, or rare insurance claims.  

## Background

Traditional methods for estimating the EVI, like the Hill estimator or the standard PWM estimator, often rely on a pre-determined threshold for selecting the number of extreme values to include in the estimation.  This choice of threshold can significantly impact the accuracy and reliability of the EVI estimate.

The adaptive EVI estimation method, as presented in the research paper "Computational Study of the Adaptive Estimation of the Extreme Value Index with Probability Weighted Moments," addresses this issue by introducing a data-driven approach for automatically selecting the optimal threshold.

## Summary

This implementation follows the methodology described in the research paper, employing a double bootstrap technique to assess the stability of EVI estimates across different thresholds. The algorithm then selects the threshold that yields the most stable and reliable estimate, thus mitigating the potential bias and inaccuracy associated with a fixed threshold.

**Key Features:**

* **Adaptive Threshold Selection:** Eliminates the need for manual and potentially subjective threshold selection.
* **Improved Accuracy:** Often leads to more accurate EVI estimates compared to traditional methods.
* **Increased Robustness:** The method is more robust to various tail behaviors and sample sizes, making it applicable to a wider range of scenarios.

**Potential Applications:**

The adaptive EVI estimation method is valuable for:

* **Risk Assessment:**  Quantifying the likelihood and potential impact of extreme events in insurance, finance, and other fields.
* **Engineering Design:** Determining safety margins and design criteria for infrastructure to withstand extreme conditions.
* **Climate Modeling:**  Analyzing extreme weather patterns and their potential consequences.


# The Algorithm

<img width="466" alt="image" src="https://github.com/muhammadut/Adaptive-EVI-Estimation/assets/36341682/0edd6d6e-093c-411d-9f88-b32b2f3f1066">




Certainly! Here's the revised version of the algorithm description using variables in backticks for better display on GitHub.

---

## Adaptive Extreme Value Index (EVI) Estimation Algorithm

This algorithm is designed to estimate the Extreme Value Index (EVI), a crucial parameter in extreme value theory used to model the tail behavior of distributions. The algorithm focuses on heavy-tailed distributions (those with a positive EVI) and employs a double bootstrap approach to adaptively select the optimal number of order statistics for EVI estimation. It supports both the Hill and the Probability Weighted Moments (PWM) estimators.

### Key Steps

1. **Bias Correction Coefficients Generation:**
    - **Rho Estimation (Fraga Alves et al., 2003):** Estimate the second-order parameter `rho` using the `rho`-estimators with tuning parameters `tau = 0` and `tau = 1`. Select the optimal `tau` based on minimizing the median squared error.
    - **Beta Estimation (Gomes and Martins, 2002):** Estimate the second-order parameter `beta` using the selected `tau` and the corresponding `rho` estimate.
    - **Calculate `rho` and `beta` at `k1`:** Compute `rho` and `beta` at `k1 = floor(n^0.999)`, where `n` is the sample size.

2. **Subsample Size Selection:**
    - Determine two subsample sizes, `n1` and `n2`, where `n1` is a fraction of the original sample size and `n2` is derived from `n1`.

3. **Bootstrap Sampling:**
    - Generate multiple bootstrap samples of sizes `n1` and `n2` from the empirical distribution of the original data.

4. **Compute Bootstrap T-statistics:**
    - For each bootstrap sample and a range of `k` values, calculate the T-statistic, which is the difference between the EVI estimates at `k/2` and `k`.

5. **Minimize Mean Squared Error (MSE):**
    - Find the optimal `k` values (`k0_star`) for both `n1` and `n2` that minimize the MSE of the T-statistics.

6. **Compute Threshold Estimate:**
    - Combine `k0_star` values from both subsamples and the estimated `rho` to determine the final threshold estimate `k0_star`.

7. **Bias-Corrected EVI:**
    - Calculate the initial EVI estimate using either the Hill or PWM estimator at `k0_star`.
    - Apply bias correction using the estimated `rho` and `beta` to obtain the final, bias-corrected EVI.

### Key References

- **Fraga Alves et al. (2003):** Introduces the `rho`-estimators for estimating the second-order parameter `rho`.
- **Gomes and Martins (2002):** Provides the method for estimating the second-order parameter `beta`.
- **Caeiro et al. (2014):** Details the semi-parametric probability-weighted moments estimation.
- **Gomes and Oliveira (2001):** Discusses the bootstrap methodology in the context of extreme value statistics.

---
