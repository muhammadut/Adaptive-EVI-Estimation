
# Adaptive EVI Estimation with Probability Weighted Moments (PWM)

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




### Step 1: Initial Data Collection
- Start with a set of data points that represent extreme events you're interested in, such as very high insurance claims or extreme weather measurements.

### Step 2: Estimate Key Patterns
- Look at your data and calculate some numbers that help you understand how these extreme events are distributed. These numbers give you a rough idea of the trends in your data.

### Step 3: Choose the Best Approach
- Compare the numbers you calculated to see which method gives a more consistent picture of your data. Based on this comparison, choose the best method for further analysis.

### Step 4: Compute Initial Estimates
- Using the chosen method, make initial guesses for the key parameter (the EVI) that describes how extreme the events are. This is like taking a first stab at estimating the EVI.

### Step 5: Divide Your Data into Smaller Chunks
- Split your data into smaller, more manageable parts. This helps ensure that your analysis is not overly influenced by any one part of the data.

### Step 6: Create New Samples
- Generate multiple new sets of data by randomly picking points from your original data. This process, called bootstrapping, helps you understand how stable your estimates are.

### Step 7: Evaluate Performance
- For each new set of data, recalculate your EVI estimates. Check how well these estimates match up with what you would expect. This involves computing a measure (MSE) that tells you how far off your estimates are from the actual values.

### Step 8: Find the Best Split Point
- Determine the point where your estimates are the most accurate. This is done by finding the point where the MSE is minimized for your bootstrapped samples.

### Step 9: Determine Optimal Threshold
- Use the best split point you found in the previous step to calculate a threshold. This threshold helps you decide how to divide your data for the most accurate estimation.

### Step 10: Final Estimate
- Combine all the information gathered from the previous steps to make a final, robust estimate of the EVI. This final step ensures that your estimate is as accurate and reliable as possible.

### Summary of the Algorithm:
1. **Collect Data:** Gather your extreme events data.
2. **Estimate Patterns:** Calculate key numbers to understand data trends.
3. **Choose Approach:** Select the best method based on these numbers.
4. **Initial Estimates:** Make first guesses for the EVI.
5. **Divide Data:** Split data into smaller chunks.
6. **Create Samples:** Generate new sets of data for stability checks.
7. **Evaluate:** Check the accuracy of your estimates.
8. **Find Split Point:** Determine where your estimates are most accurate.
9. **Set Threshold:** Calculate the best threshold for data division.
10. **Final Estimate:** Combine everything to get the best EVI estimate.
