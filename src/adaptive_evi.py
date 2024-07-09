import numpy as np
from scipy.special import gamma
from tqdm import tqdm
from typing import List, Dict, Union, Tuple

class AdaptiveEVI:
    """
    Adaptive Extreme Value Index (EVI) estimator.

    This class implements an adaptive method for estimating the Extreme Value Index
    using various techniques including bootstrap sampling, bias correction, and
    multiple estimators.

    Attributes:
        data (np.ndarray): Sorted input data for EVI estimation.
        N (int): Number of data points.
        subsample_size (float): Proportion of data to use in subsampling.
        bootstrap_samples (int): Number of bootstrap samples to generate.
        estimator (str): Type of estimator to use ('hill' or 'pwm').
        n1 (int): First subsample size.
        n2 (int): Second subsample size.
        tau_star (int): Optimal tau value.
        k_0_star (int): Optimal k value.
        rho_tau_star_k1 (float): Estimated rho parameter.
        beta_tau_star_k1 (float): Estimated beta parameter.
        initial_evi_at_k0 (float): Initial EVI estimate at k_0_star.
        bias_corrected_evi_value (float): Final bias-corrected EVI estimate.
    """

    def __init__(self, data: np.ndarray, subsample_size: float = 0.8,
                 bootstrap_samples: int = 1000, estimator: str = "hill"):
        """
        Initialize the AdaptiveEVI estimator.

        Args:
            data (np.ndarray): Input data for EVI estimation.
            subsample_size (float): Proportion of data to use in subsampling.
            bootstrap_samples (int): Number of bootstrap samples to generate.
            estimator (str): Type of estimator to use ('hill' or 'pwm').
        """
        self.data = np.sort(data)
        self.N = len(data)
        self.subsample_size = subsample_size
        self.bootstrap_samples = bootstrap_samples
        self.estimator = estimator.lower()
        
        if self.estimator not in ['hill', 'pwm']:
            raise ValueError("Estimator must be either 'hill' or 'pwm'.")
        
        self.n1, self.n2 = self._select_subsample_sizes()
        self.tau_star = None
        self.k_0_star = None
        self.rho_tau_star_k1 = None
        self.beta_tau_star_k1 = None
        self.initial_evi_at_k0 = None
        self.bias_corrected_evi_value = None

    def _rho_estimator(self, data: np.ndarray, alpha: float = 1, theta1: float = 2,
                       theta2: float = 3, tau: float = 1) -> Dict[str, np.ndarray]:
        """
        Estimate the second order parameter rho for extreme value theory.

        Args:
            data (np.ndarray): The sample data.
            alpha (float): Positive parameter for moments.
            theta1 (float): First positive parameter for moments.
            theta2 (float): Second positive parameter for moments.
            tau (float): Non-negative tuning parameter.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing 'k', 'rho', and 'Tn' arrays.

        Raises:
            ValueError: If alpha is not positive or if tau is negative.
        """
        # Input validation
        if alpha <= 0:
            raise ValueError("alpha should be strictly positive.")
        if tau < 0:
            raise ValueError("tau should be non-negative.")

        X = np.sort(data)
        n = len(X)
        K = np.arange(1, n)

        # Initialize arrays
        M_alpha = np.zeros(n)
        M_alpha_theta1 = np.zeros(n)
        M_alpha_theta2 = np.zeros(n)
        Tn = np.zeros(n)
        rho = np.zeros(n)

        l = np.log(X[n - K])

        # Compute moments
        for k in K:
            if k == 0:
                continue
            M_alpha[k] = np.sum((l[:k] - np.log(X[n - k]))**alpha) / k
            M_alpha_theta1[k] = np.sum((l[:k] - np.log(X[n - k]))**(alpha * theta1)) / k
            M_alpha_theta2[k] = np.sum((l[:k] - np.log(X[n - k]))**(alpha * theta2)) / k

        # Compute Tn based on tau
        if tau == 0:
            Tn[K] = (np.log(M_alpha[K]) - np.log(gamma(alpha + 1)) -
                     (np.log(M_alpha_theta1[K]) - np.log(gamma(alpha * theta1 + 1))) / theta1) / \
                    ((np.log(M_alpha_theta1[K]) - np.log(gamma(alpha * theta1 + 1))) / theta1 -
                     (np.log(M_alpha_theta2[K]) - np.log(gamma(alpha * theta2 + 1))) / theta2)
        else:
            Tn[K] = ((M_alpha[K] / gamma(alpha + 1))**tau - (M_alpha_theta1[K] / gamma(alpha * theta1 + 1))**(tau / theta1)) / \
                    ((M_alpha_theta1[K] / gamma(alpha * theta1 + 1))**(tau / theta1) - (M_alpha_theta2[K] / gamma(alpha * theta2 + 1))**(tau / theta2))

        # Compute rho
        for k in K:
            if Tn[k] == 0 or Tn[k] == 3:
                rho[k] = np.nan
            else:
                rho[k] = 1 - (2 * Tn[k] / (3 - Tn[k]))**(1 / alpha)

        return {'k': K, 'rho': rho[K], 'Tn': Tn[K]}

    def _get_rho_medians(self) -> Tuple[float, float, range, List[float], List[float]]:
        """
        Calculate median rho values for tau=0 and tau=1.

        Returns:
            Tuple containing eta_0, eta_1, K range, rho_0_selected, and rho_1_selected.
        """
        k_min = int(self.N**0.995)
        k_max = int(self.N**0.999)
        K = range(k_min, k_max + 1)

        rho_0 = self._rho_estimator(self.data, tau=0)
        rho_1 = self._rho_estimator(self.data, tau=1)

        rho_0_selected = [rho_0['rho'][k-1] for k in K if k-1 < len(rho_0['rho'])]
        rho_1_selected = [rho_1['rho'][k-1] for k in K if k-1 < len(rho_1['rho'])]

        eta_0 = np.median(rho_0_selected)
        eta_1 = np.median(rho_1_selected)

        return eta_0, eta_1, K, rho_0_selected, rho_1_selected

    def _choose_best_tau(self, rho_0_selected: List[float], rho_1_selected: List[float],
                         eta_0: float, eta_1: float) -> Tuple[int, float, float]:
        """
        Choose the best tau value based on sum of squared deviations.

        Args:
            rho_0_selected (List[float]): Selected rho values for tau=0.
            rho_1_selected (List[float]): Selected rho values for tau=1.
            eta_0 (float): Median rho value for tau=0.
            eta_1 (float): Median rho value for tau=1.

        Returns:
            Tuple containing tau_star, I_0, and I_1.
        """
        I_0 = np.sum((np.array(rho_0_selected) - eta_0)**2)
        I_1 = np.sum((np.array(rho_1_selected) - eta_1)**2)
        tau_star = 0 if I_0 <= I_1 else 1
        return tau_star, I_0, I_1

    def generate_tau_star(self) -> int:
        """
        Generate the optimal tau value.

        Returns:
            int: The optimal tau value (0 or 1).
        """
        eta_0, eta_1, _, rho_0_selected, rho_1_selected = self._get_rho_medians()
        self.tau_star, _, _ = self._choose_best_tau(rho_0_selected, rho_1_selected, eta_0, eta_1)
        return self.tau_star

    def _rho_est_k1(self, data: np.ndarray, tau: int, k1: int) -> float:
        """
        Estimate rho for a specific k1 value.

        Args:
            data (np.ndarray): Input data.
            tau (int): Tau value (0 or 1).
            k1 (int): k1 value.

        Returns:
            float: Estimated rho value.
        """
        X = np.sort(data)
        n = len(X)
        alpha, theta1, theta2 = 1, 2, 3

        l = np.log(X[n - k1:])
        M_alpha = np.sum((l - np.log(X[n - k1]))**alpha) / k1
        M_alpha_theta1 = np.sum((l - np.log(X[n - k1]))**(alpha * theta1)) / k1
        M_alpha_theta2 = np.sum((l - np.log(X[n - k1]))**(alpha * theta2)) / k1

        if tau == 0:
            Tn = (np.log(M_alpha) - np.log(gamma(alpha + 1)) -
                  (np.log(M_alpha_theta1) - np.log(gamma(alpha * theta1 + 1))) / theta1) / \
                 ((np.log(M_alpha_theta1) - np.log(gamma(alpha * theta1 + 1))) / theta1 -
                  (np.log(M_alpha_theta2) - np.log(gamma(alpha * theta2 + 1))) / theta2)
        else:
            Tn = ((M_alpha / gamma(alpha + 1))**tau - (M_alpha_theta1 / gamma(alpha * theta1 + 1))**(tau / theta1)) / \
                 ((M_alpha_theta1 / gamma(alpha * theta1 + 1))**(tau / theta1) - (M_alpha_theta2 / gamma(alpha * theta2 + 1))**(tau / theta2))

        if Tn == 0 or Tn == 3:
            return np.nan
        else:
            return 1 - (2 * Tn / (3 - Tn))**(1 / alpha)

    def _estimate_beta(self, data: np.ndarray, k: int, rho: float) -> float:
        """
        Estimate the beta parameter.

        Args:
            data (np.ndarray): Input data.
            k (int): k value.
            rho (float): Estimated rho value.

        Returns:
            float: Estimated beta value.
        """
        def d_k(alpha):
            return np.mean([(i/k)**(-alpha) for i in range(1, k+1)])

        def D_k(alpha):
            sorted_data = np.sort(data)
            U_i = [i * (np.log(sorted_data[-i]) - np.log(sorted_data[-(i+1)])) for i in range(1, k+1)]
            return np.mean([(i/k)**(-alpha) * U_i[i-1] for i in range(1, k+1)])

        numerator = d_k(rho) * D_k(0) - D_k(rho)
        denominator = d_k(rho) * D_k(rho) - D_k(2 * rho)

        beta_est = (k/self.N)**rho * numerator / denominator
        return beta_est

    def generate_bias_correction_coefs(self) -> Dict[str, float]:
        """
        Generate bias correction coefficients.

        Returns:
            Dict[str, float]: Dictionary containing rho_tau_star_k1 and beta_tau_star_k1.
        """
        k1 = int(self.N ** 0.999)
        self.rho_tau_star_k1 = self._rho_est_k1(self.data, self.generate_tau_star(), k1)
        self.beta_tau_star_k1 = self._estimate_beta(self.data, k1, self.rho_tau_star_k1)

        return {"rho_tau_star_k1": self.rho_tau_star_k1, "beta_tau_star_k1": self.beta_tau_star_k1}

    def _select_subsample_sizes(self) -> Tuple[int, int]:
        """
        Select subsample sizes n1 and n2.

        Returns:
            Tuple[int, int]: n1 and n2 values.
        """
        n1 = int(self.N ** self.subsample_size)
        n2 = (n1 ** 2) // self.N + 1
        return n1, n2

    def generate_bootstrap_samples(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate bootstrap samples.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Bootstrap samples for n1 and n2.
        """
        bootstrap_samples_n1 = [np.random.choice(self.data, size=self.n1, replace=True) for _ in range(self.bootstrap_samples)]
        bootstrap_samples_n2 = [np.random.choice(self.data, size=self.n2, replace=True) for _ in range(self.bootstrap_samples)]
        return bootstrap_samples_n1, bootstrap_samples_n2

    def _compute_T_statistic(self, data: np.ndarray, k: int, estimator_func: callable) -> float:
        """
        Compute the T-statistic for a given data sample and k value.

        Args:
            data (np.ndarray): Input data.
            k (int): k value.
            estimator_func (callable): Estimator function (hill or pwm).

        Returns:
            float: Computed T-statistic.
        """
        sorted_data = np.sort(data)
        return estimator_func(sorted_data, k // 2) - estimator_func(sorted_data, k)

    def generate_bootstrap_T_statistics(self) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Generate bootstrap T-statistics.

        Returns:
            Tuple[List[List[float]], List[List[float]]]: T-statistics for n1 and n2 samples.
        """
        bootstrap_T_statistics_n1 = []
        bootstrap_T_statistics_n2 = []
        bootstrap_samples_n1, bootstrap_samples_n2 = self.generate_bootstrap_samples()
        estimator_func = self._hill_estimator if self.estimator == 'hill' else self._pwm_estimator

        for sample in tqdm(bootstrap_samples_n1, desc="Processing n1 samples"):
            T_stats = [self._compute_T_statistic(sample, k, estimator_func) for k in range(2, self.n1) if not np.isnan(self._compute_T_statistic(sample, k, estimator_func))]
            bootstrap_T_statistics_n1.append(T_stats)

        for sample in tqdm(bootstrap_samples_n2, desc="Processing n2 samples"):
            T_stats = [self._compute_T_statistic(sample, k, estimator_func) for k in range(2, self.n2) if not np.isnan(self._compute_T_statistic(sample, k, estimator_func))]
            bootstrap_T_statistics_n2.append(T_stats)

        return bootstrap_T_statistics_n1, bootstrap_T_statistics_n2

    def _minimize_mse(self, bootstrap_T_statistics_n1: List[List[float]],
                      bootstrap_T_statistics_n2: List[List[float]]) -> Tuple[int, int]:
        """
        Minimize the Mean Squared Error (MSE) to find the optimal k value.

        Args:
            bootstrap_T_statistics_n1 (List[List[float]]): T-statistics for n1 samples.
            bootstrap_T_statistics_n2 (List[List[float]]): T-statistics for n2 samples.

        Returns:
            Tuple[int, int]: Optimal k values for n1 and n2.
        """
        mse_values_n1 = [np.nanmean([t[k]**2 for t in bootstrap_T_statistics_n1 if k < len(t)]) for k in range(len(bootstrap_T_statistics_n1[0]))]
        mse_values_n2 = [np.nanmean([t[k]**2 for t in bootstrap_T_statistics_n2 if k < len(t)]) for k in range(len(bootstrap_T_statistics_n2[0]))]

        k_0_star_n1 = np.argmin(mse_values_n1) + 2
        k_0_star_n2 = np.argmin(mse_values_n2) + 2

        return k_0_star_n1, k_0_star_n2

    def _compute_threshold_estimate(self, k_0_star_n1: int, k_0_star_n2: int) -> int:
        """
        Compute the threshold estimate k_0_star.

        Args:
            k_0_star_n1 (int): Optimal k value for n1.
            k_0_star_n2 (int): Optimal k value for n2.

        Returns:
            int: Computed threshold estimate k_0_star.
        """
        n = self.N
        rho_hat = self.rho_tau_star_k1

        inner_term = ((1 - 2**rho_hat)**(1/(2-2*rho_hat)) * (k_0_star_n1**2)) / k_0_star_n2
        k_0_star = min(n - 1, int(np.floor(inner_term) + 1))

        return k_0_star

    def generate_k_0_star(self) -> int:
        """
        Generate the optimal k_0_star value.

        Returns:
            int: The optimal k_0_star value.
        """
        bootstrap_T_statistics_n1, bootstrap_T_statistics_n2 = self.generate_bootstrap_T_statistics()
        k_0_star_n1, k_0_star_n2 = self._minimize_mse(bootstrap_T_statistics_n1, bootstrap_T_statistics_n2)
        self.k_0_star = self._compute_threshold_estimate(k_0_star_n1, k_0_star_n2)
        return self.k_0_star

    def generate_evi(self) -> float:
        """
        Generate the final Extreme Value Index (EVI) estimate.

        This method implements the main algorithm for EVI estimation, including
        bias correction, bootstrap sampling, and optimal k selection.

        Returns:
            float: The final bias-corrected EVI estimate.
        """
        # Generate bias correction coefficients
        self.generate_bias_correction_coefs()

        # Step 1: Subsample Size
        self.n1, self.n2 = self._select_subsample_sizes()

        # Step 2: Bootstrap Sampling
        bootstrap_samples_n1, bootstrap_samples_n2 = self.generate_bootstrap_samples()

        # Step 3: Compute Bootstrap T-statistics
        bootstrap_T_statistics_n1, bootstrap_T_statistics_n2 = self.generate_bootstrap_T_statistics()

        # Step 4: Minimize MSE
        k_0_star_n1, k_0_star_n2 = self._minimize_mse(bootstrap_T_statistics_n1, bootstrap_T_statistics_n2)

        # Step 5: Compute Threshold Estimate
        self.k_0_star = self._compute_threshold_estimate(k_0_star_n1, k_0_star_n2)

        # Step 6: Bias-Corrected EVI
        initial_evi = self._hill_estimator(self.data, self.k_0_star) if self.estimator == 'hill' else self._pwm_estimator(self.data, self.k_0_star)
        self.bias_corrected_evi_value = self._bias_corrected_evi(initial_evi, self.rho_tau_star_k1, self.beta_tau_star_k1, self.N, self.k_0_star)

        return self.bias_corrected_evi_value

    def _bias_corrected_evi(self, initial_evi: float, rho_hat: float, beta_hat: float, n: int, k: int) -> float:
        """
        Calculate the bias-corrected Extreme Value Index (EVI).

        Args:
            initial_evi (float): Initial EVI estimate.
            rho_hat (float): Estimated rho parameter.
            beta_hat (float): Estimated beta parameter.
            n (int): Total number of data points.
            k (int): Number of extreme values used.

        Returns:
            float: Bias-corrected EVI estimate.
        """
        correction_term = beta_hat * (n / k) ** rho_hat
        return initial_evi * (1 - correction_term)  

    @staticmethod
    def _pwm_estimator(data: np.ndarray, k: int) -> float:
        """
        Probability Weighted Moments (PWM) estimator for the Extreme Value Index.

        Args:
            data (np.ndarray): Input data.
            k (int): Number of extreme values to use.

        Returns:
            float: Estimated EVI using PWM method.
        """
        if k < 3:
            return np.nan
        sorted_data = np.sort(data)[-k:][::-1]
        a0 = np.nanmedian(sorted_data)
        a1 = np.nanmedian(np.arange(1, k + 1) / k**2 * sorted_data)  # Paper's formula with 1/k^2
        denominator = a0 - a1 + 1e-8
        if denominator == 0:
            return np.nan
        return 1 - (a1 / denominator)

    @staticmethod
    def _hill_estimator(data: np.ndarray, k: int) -> float:
        """
        Hill estimator for the Extreme Value Index.

        Args:
            data (np.ndarray): Input data.
            k (int): Number of extreme values to use.

        Returns:
            float: Estimated EVI using Hill estimator.
        """
        if k < 1:
            return np.nan
        sorted_data = np.sort(data)[-k:]
        return np.mean(np.log(sorted_data)) - np.log(sorted_data[0])