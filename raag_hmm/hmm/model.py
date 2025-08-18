"""
Discrete Hidden Markov Model implementation.

This module implements a discrete HMM with 36 states and 36 observations
for raag classification using chromatic pitch sequences.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DiscreteHMM:
    """
    Discrete Hidden Markov Model with 36 states and 36 observations.
    
    This implementation is designed for raag classification using chromatic
    pitch sequences. The model uses:
    - 36 hidden states (corresponding to chromatic pitches)
    - 36 observation symbols (chromatic bins 0-35)
    - Fully connected (ergodic) topology
    """
    
    def __init__(self, n_states: int = 36, n_observations: int = 36, random_state: Optional[int] = None):
        """
        Initialize DiscreteHMM with specified dimensions.
        
        Args:
            n_states: Number of hidden states (default: 36)
            n_observations: Number of observation symbols (default: 36)
            random_state: Random seed for reproducible initialization
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize parameters
        self.pi = self._init_initial_probabilities()
        self.A = self._init_transition_matrix()
        self.B = self._init_emission_matrix()
        
        logger.debug(f"Initialized DiscreteHMM with {n_states} states and {n_observations} observations")
    
    def _init_initial_probabilities(self) -> np.ndarray:
        """
        Initialize uniform initial state probabilities.
        
        Returns:
            pi: Initial state probabilities [n_states]
        """
        pi = np.ones(self.n_states) / self.n_states
        return pi
    
    def _init_transition_matrix(self) -> np.ndarray:
        """
        Initialize random transition matrix with stochastic properties.
        
        Returns:
            A: Transition matrix [n_states, n_states] where A[i,j] = P(q_t+1=j | q_t=i)
        """
        # Random initialization
        A = np.random.rand(self.n_states, self.n_states)
        
        # Normalize rows to make stochastic
        A = A / A.sum(axis=1, keepdims=True)
        
        return A
    
    def _init_emission_matrix(self) -> np.ndarray:
        """
        Initialize random emission matrix with stochastic properties.
        
        Returns:
            B: Emission matrix [n_states, n_observations] where B[i,k] = P(o_t=k | q_t=i)
        """
        # Random initialization
        B = np.random.rand(self.n_states, self.n_observations)
        
        # Normalize rows to make stochastic
        B = B / B.sum(axis=1, keepdims=True)
        
        return B
    
    def validate_stochastic_matrices(self) -> bool:
        """
        Validate that all probability matrices satisfy stochastic properties.
        
        Returns:
            bool: True if all matrices are valid stochastic matrices
        
        Raises:
            ValueError: If any matrix violates stochastic properties
        """
        tolerance = 1e-10
        
        # Check initial probabilities
        if not np.allclose(self.pi.sum(), 1.0, atol=tolerance):
            raise ValueError(f"Initial probabilities sum to {self.pi.sum()}, expected 1.0")
        
        if np.any(self.pi < 0):
            raise ValueError("Initial probabilities contain negative values")
        
        # Check transition matrix
        row_sums_A = self.A.sum(axis=1)
        if not np.allclose(row_sums_A, 1.0, atol=tolerance):
            raise ValueError(f"Transition matrix rows don't sum to 1.0: {row_sums_A}")
        
        if np.any(self.A < 0):
            raise ValueError("Transition matrix contains negative values")
        
        # Check emission matrix
        row_sums_B = self.B.sum(axis=1)
        if not np.allclose(row_sums_B, 1.0, atol=tolerance):
            raise ValueError(f"Emission matrix rows don't sum to 1.0: {row_sums_B}")
        
        if np.any(self.B < 0):
            raise ValueError("Emission matrix contains negative values")
        
        logger.debug("All stochastic matrix properties validated successfully")
        return True
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current model parameters.
        
        Returns:
            Tuple of (pi, A, B) parameters
        """
        return self.pi.copy(), self.A.copy(), self.B.copy()
    
    def set_parameters(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> None:
        """
        Set model parameters and validate dimensions.
        
        Args:
            pi: Initial state probabilities [n_states]
            A: Transition matrix [n_states, n_states]
            B: Emission matrix [n_states, n_observations]
        
        Raises:
            ValueError: If parameter dimensions don't match model configuration
        """
        # Validate dimensions
        if pi.shape != (self.n_states,):
            raise ValueError(f"pi shape {pi.shape} doesn't match expected ({self.n_states},)")
        
        if A.shape != (self.n_states, self.n_states):
            raise ValueError(f"A shape {A.shape} doesn't match expected ({self.n_states}, {self.n_states})")
        
        if B.shape != (self.n_states, self.n_observations):
            raise ValueError(f"B shape {B.shape} doesn't match expected ({self.n_states}, {self.n_observations})")
        
        # Set parameters
        self.pi = pi.copy()
        self.A = A.copy()
        self.B = B.copy()
        
        # Validate stochastic properties
        self.validate_stochastic_matrices()
        
        logger.debug("Model parameters updated and validated")
    
    def forward_backward_scaled(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute forward-backward algorithm with scaling to prevent numerical underflow.
        
        Args:
            observations: Sequence of observation indices [T]
        
        Returns:
            Tuple of:
            - alpha: Scaled forward probabilities [T, n_states]
            - beta: Scaled backward probabilities [T, n_states]
            - c_scale: Scaling coefficients [T]
            - log_likelihood: Log-likelihood of the observation sequence
        
        Raises:
            ValueError: If observations contain invalid indices
        """
        T = len(observations)
        
        # Validate observations
        if np.any(observations < 0) or np.any(observations >= self.n_observations):
            raise ValueError(f"Observations must be in range [0, {self.n_observations-1}]")
        
        # Initialize arrays
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        c_scale = np.zeros(T)
        
        # Forward pass with scaling
        # t = 0: Initialize
        alpha[0, :] = self.pi * self.B[:, observations[0]]
        c_scale[0] = alpha[0, :].sum()
        
        if c_scale[0] == 0:
            raise ValueError("Initial forward probabilities sum to zero - check model parameters")
        
        alpha[0, :] /= c_scale[0]
        
        # t = 1, ..., T-1: Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1, :] * self.A[:, j]) * self.B[j, observations[t]]
            
            c_scale[t] = alpha[t, :].sum()
            
            if c_scale[t] == 0:
                raise ValueError(f"Forward probabilities sum to zero at time {t}")
            
            alpha[t, :] /= c_scale[t]
        
        # Backward pass with scaling
        # t = T-1: Initialize
        beta[T-1, :] = 1.0
        
        # t = T-2, ..., 0: Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :])
            
            beta[t, :] /= c_scale[t]
        
        # Compute log-likelihood
        # log P(O|λ) = sum(log(c_t)) where c_t are scaling coefficients
        # This is because we scaled by dividing, so we need to add back the logs
        log_likelihood = np.sum(np.log(c_scale))
        
        logger.debug(f"Forward-backward completed: T={T}, log_likelihood={log_likelihood:.6f}")
        
        return alpha, beta, c_scale, log_likelihood
    
    def score(self, observations: np.ndarray) -> float:
        """
        Compute log-likelihood of observation sequence using forward algorithm.
        
        Args:
            observations: Sequence of observation indices [T]
        
        Returns:
            Log-likelihood of the observation sequence
        """
        _, _, c_scale, log_likelihood = self.forward_backward_scaled(observations)
        return log_likelihood
    
    def update_parameters(self, observations_list: list, regularization_alpha: float = 0.01, 
                         probability_floor: float = 1e-8) -> float:
        """
        Update HMM parameters using Baum-Welch algorithm (M-step of EM).
        
        Args:
            observations_list: List of observation sequences for training
            regularization_alpha: Dirichlet regularization parameter (default: 0.01)
            probability_floor: Minimum probability value to prevent zeros (default: 1e-8)
        
        Returns:
            Total log-likelihood across all sequences
        
        Raises:
            ValueError: If observations_list is empty or contains invalid sequences
        """
        if not observations_list:
            raise ValueError("observations_list cannot be empty")
        
        # Initialize sufficient statistics
        pi_numerator = np.zeros(self.n_states)
        A_numerator = np.zeros((self.n_states, self.n_states))
        B_numerator = np.zeros((self.n_states, self.n_observations))
        
        A_denominator = np.zeros(self.n_states)
        B_denominator = np.zeros(self.n_states)
        
        total_log_likelihood = 0.0
        
        # Accumulate statistics from all sequences
        for seq_idx, observations in enumerate(observations_list):
            observations = np.array(observations)
            
            if len(observations) == 0:
                raise ValueError(f"Sequence {seq_idx} is empty")
            
            # Run forward-backward algorithm
            alpha, beta, c_scale, log_likelihood = self.forward_backward_scaled(observations)
            total_log_likelihood += log_likelihood
            
            T = len(observations)
            
            # Compute gamma (state posterior probabilities)
            gamma = alpha * beta
            # Normalize gamma (should already be normalized due to scaling, but ensure it)
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            
            # Compute xi (transition posterior probabilities)
            xi = np.zeros((T-1, self.n_states, self.n_states))
            
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                     self.B[j, observations[t+1]] * beta[t+1, j])
                
                # Normalize xi[t]
                xi_sum = xi[t].sum()
                if xi_sum > 0:
                    xi[t] /= xi_sum
            
            # Accumulate sufficient statistics
            # Initial state probabilities
            pi_numerator += gamma[0, :]
            
            # Transition probabilities
            for i in range(self.n_states):
                A_denominator[i] += gamma[:-1, i].sum()  # Sum over t=0 to T-2
                for j in range(self.n_states):
                    A_numerator[i, j] += xi[:, i, j].sum()
            
            # Emission probabilities
            for i in range(self.n_states):
                B_denominator[i] += gamma[:, i].sum()  # Sum over all t
                for k in range(self.n_observations):
                    # Sum gamma[t, i] for all t where observations[t] == k
                    mask = (observations == k)
                    B_numerator[i, k] += gamma[mask, i].sum()
        
        # Update parameters with regularization
        # Initial probabilities (add Dirichlet prior)
        pi_new = pi_numerator + regularization_alpha
        pi_new = pi_new / pi_new.sum()
        
        # Transition probabilities (add Dirichlet prior)
        A_new = A_numerator + regularization_alpha
        for i in range(self.n_states):
            if A_denominator[i] + self.n_states * regularization_alpha > 0:
                A_new[i, :] = A_new[i, :] / (A_denominator[i] + self.n_states * regularization_alpha)
            else:
                # Fallback to uniform if denominator is zero
                A_new[i, :] = 1.0 / self.n_states
        
        # Emission probabilities (add Dirichlet prior)
        B_new = B_numerator + regularization_alpha
        for i in range(self.n_states):
            if B_denominator[i] + self.n_observations * regularization_alpha > 0:
                B_new[i, :] = B_new[i, :] / (B_denominator[i] + self.n_observations * regularization_alpha)
            else:
                # Fallback to uniform if denominator is zero
                B_new[i, :] = 1.0 / self.n_observations
        
        # Apply probability floors
        pi_new = np.maximum(pi_new, probability_floor)
        pi_new = pi_new / pi_new.sum()  # Renormalize after flooring
        
        A_new = np.maximum(A_new, probability_floor)
        A_new = A_new / A_new.sum(axis=1, keepdims=True)  # Renormalize rows
        
        B_new = np.maximum(B_new, probability_floor)
        B_new = B_new / B_new.sum(axis=1, keepdims=True)  # Renormalize rows
        
        # Update model parameters
        self.pi = pi_new
        self.A = A_new
        self.B = B_new
        
        logger.debug(f"Parameters updated: total_log_likelihood={total_log_likelihood:.6f}")
        
        return total_log_likelihood
    
    def train(self, observations_list: list, max_iterations: int = 200, 
              convergence_tolerance: float = 0.1, regularization_alpha: float = 0.01,
              probability_floor: float = 1e-8, verbose: bool = False) -> dict:
        """
        Train HMM using Baum-Welch algorithm with convergence monitoring.
        
        Args:
            observations_list: List of observation sequences for training
            max_iterations: Maximum number of EM iterations (default: 200)
            convergence_tolerance: Stop when log-likelihood improvement < tolerance (default: 0.1)
            regularization_alpha: Dirichlet regularization parameter (default: 0.01)
            probability_floor: Minimum probability value (default: 1e-8)
            verbose: Print training progress (default: False)
        
        Returns:
            Dictionary with training statistics:
            - 'converged': Whether training converged
            - 'iterations': Number of iterations performed
            - 'final_log_likelihood': Final log-likelihood
            - 'log_likelihood_history': List of log-likelihoods per iteration
            - 'improvement_history': List of improvements per iteration
        
        Raises:
            ValueError: If observations_list is empty or contains invalid sequences
        """
        if not observations_list:
            raise ValueError("observations_list cannot be empty")
        
        # Validate all sequences
        for seq_idx, observations in enumerate(observations_list):
            observations = np.array(observations)
            if len(observations) == 0:
                raise ValueError(f"Sequence {seq_idx} is empty")
            if np.any(observations < 0) or np.any(observations >= self.n_observations):
                raise ValueError(f"Sequence {seq_idx} contains invalid observation indices")
        
        # Initialize training statistics
        log_likelihood_history = []
        improvement_history = []
        converged = False
        
        # Initial log-likelihood
        prev_log_likelihood = self.compute_total_log_likelihood(observations_list)
        log_likelihood_history.append(prev_log_likelihood)
        
        if verbose:
            logger.info(f"Starting HMM training with {len(observations_list)} sequences")
            logger.info(f"Initial log-likelihood: {prev_log_likelihood:.6f}")
        
        # EM iterations
        for iteration in range(max_iterations):
            # M-step: Update parameters
            current_log_likelihood = self.update_parameters(
                observations_list, 
                regularization_alpha=regularization_alpha,
                probability_floor=probability_floor
            )
            
            # Compute improvement
            improvement = current_log_likelihood - prev_log_likelihood
            log_likelihood_history.append(current_log_likelihood)
            improvement_history.append(improvement)
            
            if verbose:
                logger.info(f"Iteration {iteration + 1}: log_likelihood={current_log_likelihood:.6f}, "
                          f"improvement={improvement:.6f}")
            
            # Check convergence
            if improvement < convergence_tolerance:
                converged = True
                if verbose:
                    logger.info(f"Converged after {iteration + 1} iterations "
                              f"(improvement {improvement:.6f} < tolerance {convergence_tolerance})")
                break
            
            # Check for decreasing log-likelihood (should not happen with proper EM)
            if improvement < -1e-6:  # Allow small numerical errors
                logger.warning(f"Log-likelihood decreased by {-improvement:.6f} at iteration {iteration + 1}")
            
            prev_log_likelihood = current_log_likelihood
        
        if not converged and verbose:
            logger.info(f"Training stopped after {max_iterations} iterations without convergence")
        
        # Prepare training statistics
        training_stats = {
            'converged': converged,
            'iterations': len(improvement_history),
            'final_log_likelihood': log_likelihood_history[-1],
            'log_likelihood_history': log_likelihood_history,
            'improvement_history': improvement_history
        }
        
        logger.debug(f"Training completed: converged={converged}, iterations={len(improvement_history)}")
        
        return training_stats
    
    def compute_total_log_likelihood(self, observations_list: list) -> float:
        """
        Compute total log-likelihood across all observation sequences.
        
        Args:
            observations_list: List of observation sequences
        
        Returns:
            Total log-likelihood across all sequences
        """
        total_log_likelihood = 0.0
        
        for observations in observations_list:
            observations = np.array(observations)
            log_likelihood = self.score(observations)
            total_log_likelihood += log_likelihood
        
        return total_log_likelihood
    
    def __repr__(self) -> str:
        """String representation of the HMM."""
        return f"DiscreteHMM(n_states={self.n_states}, n_observations={self.n_observations})"