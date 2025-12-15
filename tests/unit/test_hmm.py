"""
Unit tests for the HMM module.

Tests for the DiscreteHMM class and its methods.
"""

import numpy as np
import pytest
from src.raag_hmm.hmm.model import DiscreteHMM


class TestDiscreteHMM:
    """Test cases for the DiscreteHMM class."""
    
    def test_initialization(self):
        """Test HMM initialization with default parameters."""
        hmm = DiscreteHMM(n_states=5, n_observations=10)
        
        assert hmm.n_states == 5
        assert hmm.n_observations == 10
        assert hmm.pi.shape == (5,)
        assert hmm.A.shape == (5, 5)
        assert hmm.B.shape == (5, 10)
        
        # Check stochastic properties
        assert np.allclose(hmm.pi.sum(), 1.0)
        assert np.allclose(hmm.A.sum(axis=1), 1.0)
        assert np.allclose(hmm.B.sum(axis=1), 1.0)
    
    def test_initialization_with_random_state(self):
        """Test HMM initialization with reproducible random state."""
        hmm1 = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        hmm2 = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        
        pi1, A1, B1 = hmm1.get_parameters()
        pi2, A2, B2 = hmm2.get_parameters()
        
        np.testing.assert_array_almost_equal(pi1, pi2)
        np.testing.assert_array_almost_equal(A1, A2)
        np.testing.assert_array_almost_equal(B1, B2)
    
    def test_validate_stochastic_matrices(self):
        """Test stochastic matrix validation."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        
        # Should pass for valid matrices
        assert hmm.validate_stochastic_matrices() is True
        
        # Modify to create invalid matrix and test
        hmm.pi[0] = -0.1  # Negative probability
        with pytest.raises(ValueError):
            hmm.validate_stochastic_matrices()
    
    def test_forward_backward_scaled(self):
        """Test forward-backward algorithm."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        observations = np.array([0, 1, 2, 1, 0])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Check shapes
        assert alpha.shape == (len(observations), hmm.n_states)
        assert beta.shape == (len(observations), hmm.n_states)
        assert c_scale.shape == (len(observations),)
        assert isinstance(log_likelihood, float)
        
        # Check scaling factors are positive
        assert np.all(c_scale > 0)
    
    def test_score(self):
        """Test scoring of observation sequences."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        observations = np.array([0, 1, 2, 1, 0])
        
        score = hmm.score(observations)
        
        # Should return a float
        assert isinstance(score, float)
    
    def test_update_parameters(self):
        """Test parameter updates using Baum-Welch."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        sequences = [np.array([0, 1, 2, 1]), np.array([1, 2, 0, 1])]
        
        total_log_likelihood = hmm.update_parameters(
            sequences, 
            regularization_alpha=0.01,
            probability_floor=1e-8
        )
        
        # Should return a float
        assert isinstance(total_log_likelihood, float)
    
    def test_train(self):
        """Test complete training process."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        sequences = [np.array([0, 1, 2, 1]), np.array([1, 2, 0, 1, 0])]
        
        training_stats = hmm.train(
            sequences,
            max_iterations=5,
            convergence_tolerance=0.01,
            verbose=False
        )
        
        # Check that training completed
        assert isinstance(training_stats, dict)
        assert 'converged' in training_stats
        assert 'iterations' in training_stats
        assert 'final_log_likelihood' in training_stats
    
    def test_get_set_parameters(self):
        """Test parameter getting and setting."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        
        # Get original parameters
        original_pi, original_A, original_B = hmm.get_parameters()
        
        # Create new parameters
        new_pi = np.array([0.5, 0.3, 0.2])
        new_A = np.array([[0.7, 0.2, 0.1],
                         [0.1, 0.8, 0.1],
                         [0.2, 0.3, 0.5]])
        new_B = np.array([[0.8, 0.1, 0.05, 0.05],
                         [0.1, 0.8, 0.05, 0.05],
                         [0.05, 0.05, 0.8, 0.1]])
        
        # Set new parameters
        hmm.set_parameters(new_pi, new_A, new_B)
        
        # Get parameters back
        current_pi, current_A, current_B = hmm.get_parameters()
        
        np.testing.assert_array_almost_equal(current_pi, new_pi)
        np.testing.assert_array_almost_equal(current_A, new_A)
        np.testing.assert_array_almost_equal(current_B, new_B)
    
    def test_compute_total_log_likelihood(self):
        """Test total log-likelihood computation."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        sequences = [np.array([0, 1, 2]), np.array([1, 2, 0])]
        
        total_ll = hmm.compute_total_log_likelihood(sequences)
        
        assert isinstance(total_ll, float)
    
    def test_invalid_observations(self):
        """Test handling of invalid observations."""
        hmm = DiscreteHMM(n_states=3, n_observations=4, random_state=42)
        invalid_obs = np.array([0, 1, 5, 1])  # 5 is out of bounds for 4 observations
        
        with pytest.raises(ValueError):
            hmm.score(invalid_obs)
        
        with pytest.raises(ValueError):
            hmm.forward_backward_scaled(invalid_obs)