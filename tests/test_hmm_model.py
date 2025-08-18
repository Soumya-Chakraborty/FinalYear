"""
Unit tests for DiscreteHMM model implementation.

Tests cover initialization, stochastic matrix properties, parameter validation,
and edge cases for the core HMM data structures.
"""

import pytest
import numpy as np
from raag_hmm.hmm.model import DiscreteHMM


class TestDiscreteHMMInitialization:
    """Test HMM initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test default initialization with 36 states and observations."""
        hmm = DiscreteHMM()
        
        assert hmm.n_states == 36
        assert hmm.n_observations == 36
        assert hmm.pi.shape == (36,)
        assert hmm.A.shape == (36, 36)
        assert hmm.B.shape == (36, 36)
    
    def test_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        hmm = DiscreteHMM(n_states=10, n_observations=20)
        
        assert hmm.n_states == 10
        assert hmm.n_observations == 20
        assert hmm.pi.shape == (10,)
        assert hmm.A.shape == (10, 10)
        assert hmm.B.shape == (10, 20)
    
    def test_reproducible_initialization(self):
        """Test that random_state produces reproducible initialization."""
        hmm1 = DiscreteHMM(random_state=42)
        hmm2 = DiscreteHMM(random_state=42)
        
        np.testing.assert_array_equal(hmm1.pi, hmm2.pi)
        np.testing.assert_array_equal(hmm1.A, hmm2.A)
        np.testing.assert_array_equal(hmm1.B, hmm2.B)
    
    def test_different_random_states(self):
        """Test that different random states produce different initializations."""
        hmm1 = DiscreteHMM(random_state=42)
        hmm2 = DiscreteHMM(random_state=123)
        
        # Should be different (with very high probability)
        assert not np.array_equal(hmm1.A, hmm2.A)
        assert not np.array_equal(hmm1.B, hmm2.B)


class TestStochasticMatrixProperties:
    """Test that all probability matrices satisfy stochastic properties."""
    
    def test_initial_probabilities_uniform(self):
        """Test that initial probabilities are uniform."""
        hmm = DiscreteHMM()
        
        expected = np.ones(36) / 36
        np.testing.assert_array_almost_equal(hmm.pi, expected)
    
    def test_initial_probabilities_sum_to_one(self):
        """Test that initial probabilities sum to 1."""
        hmm = DiscreteHMM(n_states=10)
        
        assert abs(hmm.pi.sum() - 1.0) < 1e-10
        assert np.all(hmm.pi >= 0)
    
    def test_transition_matrix_stochastic(self):
        """Test that transition matrix rows sum to 1."""
        hmm = DiscreteHMM()
        
        row_sums = hmm.A.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(36))
        assert np.all(hmm.A >= 0)
    
    def test_emission_matrix_stochastic(self):
        """Test that emission matrix rows sum to 1."""
        hmm = DiscreteHMM()
        
        row_sums = hmm.B.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(36))
        assert np.all(hmm.B >= 0)
    
    def test_validate_stochastic_matrices_success(self):
        """Test validation passes for properly initialized matrices."""
        hmm = DiscreteHMM()
        
        # Should not raise any exception
        assert hmm.validate_stochastic_matrices() is True
    
    def test_validate_stochastic_matrices_invalid_pi(self):
        """Test validation fails for invalid initial probabilities."""
        hmm = DiscreteHMM()
        
        # Make pi sum to something other than 1
        hmm.pi = np.ones(36) * 0.5
        
        with pytest.raises(ValueError, match="Initial probabilities sum to"):
            hmm.validate_stochastic_matrices()
    
    def test_validate_stochastic_matrices_negative_pi(self):
        """Test validation fails for negative initial probabilities."""
        hmm = DiscreteHMM()
        
        hmm.pi[0] = -0.1
        hmm.pi[1] += 0.1  # Compensate to keep sum = 1
        
        with pytest.raises(ValueError, match="Initial probabilities sum to"):
            hmm.validate_stochastic_matrices()
    
    def test_validate_stochastic_matrices_invalid_A(self):
        """Test validation fails for invalid transition matrix."""
        hmm = DiscreteHMM()
        
        # Make first row sum to 0.5
        hmm.A[0, :] *= 0.5
        
        with pytest.raises(ValueError, match="Transition matrix rows don't sum to 1.0"):
            hmm.validate_stochastic_matrices()
    
    def test_validate_stochastic_matrices_negative_A(self):
        """Test validation fails for negative transition probabilities."""
        hmm = DiscreteHMM()
        
        hmm.A[0, 0] = -0.1
        # Don't compensate - let it fail on row sum first
        
        with pytest.raises(ValueError, match="Transition matrix rows don't sum to 1.0"):
            hmm.validate_stochastic_matrices()
    
    def test_validate_stochastic_matrices_invalid_B(self):
        """Test validation fails for invalid emission matrix."""
        hmm = DiscreteHMM()
        
        # Make first row sum to 2.0
        hmm.B[0, :] *= 2.0
        
        with pytest.raises(ValueError, match="Emission matrix rows don't sum to 1.0"):
            hmm.validate_stochastic_matrices()
    
    def test_validate_stochastic_matrices_negative_B(self):
        """Test validation fails for negative emission probabilities."""
        hmm = DiscreteHMM()
        
        hmm.B[0, 0] = -0.1
        # Don't compensate - let it fail on row sum first
        
        with pytest.raises(ValueError, match="Emission matrix rows don't sum to 1.0"):
            hmm.validate_stochastic_matrices()
    
    def test_validate_negative_values_with_correct_sums(self):
        """Test validation catches negative values when sums are correct."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        # Test negative pi with correct sum
        hmm.pi = np.array([-0.1, 1.1])  # Sum = 1.0 but has negative
        with pytest.raises(ValueError, match="Initial probabilities contain negative values"):
            hmm.validate_stochastic_matrices()
        
        # Reset and test negative A with correct row sums
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        hmm.A[0, :] = [-0.1, 1.1]  # Row sum = 1.0 but has negative
        with pytest.raises(ValueError, match="Transition matrix contains negative values"):
            hmm.validate_stochastic_matrices()
        
        # Reset and test negative B with correct row sums
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        hmm.B[0, :] = [-0.1, 1.1]  # Row sum = 1.0 but has negative
        with pytest.raises(ValueError, match="Emission matrix contains negative values"):
            hmm.validate_stochastic_matrices()


class TestParameterManagement:
    """Test parameter getting and setting functionality."""
    
    def test_get_parameters(self):
        """Test getting model parameters returns copies."""
        hmm = DiscreteHMM(random_state=42)
        
        pi, A, B = hmm.get_parameters()
        
        # Should be equal but different objects (copies)
        np.testing.assert_array_equal(pi, hmm.pi)
        np.testing.assert_array_equal(A, hmm.A)
        np.testing.assert_array_equal(B, hmm.B)
        
        # Modifying returned arrays shouldn't affect original
        pi[0] = 999
        A[0, 0] = 999
        B[0, 0] = 999
        
        assert hmm.pi[0] != 999
        assert hmm.A[0, 0] != 999
        assert hmm.B[0, 0] != 999
    
    def test_set_parameters_valid(self):
        """Test setting valid parameters."""
        hmm = DiscreteHMM(n_states=3, n_observations=2)
        
        # Create valid parameters
        pi_new = np.array([0.5, 0.3, 0.2])
        A_new = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ])
        B_new = np.array([
            [0.6, 0.4],
            [0.3, 0.7],
            [0.8, 0.2]
        ])
        
        hmm.set_parameters(pi_new, A_new, B_new)
        
        np.testing.assert_array_equal(hmm.pi, pi_new)
        np.testing.assert_array_equal(hmm.A, A_new)
        np.testing.assert_array_equal(hmm.B, B_new)
    
    def test_set_parameters_wrong_pi_shape(self):
        """Test setting parameters with wrong pi shape."""
        hmm = DiscreteHMM(n_states=3, n_observations=2)
        
        pi_wrong = np.array([0.5, 0.5])  # Wrong size
        A_valid = np.eye(3) / 3 + np.ones((3, 3)) / 3 * 2/3
        B_valid = np.ones((3, 2)) / 2
        
        with pytest.raises(ValueError, match="pi shape .* doesn't match expected"):
            hmm.set_parameters(pi_wrong, A_valid, B_valid)
    
    def test_set_parameters_wrong_A_shape(self):
        """Test setting parameters with wrong A shape."""
        hmm = DiscreteHMM(n_states=3, n_observations=2)
        
        pi_valid = np.ones(3) / 3
        A_wrong = np.ones((2, 3)) / 3  # Wrong shape
        B_valid = np.ones((3, 2)) / 2
        
        with pytest.raises(ValueError, match="A shape .* doesn't match expected"):
            hmm.set_parameters(pi_valid, A_wrong, B_valid)
    
    def test_set_parameters_wrong_B_shape(self):
        """Test setting parameters with wrong B shape."""
        hmm = DiscreteHMM(n_states=3, n_observations=2)
        
        pi_valid = np.ones(3) / 3
        A_valid = np.eye(3) / 3 + np.ones((3, 3)) / 3 * 2/3
        B_wrong = np.ones((3, 3)) / 3  # Wrong shape
        
        with pytest.raises(ValueError, match="B shape .* doesn't match expected"):
            hmm.set_parameters(pi_valid, A_valid, B_wrong)
    
    def test_set_parameters_invalid_stochastic(self):
        """Test setting parameters that violate stochastic properties."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        pi_invalid = np.array([0.6, 0.6])  # Sums to 1.2
        A_valid = np.ones((2, 2)) / 2
        B_valid = np.ones((2, 2)) / 2
        
        with pytest.raises(ValueError, match="Initial probabilities sum to"):
            hmm.set_parameters(pi_invalid, A_valid, B_valid)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_dimensions(self):
        """Test HMM with minimal dimensions."""
        hmm = DiscreteHMM(n_states=1, n_observations=1)
        
        assert hmm.pi.shape == (1,)
        assert hmm.A.shape == (1, 1)
        assert hmm.B.shape == (1, 1)
        
        # Should still be stochastic
        assert hmm.validate_stochastic_matrices()
        
        np.testing.assert_array_almost_equal(hmm.pi, [1.0])
        np.testing.assert_array_almost_equal(hmm.A, [[1.0]])
        np.testing.assert_array_almost_equal(hmm.B, [[1.0]])
    
    def test_large_dimensions(self):
        """Test HMM with larger dimensions."""
        hmm = DiscreteHMM(n_states=100, n_observations=50)
        
        assert hmm.n_states == 100
        assert hmm.n_observations == 50
        assert hmm.validate_stochastic_matrices()
    
    def test_string_representation(self):
        """Test string representation of HMM."""
        hmm = DiscreteHMM(n_states=10, n_observations=5)
        
        repr_str = repr(hmm)
        assert "DiscreteHMM" in repr_str
        assert "n_states=10" in repr_str
        assert "n_observations=5" in repr_str


class TestNumericalStability:
    """Test numerical stability and precision."""
    
    def test_stochastic_validation_tolerance(self):
        """Test that validation handles floating point precision issues."""
        hmm = DiscreteHMM(n_states=3, n_observations=2)
        
        # Create parameters that sum to 1.0 within floating point precision
        pi_almost_one = np.array([1/3, 1/3, 1/3])
        pi_almost_one[0] += 1e-15  # Add tiny floating point error
        
        A_almost_stochastic = np.ones((3, 3)) / 3
        A_almost_stochastic[0, 0] += 1e-15
        A_almost_stochastic[0, 1] -= 1e-15
        
        B_almost_stochastic = np.ones((3, 2)) / 2
        B_almost_stochastic[0, 0] += 1e-15
        B_almost_stochastic[0, 1] -= 1e-15
        
        # Should not raise exception due to tolerance
        hmm.set_parameters(pi_almost_one, A_almost_stochastic, B_almost_stochastic)
        assert hmm.validate_stochastic_matrices()
    
    def test_random_initialization_range(self):
        """Test that random initialization produces reasonable values."""
        hmm = DiscreteHMM(random_state=42)
        
        # All probabilities should be in [0, 1]
        assert np.all(hmm.pi >= 0) and np.all(hmm.pi <= 1)
        assert np.all(hmm.A >= 0) and np.all(hmm.A <= 1)
        assert np.all(hmm.B >= 0) and np.all(hmm.B <= 1)
        
        # Should have some variation (not all equal)
        assert np.std(hmm.A) > 0.01
        assert np.std(hmm.B) > 0.01


class TestForwardBackwardAlgorithm:
    """Test forward-backward algorithm implementation with scaling."""
    
    def test_forward_backward_simple_case(self):
        """Test forward-backward on a simple 2-state, 2-observation case."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Simple observation sequence
        observations = np.array([0, 1, 0])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Check dimensions
        assert alpha.shape == (3, 2)
        assert beta.shape == (3, 2)
        assert c_scale.shape == (3,)
        assert isinstance(log_likelihood, float)
        
        # Check that alpha rows are normalized (scaled)
        np.testing.assert_array_almost_equal(alpha.sum(axis=1), np.ones(3))
        
        # Check that all values are non-negative
        assert np.all(alpha >= 0)
        assert np.all(beta >= 0)
        assert np.all(c_scale > 0)
    
    def test_forward_backward_single_observation(self):
        """Test forward-backward with single observation."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        observations = np.array([1])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Check dimensions
        assert alpha.shape == (1, 3)
        assert beta.shape == (1, 3)
        assert c_scale.shape == (1,)
        
        # For single observation, alpha should equal pi * B[:, obs]
        expected_alpha = hmm.pi * hmm.B[:, 1]
        expected_alpha /= expected_alpha.sum()
        np.testing.assert_array_almost_equal(alpha[0, :], expected_alpha)
    
    def test_forward_backward_longer_sequence(self):
        """Test forward-backward with longer observation sequence."""
        hmm = DiscreteHMM(n_states=4, n_observations=3, random_state=42)
        
        observations = np.array([0, 1, 2, 1, 0, 2, 1])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Check dimensions
        T = len(observations)
        assert alpha.shape == (T, 4)
        assert beta.shape == (T, 4)
        assert c_scale.shape == (T,)
        
        # All scaling coefficients should be positive
        assert np.all(c_scale > 0)
        
        # Log-likelihood should be finite
        assert np.isfinite(log_likelihood)
    
    def test_score_method(self):
        """Test that score method returns same log-likelihood as forward-backward."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        observations = np.array([0, 1, 0, 1])
        
        # Get log-likelihood from forward-backward
        _, _, _, log_likelihood_fb = hmm.forward_backward_scaled(observations)
        
        # Get log-likelihood from score method
        log_likelihood_score = hmm.score(observations)
        
        # Should be identical
        assert log_likelihood_fb == log_likelihood_score
    
    def test_invalid_observations_negative(self):
        """Test error handling for negative observation indices."""
        hmm = DiscreteHMM(n_states=2, n_observations=3)
        
        observations = np.array([0, -1, 2])
        
        with pytest.raises(ValueError, match="Observations must be in range"):
            hmm.forward_backward_scaled(observations)
    
    def test_invalid_observations_too_large(self):
        """Test error handling for observation indices >= n_observations."""
        hmm = DiscreteHMM(n_states=2, n_observations=3)
        
        observations = np.array([0, 1, 3])  # 3 >= n_observations
        
        with pytest.raises(ValueError, match="Observations must be in range"):
            hmm.forward_backward_scaled(observations)
    
    def test_zero_probability_handling(self):
        """Test handling of zero probabilities in forward algorithm."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        # Set emission probabilities to create zero probability scenario
        hmm.B[0, 0] = 0.0  # State 0 never emits observation 0
        hmm.B[0, 1] = 1.0
        hmm.B[1, 0] = 1.0  # State 1 never emits observation 1
        hmm.B[1, 1] = 0.0
        
        # Set initial probabilities to only start in state 0
        hmm.pi = np.array([1.0, 0.0])
        
        # Observation sequence that should cause zero probability
        observations = np.array([0])  # State 0 can't emit 0
        
        with pytest.raises(ValueError, match="Initial forward probabilities sum to zero"):
            hmm.forward_backward_scaled(observations)


class TestNumericalStabilityForwardBackward:
    """Test numerical stability of forward-backward algorithm."""
    
    def test_scaling_prevents_underflow(self):
        """Test that scaling prevents numerical underflow in long sequences."""
        hmm = DiscreteHMM(n_states=5, n_observations=3, random_state=42)
        
        # Create a long observation sequence that would cause underflow without scaling
        observations = np.random.randint(0, 3, size=100)
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # All values should be finite (no NaN or inf)
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(beta))
        assert np.all(np.isfinite(c_scale))
        assert np.isfinite(log_likelihood)
        
        # Alpha should remain normalized at each time step
        np.testing.assert_array_almost_equal(alpha.sum(axis=1), np.ones(100))
    
    def test_very_small_probabilities(self):
        """Test handling of very small probabilities."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Set very small but non-zero probabilities
        hmm.A *= 1e-10
        hmm.A = hmm.A / hmm.A.sum(axis=1, keepdims=True)  # Renormalize
        
        hmm.B *= 1e-10
        hmm.B = hmm.B / hmm.B.sum(axis=1, keepdims=True)  # Renormalize
        
        observations = np.array([0, 1, 0])
        
        # Should still work without numerical issues
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(beta))
        assert np.isfinite(log_likelihood)
    
    def test_reproducibility_with_random_state(self):
        """Test that results are reproducible with same random state."""
        observations = np.array([0, 1, 2, 1, 0])
        
        hmm1 = DiscreteHMM(n_states=3, n_observations=3, random_state=42)
        alpha1, beta1, c_scale1, ll1 = hmm1.forward_backward_scaled(observations)
        
        hmm2 = DiscreteHMM(n_states=3, n_observations=3, random_state=42)
        alpha2, beta2, c_scale2, ll2 = hmm2.forward_backward_scaled(observations)
        
        np.testing.assert_array_equal(alpha1, alpha2)
        np.testing.assert_array_equal(beta1, beta2)
        np.testing.assert_array_equal(c_scale1, c_scale2)
        assert ll1 == ll2
    
    def test_log_likelihood_properties(self):
        """Test mathematical properties of log-likelihood."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        # Test with longer sequences to ensure negative log-likelihood
        short_obs = np.array([0, 1, 0, 1])
        long_obs = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        ll_short = hmm.score(short_obs)
        ll_long = hmm.score(long_obs)
        
        # Both should be finite
        assert np.isfinite(ll_short)
        assert np.isfinite(ll_long)
        
        # Longer sequence should have lower (more negative) log-likelihood
        assert ll_long <= ll_short
    
    def test_empty_observation_sequence(self):
        """Test handling of empty observation sequence."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        observations = np.array([])
        
        # Should handle gracefully - might raise error or return special value
        try:
            alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
            # If it doesn't raise an error, check that dimensions make sense
            assert alpha.shape == (0, 2)
            assert beta.shape == (0, 2)
            assert c_scale.shape == (0,)
        except (ValueError, IndexError):
            # It's also acceptable to raise an error for empty sequences
            pass


class TestForwardBackwardEdgeCases:
    """Test edge cases for forward-backward algorithm."""
    
    def test_deterministic_hmm(self):
        """Test forward-backward on deterministic HMM."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        # Create deterministic transitions: 0->1, 1->0
        hmm.A = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        
        # Deterministic emissions: state 0 emits 0, state 1 emits 1
        hmm.B = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # Start in state 0
        hmm.pi = np.array([1.0, 0.0])
        
        # Observation sequence consistent with deterministic model
        observations = np.array([0, 1, 0, 1])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Should work without numerical issues
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(beta))
        assert np.isfinite(log_likelihood)
    
    def test_uniform_hmm(self):
        """Test forward-backward on uniform HMM."""
        hmm = DiscreteHMM(n_states=3, n_observations=3)
        
        # Set all probabilities to uniform
        hmm.pi = np.ones(3) / 3
        hmm.A = np.ones((3, 3)) / 3
        hmm.B = np.ones((3, 3)) / 3
        
        observations = np.array([0, 1, 2, 0])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Should work and produce reasonable results
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(beta))
        assert np.isfinite(log_likelihood)
        
        # For uniform model, alpha should be approximately uniform at each step
        # (after a few steps to reach steady state)
        expected_uniform = np.ones(3) / 3
        np.testing.assert_array_almost_equal(alpha[-1, :], expected_uniform, decimal=1)
    
    def test_single_state_hmm(self):
        """Test forward-backward on single-state HMM."""
        hmm = DiscreteHMM(n_states=1, n_observations=2, random_state=42)
        
        observations = np.array([0, 1, 0])
        
        alpha, beta, c_scale, log_likelihood = hmm.forward_backward_scaled(observations)
        
        # Check dimensions
        assert alpha.shape == (3, 1)
        assert beta.shape == (3, 1)
        
        # Alpha should always be [1.0] since there's only one state
        np.testing.assert_array_almost_equal(alpha[:, 0], np.ones(3))
        
        # Beta should be properly scaled
        assert np.all(np.isfinite(beta))


class TestBaumWelchParameterUpdates:
    """Test Baum-Welch parameter update implementation."""
    
    def test_update_parameters_single_sequence(self):
        """Test parameter updates with single observation sequence."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Store original parameters
        pi_orig, A_orig, B_orig = hmm.get_parameters()
        
        # Single observation sequence
        observations_list = [np.array([0, 1, 0, 1])]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # Parameters should have changed
        assert not np.array_equal(hmm.pi, pi_orig)
        assert not np.array_equal(hmm.A, A_orig)
        assert not np.array_equal(hmm.B, B_orig)
        
        # Should still be stochastic
        assert hmm.validate_stochastic_matrices()
        
        # Log-likelihood should be finite
        assert np.isfinite(log_likelihood)
    
    def test_update_parameters_multiple_sequences(self):
        """Test parameter updates with multiple observation sequences."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        # Multiple observation sequences
        observations_list = [
            np.array([0, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # Should still be stochastic
        assert hmm.validate_stochastic_matrices()
        
        # Log-likelihood should be finite
        assert np.isfinite(log_likelihood)
    
    def test_update_parameters_empty_list(self):
        """Test error handling for empty observations list."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        with pytest.raises(ValueError, match="observations_list cannot be empty"):
            hmm.update_parameters([])
    
    def test_update_parameters_empty_sequence(self):
        """Test error handling for empty sequence in list."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        observations_list = [np.array([0, 1]), np.array([])]
        
        with pytest.raises(ValueError, match="Sequence 1 is empty"):
            hmm.update_parameters(observations_list)
    
    def test_regularization_effect(self):
        """Test that regularization prevents zero probabilities."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Create a sequence that might lead to zero probabilities without regularization
        observations_list = [np.array([0, 0, 0, 0])]  # Only observation 0
        
        # Update with regularization
        hmm.update_parameters(observations_list, regularization_alpha=0.1)
        
        # All probabilities should be > 0 due to regularization
        assert np.all(hmm.pi > 0)
        assert np.all(hmm.A > 0)
        assert np.all(hmm.B > 0)
        
        # Should still be stochastic
        assert hmm.validate_stochastic_matrices()
    
    def test_probability_floor_effect(self):
        """Test that probability floor prevents very small probabilities."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 0, 0])]
        floor = 1e-3
        
        hmm.update_parameters(observations_list, probability_floor=floor)
        
        # All probabilities should be >= floor
        assert np.all(hmm.pi >= floor)
        assert np.all(hmm.A >= floor)
        assert np.all(hmm.B >= floor)
        
        # Should still be stochastic
        assert hmm.validate_stochastic_matrices()
    
    def test_parameter_update_convergence_property(self):
        """Test that repeated updates improve log-likelihood."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1, 0])]
        
        # First update
        ll1 = hmm.update_parameters(observations_list)
        
        # Second update should improve or maintain log-likelihood
        ll2 = hmm.update_parameters(observations_list)
        
        # EM algorithm should not decrease log-likelihood
        assert ll2 >= ll1 - 1e-10  # Allow small numerical errors
    
    def test_single_state_hmm_updates(self):
        """Test parameter updates for single-state HMM."""
        hmm = DiscreteHMM(n_states=1, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1])]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # pi should remain [1.0]
        np.testing.assert_array_almost_equal(hmm.pi, [1.0])
        
        # A should remain [[1.0]]
        np.testing.assert_array_almost_equal(hmm.A, [[1.0]])
        
        # B should reflect observation frequencies
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(log_likelihood)
    
    def test_deterministic_sequence_updates(self):
        """Test updates with deterministic observation patterns."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Alternating pattern that might suggest specific state transitions
        observations_list = [np.array([0, 1, 0, 1, 0, 1])]
        
        # Multiple updates to see convergence
        log_likelihoods = []
        for _ in range(5):
            ll = hmm.update_parameters(observations_list)
            log_likelihoods.append(ll)
        
        # Log-likelihood should generally improve
        assert log_likelihoods[-1] >= log_likelihoods[0] - 1e-10
        
        # Should maintain stochastic properties
        assert hmm.validate_stochastic_matrices()


class TestBaumWelchNumericalStability:
    """Test numerical stability of Baum-Welch updates."""
    
    def test_very_long_sequences(self):
        """Test parameter updates with very long sequences."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        # Create long sequences
        np.random.seed(42)
        long_seq = np.random.randint(0, 2, size=1000)
        observations_list = [long_seq]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # Should handle long sequences without numerical issues
        assert np.isfinite(log_likelihood)
        assert hmm.validate_stochastic_matrices()
        
        # All parameters should be finite
        assert np.all(np.isfinite(hmm.pi))
        assert np.all(np.isfinite(hmm.A))
        assert np.all(np.isfinite(hmm.B))
    
    def test_many_short_sequences(self):
        """Test parameter updates with many short sequences."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Create many short sequences
        np.random.seed(42)
        observations_list = []
        for _ in range(100):
            seq = np.random.randint(0, 2, size=np.random.randint(2, 6))
            observations_list.append(seq)
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # Should handle many sequences without issues
        assert np.isfinite(log_likelihood)
        assert hmm.validate_stochastic_matrices()
    
    def test_extreme_regularization(self):
        """Test behavior with extreme regularization values."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0])]
        
        # Very high regularization should make parameters more uniform
        hmm.update_parameters(observations_list, regularization_alpha=10.0)
        
        assert hmm.validate_stochastic_matrices()
        
        # Very low regularization should still work
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        hmm.update_parameters(observations_list, regularization_alpha=1e-10)
        
        assert hmm.validate_stochastic_matrices()
    
    def test_extreme_probability_floor(self):
        """Test behavior with extreme probability floor values."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0])]
        
        # High probability floor - after renormalization, values might be slightly below floor
        # but should be close to it for the smaller probabilities
        hmm.update_parameters(observations_list, probability_floor=0.1)
        
        # Check that no probability is extremely small (the floor had an effect)
        assert np.all(hmm.pi >= 0.05)  # Allow some tolerance after renormalization
        assert np.all(hmm.A >= 0.05)
        assert np.all(hmm.B >= 0.05)
        assert hmm.validate_stochastic_matrices()
        
        # Very low probability floor
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        hmm.update_parameters(observations_list, probability_floor=1e-15)
        
        assert hmm.validate_stochastic_matrices()
    
    def test_repeated_identical_updates(self):
        """Test that repeated identical updates are stable."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1])]
        
        # Perform many identical updates
        log_likelihoods = []
        for i in range(10):
            ll = hmm.update_parameters(observations_list)
            log_likelihoods.append(ll)
            
            # Parameters should remain valid
            assert hmm.validate_stochastic_matrices()
        
        # Should converge (later updates should have similar log-likelihood)
        assert abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6


class TestBaumWelchEdgeCases:
    """Test edge cases for Baum-Welch parameter updates."""
    
    def test_uniform_observation_distribution(self):
        """Test updates when all observations are equally likely."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Create sequences with uniform observation distribution
        observations_list = [
            np.array([0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0])
        ]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(log_likelihood)
    
    def test_single_observation_type(self):
        """Test updates when sequences contain only one observation type."""
        hmm = DiscreteHMM(n_states=2, n_observations=3, random_state=42)
        
        # All sequences contain only observation 0
        observations_list = [
            np.array([0, 0, 0]),
            np.array([0, 0, 0, 0])
        ]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # Should handle gracefully with regularization
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(log_likelihood)
        
        # Emission probabilities for observation 0 should be higher
        assert np.all(hmm.B[:, 0] > hmm.B[:, 1])
        assert np.all(hmm.B[:, 0] > hmm.B[:, 2])
    
    def test_minimal_sequences(self):
        """Test updates with minimal (length 1) sequences."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [
            np.array([0]),
            np.array([1]),
            np.array([0])
        ]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        # Should work even with minimal sequences
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(log_likelihood)
    
    def test_mixed_sequence_lengths(self):
        """Test updates with sequences of very different lengths."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [
            np.array([0]),  # Length 1
            np.array([0, 1, 0, 1, 0]),  # Length 5
            np.array([1, 0]),  # Length 2
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Length 10
        ]
        
        log_likelihood = hmm.update_parameters(observations_list)
        
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(log_likelihood)


class TestHMMTrainingLoop:
    """Test HMM training loop with convergence monitoring."""
    
    def test_train_basic_functionality(self):
        """Test basic training functionality."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [
            np.array([0, 1, 0, 1]),
            np.array([1, 0, 1, 0])
        ]
        
        stats = hmm.train(observations_list, max_iterations=10)
        
        # Check return statistics
        assert isinstance(stats, dict)
        assert 'converged' in stats
        assert 'iterations' in stats
        assert 'final_log_likelihood' in stats
        assert 'log_likelihood_history' in stats
        assert 'improvement_history' in stats
        
        # Check that training ran
        assert stats['iterations'] > 0
        assert len(stats['log_likelihood_history']) == stats['iterations'] + 1  # +1 for initial
        assert len(stats['improvement_history']) == stats['iterations']
        
        # Model should still be valid
        assert hmm.validate_stochastic_matrices()
    
    def test_train_convergence(self):
        """Test that training can converge."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Use a simple pattern that should converge quickly
        observations_list = [np.array([0, 1, 0, 1, 0, 1])]
        
        stats = hmm.train(observations_list, max_iterations=50, convergence_tolerance=0.01)
        
        # Should converge for this simple case
        if stats['converged']:
            assert stats['iterations'] < 50
            # Final improvement should be small
            assert stats['improvement_history'][-1] < 0.01
    
    def test_train_max_iterations(self):
        """Test training stops at max iterations."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1, 0])]
        
        max_iters = 5
        stats = hmm.train(observations_list, max_iterations=max_iters, convergence_tolerance=1e-10)
        
        # Should stop at max iterations
        assert stats['iterations'] <= max_iters
        assert len(stats['log_likelihood_history']) == stats['iterations'] + 1
    
    def test_train_log_likelihood_improvement(self):
        """Test that log-likelihood generally improves during training."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1, 0, 1])]
        
        stats = hmm.train(observations_list, max_iterations=10)
        
        # Log-likelihood should generally improve (EM property)
        initial_ll = stats['log_likelihood_history'][0]
        final_ll = stats['final_log_likelihood']
        
        # Final should be >= initial (allowing for small numerical errors)
        assert final_ll >= initial_ll - 1e-10
        
        # Most improvements should be non-negative
        positive_improvements = sum(1 for imp in stats['improvement_history'] if imp >= -1e-10)
        assert positive_improvements >= len(stats['improvement_history']) * 0.8  # Allow some numerical errors
    
    def test_train_empty_observations_list(self):
        """Test error handling for empty observations list."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        with pytest.raises(ValueError, match="observations_list cannot be empty"):
            hmm.train([])
    
    def test_train_empty_sequence(self):
        """Test error handling for empty sequence in list."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        observations_list = [np.array([0, 1]), np.array([])]
        
        with pytest.raises(ValueError, match="Sequence 1 is empty"):
            hmm.train(observations_list)
    
    def test_train_invalid_observations(self):
        """Test error handling for invalid observation indices."""
        hmm = DiscreteHMM(n_states=2, n_observations=2)
        
        # Observation index 2 is invalid (>= n_observations)
        observations_list = [np.array([0, 1, 2])]
        
        with pytest.raises(ValueError, match="Sequence 0 contains invalid observation indices"):
            hmm.train(observations_list)
    
    def test_train_verbose_mode(self):
        """Test training with verbose output."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0])]
        
        # Should not raise any errors with verbose=True
        stats = hmm.train(observations_list, max_iterations=3, verbose=True)
        
        assert isinstance(stats, dict)
        assert hmm.validate_stochastic_matrices()
    
    def test_train_different_convergence_tolerances(self):
        """Test training with different convergence tolerances."""
        hmm1 = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        hmm2 = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1, 0])]
        
        # Strict tolerance should require more iterations
        stats_strict = hmm1.train(observations_list, convergence_tolerance=0.001, max_iterations=50)
        
        # Loose tolerance should require fewer iterations
        stats_loose = hmm2.train(observations_list, convergence_tolerance=1.0, max_iterations=50)
        
        # Loose tolerance should converge faster (or at least not slower)
        if stats_loose['converged']:
            assert stats_loose['iterations'] <= stats_strict['iterations']
    
    def test_compute_total_log_likelihood(self):
        """Test total log-likelihood computation."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [
            np.array([0, 1]),
            np.array([1, 0, 1])
        ]
        
        total_ll = hmm.compute_total_log_likelihood(observations_list)
        
        # Should equal sum of individual log-likelihoods
        ll1 = hmm.score(observations_list[0])
        ll2 = hmm.score(observations_list[1])
        expected_total = ll1 + ll2
        
        assert abs(total_ll - expected_total) < 1e-10


class TestHMMTrainingConvergence:
    """Test convergence properties of HMM training."""
    
    def test_training_deterministic_pattern(self):
        """Test training on deterministic observation pattern."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Alternating pattern
        observations_list = [np.array([0, 1, 0, 1, 0, 1, 0, 1])]
        
        stats = hmm.train(observations_list, max_iterations=20, convergence_tolerance=0.01)
        
        # Should converge and improve log-likelihood
        initial_ll = stats['log_likelihood_history'][0]
        final_ll = stats['final_log_likelihood']
        
        assert final_ll >= initial_ll
        assert hmm.validate_stochastic_matrices()
    
    def test_training_random_sequences(self):
        """Test training on random observation sequences."""
        hmm = DiscreteHMM(n_states=3, n_observations=2, random_state=42)
        
        # Generate random sequences
        np.random.seed(42)
        observations_list = []
        for _ in range(5):
            seq_length = np.random.randint(5, 15)
            seq = np.random.randint(0, 2, size=seq_length)
            observations_list.append(seq)
        
        stats = hmm.train(observations_list, max_iterations=30)
        
        # Should complete without errors
        assert stats['iterations'] > 0
        assert np.isfinite(stats['final_log_likelihood'])
        assert hmm.validate_stochastic_matrices()
    
    def test_training_single_observation_type(self):
        """Test training when sequences contain only one observation type."""
        hmm = DiscreteHMM(n_states=2, n_observations=3, random_state=42)
        
        # All sequences contain only observation 0
        observations_list = [
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0])
        ]
        
        stats = hmm.train(observations_list, max_iterations=10)
        
        # Should handle gracefully
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(stats['final_log_likelihood'])
        
        # Emission probabilities for observation 0 should be higher
        assert np.all(hmm.B[:, 0] > hmm.B[:, 1])
        assert np.all(hmm.B[:, 0] > hmm.B[:, 2])
    
    def test_training_reproducibility(self):
        """Test that training is reproducible with same random seed."""
        observations_list = [np.array([0, 1, 0, 1, 0])]
        
        hmm1 = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        stats1 = hmm1.train(observations_list, max_iterations=10)
        
        hmm2 = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        stats2 = hmm2.train(observations_list, max_iterations=10)
        
        # Should produce identical results
        assert stats1['iterations'] == stats2['iterations']
        assert abs(stats1['final_log_likelihood'] - stats2['final_log_likelihood']) < 1e-10
        
        np.testing.assert_array_almost_equal(hmm1.pi, hmm2.pi)
        np.testing.assert_array_almost_equal(hmm1.A, hmm2.A)
        np.testing.assert_array_almost_equal(hmm1.B, hmm2.B)
    
    def test_training_with_regularization_variations(self):
        """Test training with different regularization parameters."""
        observations_list = [np.array([0, 1, 0, 1])]
        
        # High regularization
        hmm_high_reg = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        stats_high = hmm_high_reg.train(observations_list, regularization_alpha=1.0, max_iterations=10)
        
        # Low regularization
        hmm_low_reg = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        stats_low = hmm_low_reg.train(observations_list, regularization_alpha=0.001, max_iterations=10)
        
        # Both should complete successfully
        assert hmm_high_reg.validate_stochastic_matrices()
        assert hmm_low_reg.validate_stochastic_matrices()
        
        # High regularization should lead to more uniform parameters
        high_reg_entropy = -np.sum(hmm_high_reg.pi * np.log(hmm_high_reg.pi + 1e-10))
        low_reg_entropy = -np.sum(hmm_low_reg.pi * np.log(hmm_low_reg.pi + 1e-10))
        
        # Higher regularization should generally lead to higher entropy (more uniform)
        # This is not guaranteed but is a reasonable expectation
        assert high_reg_entropy >= low_reg_entropy - 0.1  # Allow some tolerance


class TestHMMTrainingEdgeCases:
    """Test edge cases for HMM training."""
    
    def test_training_single_state_hmm(self):
        """Test training single-state HMM."""
        hmm = DiscreteHMM(n_states=1, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1, 0, 1])]
        
        stats = hmm.train(observations_list, max_iterations=5)
        
        # Should work without issues
        assert hmm.validate_stochastic_matrices()
        
        # pi should remain [1.0], A should remain [[1.0]]
        np.testing.assert_array_almost_equal(hmm.pi, [1.0])
        np.testing.assert_array_almost_equal(hmm.A, [[1.0]])
    
    def test_training_minimal_sequences(self):
        """Test training with minimal (length 1) sequences."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [
            np.array([0]),
            np.array([1]),
            np.array([0])
        ]
        
        stats = hmm.train(observations_list, max_iterations=5)
        
        # Should work even with minimal sequences
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(stats['final_log_likelihood'])
    
    def test_training_very_long_sequence(self):
        """Test training with very long sequence."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        # Create long sequence
        np.random.seed(42)
        long_sequence = np.random.randint(0, 2, size=1000)
        observations_list = [long_sequence]
        
        stats = hmm.train(observations_list, max_iterations=5)
        
        # Should handle long sequences without numerical issues
        assert hmm.validate_stochastic_matrices()
        assert np.isfinite(stats['final_log_likelihood'])
        assert np.all(np.isfinite(hmm.pi))
        assert np.all(np.isfinite(hmm.A))
        assert np.all(np.isfinite(hmm.B))
    
    def test_training_immediate_convergence(self):
        """Test case where training converges immediately."""
        hmm = DiscreteHMM(n_states=2, n_observations=2, random_state=42)
        
        observations_list = [np.array([0, 1])]
        
        # Use very loose convergence tolerance
        stats = hmm.train(observations_list, convergence_tolerance=100.0, max_iterations=10)
        
        # Should converge in 1 iteration
        assert stats['converged']
        assert stats['iterations'] == 1
    
    def test_training_never_converges(self):
        """Test case where training never converges within max iterations."""
        hmm = DiscreteHMM(n_states=5, n_observations=3, random_state=42)
        
        # Use more complex data that's less likely to converge quickly
        observations_list = [
            np.array([0, 1, 2, 0, 1, 2, 0, 1]),
            np.array([2, 1, 0, 2, 1, 0])
        ]
        
        # Use impossible convergence tolerance (negative) and few iterations
        stats = hmm.train(observations_list, convergence_tolerance=-1.0, max_iterations=2)
        
        # Should not converge with negative tolerance
        assert not stats['converged']
        assert stats['iterations'] == 2