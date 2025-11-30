"""
Unit tests for learning.py
"""
import unittest
from typing import Dict, Set

from LearningPriority.orders import PartialOrder
from LearningPriority.game import ActionProfile, Metric, Player, PosetalGame
from LearningPriority.learning import (
    PreferenceBelief,
    update_belief,
    WeightedVotingPureNEProbability,
    WeightedVotingPureNEMax,
    LearningFramework,
)
from LearningPriority.nash_finder import find_admissible_nash_equilibria_with_preferences


def ne_finder_wrapper(game: PosetalGame, pref_dict: Dict[str, PartialOrder]) -> Set[ActionProfile]:
    """Wrapper to avoid lru_cache type issues in tests."""
    return find_admissible_nash_equilibria_with_preferences(game, pref_dict)


class TestPreferenceBelief(unittest.TestCase):
    def test_initialization_and_normalization(self):
        """Test that PreferenceBelief normalizes properly."""
        pref1 = PartialOrder({"M1"}, {("M1", "M1")})
        pref2 = PartialOrder({"M1"}, {("M1", "M1")})
        
        # Non-normalized but valid probabilities (each in [0,1])
        belief = PreferenceBelief({pref1: 0.5, pref2: 0.5})
        
        # Should be normalized (already sums to 1)
        self.assertAlmostEqual(sum(belief.belief.values()), 1.0)
    
    def test_sample(self):
        """Test sampling from belief distribution."""
        pref1 = PartialOrder({"M1"}, {("M1", "M1")})
        belief = PreferenceBelief({pref1: 1.0})
        
        sampled = belief.sample()
        self.assertEqual(sampled, pref1)
    
    def test_most_likely(self):
        """Test getting most likely preference."""
        pref1 = PartialOrder({"M1"}, {("M1", "M1")})
        pref2 = PartialOrder({"M1"}, {("M1", "M1")})
        
        belief = PreferenceBelief({pref1: 0.7, pref2: 0.3})
        self.assertEqual(belief.most_likely(), pref1)


class TestUpdateBelief(unittest.TestCase):
    def test_bayesian_update(self):
        """Test Bayesian belief update."""
        pref1 = PartialOrder({"M1"}, {("M1", "M1")})
        pref2 = PartialOrder({"M1"}, {("M1", "M1")})
        
        prior = PreferenceBelief({pref1: 0.5, pref2: 0.5})
        
        # Observation strongly supports pref1
        likelihoods = {pref1: 0.9, pref2: 0.1}
        
        posterior = update_belief(prior, likelihoods)
        
        # Posterior should favor pref1
        self.assertGreater(posterior.belief[pref1], posterior.belief[pref2])
        self.assertAlmostEqual(sum(posterior.belief.values()), 1.0)


def build_test_game() -> PosetalGame:
    """Build a simple coordination game for testing."""
    def p1_payoff(ap: ActionProfile) -> float:
        return 1.0 if ap["P1"] == ap["P2"] else 0.0
    
    def p2_payoff(ap: ActionProfile) -> float:
        return 1.0 if ap["P1"] == ap["P2"] else 0.0
    
    m1 = Metric("Payoff", p1_payoff)
    m2 = Metric("Payoff", p2_payoff)
    
    pref = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
    P1 = Player("P1", actions={"A", "B"}, metrics={m1}, preference=pref)
    P2 = Player("P2", actions={"A", "B"}, metrics={m2}, preference=pref)
    
    return PosetalGame([P1, P2])


class TestWeightedVotingAlgorithms(unittest.TestCase):
    def setUp(self):
        self.game = build_test_game()
        self.player = self.game.players[0]
        
        # Create simple belief over one preference
        pref = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
        self.beliefs = {
            "P2": PreferenceBelief({pref: 1.0})
        }
    
    def test_weighted_voting_probability(self):
        """Test WeightedVotingPureNEProbability algorithm."""
        algo = WeightedVotingPureNEProbability()
        
        action_dist = algo.compute_action_distribution(
            my_player_id=self.player.player_id,
            my_preference=self.player.preference,
            beliefs=self.beliefs,
            base_game=self.game,
            ne_finder=ne_finder_wrapper,
        )
        
        # Should return a valid distribution
        self.assertAlmostEqual(sum(action_dist.values()), 1.0)
        self.assertTrue(all(p >= 0 for p in action_dist.values()))
        
        # Both A and B should have positive probability (both are NE)
        self.assertGreater(action_dist.get("A", 0), 0)
        self.assertGreater(action_dist.get("B", 0), 0)
    
    def test_weighted_voting_max(self):
        """Test WeightedVotingPureNEMax algorithm."""
        algo = WeightedVotingPureNEMax()
        
        action_dist = algo.compute_action_distribution(
            my_player_id=self.player.player_id,
            my_preference=self.player.preference,
            beliefs=self.beliefs,
            base_game=self.game,
            ne_finder=ne_finder_wrapper,
        )
        
        # Should return a valid distribution
        self.assertAlmostEqual(sum(action_dist.values()), 1.0)
        self.assertTrue(all(p >= 0 for p in action_dist.values()))


class TestLearningFramework(unittest.TestCase):
    def test_initialization(self):
        """Test LearningFramework initialization."""
        game = build_test_game()
        
        pref = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
        prior = {
            "P1": PreferenceBelief({pref: 1.0}),
            "P2": PreferenceBelief({pref: 1.0}),
        }
        algorithms = {
            "P1": WeightedVotingPureNEProbability(),
            "P2": WeightedVotingPureNEProbability(),
        }
        framework = LearningFramework(game, prior, algorithms)
        
        # Check beliefs initialized
        self.assertEqual(len(framework.current_beliefs()), 2)
        for belief in framework.current_beliefs().values():
            self.assertIsInstance(belief, PreferenceBelief)
    
    def test_run_iteration(self):
        """Test running one iteration."""
        game = build_test_game()
        
        pref = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
        prior = {
            "P1": PreferenceBelief({pref: 1.0}),
            "P2": PreferenceBelief({pref: 1.0}),
        }
        algorithms = {
            "P1": WeightedVotingPureNEProbability(),
            "P2": WeightedVotingPureNEProbability(),
        }
        framework = LearningFramework(game, prior, algorithms)
        
        # Run one iteration
        action_profile = framework.run_iteration()
        
        # Should return an ActionProfile
        self.assertIsInstance(action_profile, ActionProfile)
        self.assertEqual(len(action_profile), 2)
    
    def test_simulate(self):
        """Test simulating multiple iterations."""
        game = build_test_game()
        
        pref = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
        prior = {
            "P1": PreferenceBelief({pref: 1.0}),
            "P2": PreferenceBelief({pref: 1.0}),
        }
        algorithms = {
            "P1": WeightedVotingPureNEProbability(),
            "P2": WeightedVotingPureNEProbability(),
        }
        framework = LearningFramework(game, prior, algorithms)
        
        # Simulate 10 iterations
        trajectory = framework.simulate(num_iterations=10)
        
        # Should return list of action profiles
        self.assertEqual(len(trajectory), 10)
        for ap in trajectory:
            self.assertIsInstance(ap, ActionProfile)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
