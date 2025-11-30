"""
Unit tests for nash_finder.py
"""
import unittest

from LearningPriority import PartialOrder
from LearningPriority.game import ActionProfile, Metric, Player, PosetalGame
from LearningPriority.nash_finder import (
    is_pure_nash_equilibrium,
    find_pure_nash_equilibria,
    find_admissible_nash_equilibria,
)


def build_coordination_game() -> PosetalGame:
    """Two-player coordination game: both prefer matching actions."""
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


class TestNashFinderCoordination(unittest.TestCase):
    def setUp(self):
        self.game = build_coordination_game()
        self.ap_AA = ActionProfile({"P1": "A", "P2": "A"})
        self.ap_AB = ActionProfile({"P1": "A", "P2": "B"})
        self.ap_BA = ActionProfile({"P1": "B", "P2": "A"})
        self.ap_BB = ActionProfile({"P1": "B", "P2": "B"})

    def test_is_pure_nash_equilibrium(self):
        # Matching profiles should be NE
        try:
            self.assertTrue(is_pure_nash_equilibrium(self.game, self.ap_AA))
            self.assertTrue(is_pure_nash_equilibrium(self.game, self.ap_BB))
        except Exception as e:
            self.fail(f"NE check raised {e}; check source best_response/game setup")
        # Non-matching should not be NE
        self.assertFalse(is_pure_nash_equilibrium(self.game, self.ap_AB))
        self.assertFalse(is_pure_nash_equilibrium(self.game, self.ap_BA))

    def test_find_pure_nash_equilibria(self):
        try:
            ne_set = find_pure_nash_equilibria(self.game)
        except Exception as e:
            self.fail(f"find_pure_nash_equilibria raised {e}; likely source bug (e.g., player_id typo)")
        self.assertIn(self.ap_AA, ne_set)
        self.assertIn(self.ap_BB, ne_set)
        self.assertEqual(len(ne_set), 2)

    def test_find_admissible_nash_equilibria(self):
        try:
            admissible_set = find_admissible_nash_equilibria(self.game)
        except Exception as e:
            self.fail(f"find_admissible_nash_equilibria raised {e}; check action_profile_is_dominated")
        # In coordination game, both equilibria are admissible (none Pareto-dominates the other)
        self.assertIn(self.ap_AA, admissible_set)
        self.assertIn(self.ap_BB, admissible_set)
        self.assertEqual(len(admissible_set), 2)


def build_asymmetric_game() -> PosetalGame:
    """An asymmetric game where one equilibrium dominates the other for both players."""
    # Player 1 prefers A regardless; player 2 prefers A if P1 chooses A, else B
    def p1_metric(ap: ActionProfile) -> float:
        return 1.0 if ap["P1"] == "A" else 0.0

    def p2_metric(ap: ActionProfile) -> float:
        # Rewards when P1 is A and P2 is A, else rewards B
        if ap["P1"] == "A":
            return 1.0 if ap["P2"] == "A" else 0.5
        else:
            return 1.0 if ap["P2"] == "B" else 0.2

    m1 = Metric("M1", p1_metric)
    m2 = Metric("M2", p2_metric)
    pref1 = PartialOrder({"M1"}, {("M1", "M1")})
    pref2 = PartialOrder({"M2"}, {("M2", "M2")})
    P1 = Player("P1", actions={"A", "B"}, metrics={m1}, preference=pref1)
    P2 = Player("P2", actions={"A", "B"}, metrics={m2}, preference=pref2)
    return PosetalGame([P1, P2])


class TestNashFinderAdmissible(unittest.TestCase):
    def setUp(self):
        self.game = build_asymmetric_game()
        self.ap_AA = ActionProfile({"P1": "A", "P2": "A"})
        self.ap_AB = ActionProfile({"P1": "A", "P2": "B"})
        self.ap_BA = ActionProfile({"P1": "B", "P2": "A"})
        self.ap_BB = ActionProfile({"P1": "B", "P2": "B"})

    def test_pure_nes_exist(self):
        ne_set = find_pure_nash_equilibria(self.game)
        # At least one equilibrium should exist
        self.assertTrue(len(ne_set) >= 1)

    def test_admissible_subset_of_pure(self):
        ne_set = find_pure_nash_equilibria(self.game)
        admissible_set = find_admissible_nash_equilibria(self.game)
        # admissible is subset
        self.assertTrue(admissible_set.issubset(ne_set))


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
