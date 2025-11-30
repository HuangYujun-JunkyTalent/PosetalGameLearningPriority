"""
Unit tests for game.py: ActionProfile, Metric, Player, PosetalGame, best_response, and induced preorder functions.
"""
import unittest
from typing import Dict

from LearningPriority import PartialOrder
from LearningPriority.game import ActionProfile, Metric, Player, PosetalGame, best_response


class TestActionProfile(unittest.TestCase):
    def test_basic_behaviour(self):
        ap = ActionProfile({"P1": "A", "P2": "B"})
        self.assertEqual(ap["P1"], "A")
        self.assertEqual(len(ap), 2)
        # hashable and equality
        ap2 = ActionProfile({"P2": "B", "P1": "A"})
        self.assertEqual(ap, ap2)
        s = {ap}
        self.assertIn(ap2, s)


class TestMetric(unittest.TestCase):
    def test_evaluate(self):
        def payoff(ap: ActionProfile) -> float:
            return 1.0 if ap["P1"] == "A" else 0.0
        m = Metric("Payoff", payoff)
        apA = ActionProfile({"P1": "A"})
        apB = ActionProfile({"P1": "B"})
        self.assertEqual(m.evaluate(apA), 1.0)
        self.assertEqual(m.evaluate(apB), 0.0)


class TestPlayer(unittest.TestCase):
    def test_preference_over_metric_names(self):
        # Define two metrics
        def m1(ap: ActionProfile) -> float:
            return 1.0
        def m2(ap: ActionProfile) -> float:
            return 0.5
        M1 = Metric("M1", m1)
        M2 = Metric("M2", m2)
        # Preference over metric names
        pref = PartialOrder({"M1", "M2"}, {("M1", "M1"), ("M2", "M2"), ("M1", "M2")})
        # Construct player
        p = Player("P1", actions={"A", "B"}, metrics={M1, M2}, preference=pref)
        self.assertEqual(p.player_id, "P1")
        self.assertEqual(p.actions, frozenset({"A", "B"}))
        # preference must be defined over metric names
        with self.assertRaises(ValueError):
            wrong_pref = PartialOrder({"X", "Y"}, {("X", "X"), ("Y", "Y")})
            Player("P2", actions={"A"}, metrics={M1, M2}, preference=wrong_pref)


class TestPosetalGame(unittest.TestCase):
    def _build_simple_game(self) -> PosetalGame:
        # Two players, each with two actions
        def p1_payoff(ap: ActionProfile) -> float:
            # prefers matching actions
            return 1.0 if ap["P1"] == ap["P2"] else 0.0
        def p2_payoff(ap: ActionProfile) -> float:
            return 1.0 if ap["P1"] == ap["P2"] else 0.0
        M1 = Metric("Payoff", p1_payoff)
        M2 = Metric("Payoff", p2_payoff)
        pref1 = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
        pref2 = PartialOrder({"Payoff"}, {("Payoff", "Payoff")})
        P1 = Player("P1", actions={"A", "B"}, metrics={M1}, preference=pref1)
        P2 = Player("P2", actions={"A", "B"}, metrics={M2}, preference=pref2)
        return PosetalGame([P1, P2])

    def test_construction_and_action_profiles(self):
        game = self._build_simple_game()
        # player ids
        self.assertEqual(set(game.player_ids), {"P1", "P2"})
        # all action profiles count = 4
        self.assertEqual(len(game.action_profiles), 4)
        # each is ActionProfile
        for ap in game.action_profiles:
            self.assertIsInstance(ap, ActionProfile)

    def test_induced_preorder_action_profiles(self):
        game = self._build_simple_game()
        # Should return a PreOrder over ActionProfile objects
        po = game.induced_preorder_action_profiles("P1")
        # Elements are the action profiles
        self.assertEqual(set(po.elements), set(game.action_profiles))
        # relation reflexivity
        for ap in game.action_profiles:
            self.assertIn((ap, ap), po.relations)

    def test_induced_preorder_actions_of_player(self):
        game = self._build_simple_game()
        # Fix P2's action and vary P1's
        other_ap = ActionProfile({"P2": "A", "P1": "B"})  # P1 value is ignored by method per docs
        pre = game.induced_preorder_actions_of_player("P1", other_ap)
        # Elements should be exactly two action profiles with P2=A and P1 in {A,B}
        elements = pre.elements
        self.assertEqual(len(elements), 2)
        self.assertTrue(any(ap["P1"] == "A" and ap["P2"] == "A" for ap in elements))
        self.assertTrue(any(ap["P1"] == "B" and ap["P2"] == "A" for ap in elements))

    def test_best_response(self):
        game = self._build_simple_game()
        # When P2 plays A, P1's best responses should include A (matching)
        other_ap = ActionProfile({"P2": "A", "P1": "B"})
        p1 = game.get_player_by_id("P1")
        # NOTE: source bug: best_response uses player.palyer_id typo; this will raise KeyError.
        # Once fixed to player.player_id, expected behaviour below should pass.
        try:
            br = best_response(p1, other_ap, game)
        except Exception as e:
            self.fail(f"best_response raised {e}; likely a bug in source (palyer_id vs player_id)")
        self.assertIn("A", br)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
