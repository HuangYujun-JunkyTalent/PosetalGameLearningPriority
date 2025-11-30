import random
from typing import cast

from LearningPriority.learning import (
    PreferenceBelief,
    PreferenceProfile,
    LearningFramework,
    WeightedVotingPureNEProbability,
)
from LearningPriority.game import PosetalGame, Player, Metric, ActionProfile
from LearningPriority.orders import PartialOrder
from LearningPriority.nash_finder import find_admissible_nash_equilibria_with_preferences


def make_coordination_game():
    # Two players, actions A/B, single metric 'u' that rewards matching
    metric = Metric('u', lambda ap: 1 if ap['P1'] == ap['P2'] else 0)
    p1_pref = PartialOrder({'u'}, {('u','u')})
    p2_pref = PartialOrder({'u'}, {('u','u')})
    p1 = Player('P1', actions={'A','B'}, metrics=[metric], preference=p1_pref)
    p2 = Player('P2', actions={'A','B'}, metrics=[metric], preference=p2_pref)
    return PosetalGame([p1, p2])


def test_preference_profile_hash_and_equality():
    po1 = PartialOrder({'u'}, {('u','u')})
    po2 = PartialOrder({'u'}, {('u','u')})
    pp1 = PreferenceProfile({'P1': po1, 'P2': po2})
    pp2 = PreferenceProfile({'P2': po2, 'P1': po1})
    assert pp1 == pp2
    s = {pp1}
    assert pp2 in s  # hash must be consistent with equality


def test_learning_iteration_valid_actions():
    game = make_coordination_game()
    prior = {
        'P1': PreferenceBelief({game.players[0].preference: 1.0}),
        'P2': PreferenceBelief({game.players[1].preference: 1.0}),
    }
    algorithms = {
        'P1': WeightedVotingPureNEProbability(),
        'P2': WeightedVotingPureNEProbability(),
    }
    lf = LearningFramework(game, prior, cast(dict, algorithms))
    ap = lf.run_iteration()
    assert isinstance(ap, ActionProfile)
    assert len(ap) == len(game.players)
    for player in game.players:
        assert ap[player.player_id] in player.actions


def test_learning_framework_caches_ne_results_and_updates_beliefs():
    random.seed(0)
    game = make_coordination_game()
    prior = {
        'P1': PreferenceBelief({game.players[0].preference: 1.0}),
        'P2': PreferenceBelief({game.players[1].preference: 1.0}),
    }
    algorithms = {
        'P1': WeightedVotingPureNEProbability(),
        'P2': WeightedVotingPureNEProbability(),
    }
    lf = LearningFramework(game, prior, cast(dict, algorithms))

    # Run one iteration; with coordination game, actions should be consistent
    actions = lf.run_iteration()
    assert len(actions) == 2
    # Cache should have at least one entry
    assert len(lf._ne_cache) >= 1

    # Run a few iterations to ensure beliefs history grows and remains valid
    traj = lf.simulate(5)
    assert len(traj) == 5
    assert len(lf.belief_history) == 1 + 1 + 5  # prior + first run + 5 simulate
    # Converged beliefs should match true preferences in this simple setup
    converged = lf.get_converged_beliefs()
    assert converged['P1'] == game.players[0].preference
    assert converged['P2'] == game.players[1].preference
