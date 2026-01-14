"""
LearningPriority: A framework for posetal games and preference learning.
"""

from .orders import PartialOrder, PreOrder, minimal_elements, maximal_elements, total_order_from_list, completions_of_poset
from .order_of_priority import all_partial_orders
from .game import Metric, Player, PosetalGame, ActionProfile, best_response, game_from_preference_dict
from .nash_finder import is_pure_nash_equilibrium, find_pure_nash_equilibria, find_admissible_nash_equilibria, find_admissible_nash_equilibria_with_preferences
from .learning import PreferenceBelief, IndividualLearningAlgorithm, WeightedVotingPureNEMax, WeightedVotingPureNEProbability, LearningFramework
from .case_study_pipeline import run_case_study, generate_random_metrics, build_single_player, enumerate_preferences, plot_belief_trajectories

# __all__ = [
#     'PartialOrder',
#     'PreOrder',
#     'minimal_elements',
#     'maximal_elements',
#     'total_order_from_list',
#     'all_partial_orders',
#     'Metric',
#     'Player',
#     'PosetalGame',
#     'ActionProfile',
#     'best_response',
#     'NashEquilibriumFinder',
#     'BestResponseDynamics',
    # 'PreferenceBelief',
    # 'LearningAlgorithm',
    # 'WeightedVotingPureNE',
    # 'LearningFramework',
# ]
