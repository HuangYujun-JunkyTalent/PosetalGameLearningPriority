"""
Algorithms for finding Nash equilibria in posetal games.
"""
from typing import Dict, List, Set, Tuple, Optional
from itertools import product


from .orders import PartialOrder
from .game import PosetalGame, Player, best_response, ActionProfile, game_from_preference_dict


def find_admissible_nash_equilibria_with_preferences(
        base_game: PosetalGame,
        preference_dict: Dict[str, PartialOrder]
    ) -> Set[ActionProfile]:
    """
    Find all admissible Nash equilibria in the posetal game constructed from the base game
    and the given preference dictionary.
    """
    # Construct the game with the specified preferences
    game = game_from_preference_dict(base_game, preference_dict)
    return find_admissible_nash_equilibria(game)


def is_pure_nash_equilibrium(game: PosetalGame, action_profile: ActionProfile) -> bool:
    """
    Check if the given action profile is a pure Nash equilibrium in the posetal game.

    An action profile is a pure Nash equilibrium if no player can unilaterally
    deviate to improve their outcome according to their preference order.
    """
    for player in game.players:
        current_action = action_profile[player.player_id]
        best_responses = best_response(player, action_profile, game)
        if current_action not in best_responses:
            return False
    return True

def find_pure_nash_equilibria(game: PosetalGame) -> Set[ActionProfile]:
    """
    Find all pure Nash equilibria in the given posetal game.

    A pure Nash equilibrium is an action profile where no player can unilaterally
    deviate to improve their outcome according to their preference order.

    Returns a set of ActionProfile instances representing the pure Nash equilibria.
    """
    equilibria: Set[ActionProfile] = set()

    for action_profile in game.action_profiles:
        if is_pure_nash_equilibrium(game, action_profile):
            equilibria.add(action_profile)

    return equilibria

def find_admissible_nash_equilibria(game: PosetalGame) -> Set[ActionProfile]:
    """
    Find all admissible Nash equilibria in the given posetal game.

    An admissible Nash equilibrium is a Nash equilibrium that is not dominated
    by other NEs.
    """
    pure_nash_equilibria = find_pure_nash_equilibria(game)
    admissible_equilibria: Set[ActionProfile] = set(pure_nash_equilibria)

    for ne1, ne2 in product(pure_nash_equilibria, repeat=2):
        if ne1 != ne2 and action_profile_is_dominated(game, ne1, ne2):
            admissible_equilibria.discard(ne1)
    
    return admissible_equilibria

def action_profile_is_dominated(
    game: PosetalGame,
    ap1: ActionProfile,
    ap2: ActionProfile
) -> bool:
    """
    Check if action profile ap1 is dominated by ap2 for all players in the game.
    """
    exist_strict_leq = False
    for player in game.players:
        if not game.induced_pre_order_for_players[player.player_id].leq(ap1, ap2):
            return False
        elif game.induced_pre_order_for_players[player.player_id].less(ap1, ap2):
            exist_strict_leq = True
    
    if exist_strict_leq:
        return True
    return False
