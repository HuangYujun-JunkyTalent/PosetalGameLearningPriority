"""
Data structures for posetal games.
"""
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Callable, FrozenSet
from itertools import product
from copy import deepcopy

import networkx as nx

from .orders import PartialOrder, PreOrder, maximal_elements


@dataclass(frozen=True)
class ActionProfile:
    """
    Represents an action profile (tuple of actions for all players).
    """
    action_profile: Dict[str, str]  # Mapping from player_id to action

    def __getitem__(self, player_id: str) -> str:
        return self.action_profile[player_id]

    def __iter__(self):
        return iter(self.action_profile.items())
    
    def __len__(self) -> int:
        return len(self.action_profile)
    
    def __hash__(self) -> int:
        return hash(frozenset(self.action_profile.items()))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionProfile):
            return NotImplemented
        return self.action_profile == other.action_profile


class Metric:
    """
    Represents a metric (outcome/reward function) for a player.
    
    A metric maps action profiles to real-valued outcomes.
    """
    name: str
    outcome_function: Callable[[ActionProfile], float]
    
    def __init__(self, name: str, outcome_function: Callable[[ActionProfile], float]):
        """
        Initialize a metric.
        
        Args:
            name: Name/identifier for this metric
            outcome_function: Function that takes an action profile and returns a real value
        """
        self.name = name
        self.outcome_function = outcome_function
    
    def evaluate(self, action_profile: ActionProfile) -> float:
        """Evaluate the metric for a given action profile."""
        return self.outcome_function(action_profile)

class Player:
    """
    Represents a player in a posetal game.
    """
    player_id: str
    actions: FrozenSet[str]
    metrics: FrozenSet[Metric]
    preference: PartialOrder  # Partial order over metric names

    def __init__(self, 
                 player_id: str,
                 actions,
                 metrics,
                 preference: PartialOrder):
        """
        Initialize a player.
        
        Args:
            player_id: Unique identifier for the player
            actions: Set of available actions
            metrics: Set of metrics the player cares about
            preference: Partial order over the metrics (priority)
        """
        self.player_id = player_id
        self.actions = frozenset(actions) if not isinstance(actions, frozenset) else actions
        self.metrics = frozenset(metrics) if not isinstance(metrics, frozenset) else metrics
        
        # Validate that preference is over the metric names
        metric_names = {m.name for m in self.metrics}
        if set(preference.elements) != metric_names:
            raise ValueError("Preference must be defined over metric names")
        
        self.preference = preference

class PosetalGame:
    """
    Represents a posetal game with multiple players.
    """
    player_ids: List[str]
    players: List[Player]
    player_dict: Dict[str, Player]

    action_profiles: List[ActionProfile]
    induced_pre_order_for_players: Dict[str, PreOrder] # induced pre-order for each player over action profiles
    
    def __init__(self, players: List[Player]):
        """
        Initialize a posetal game.
        
        Args:
            players: List of players in the game
        """
        self.players = players
        self.player_ids = [p.player_id for p in players]
        self.player_dict = {p.player_id: p for p in players}
        
        # Compute all possible action profiles
        self.action_profiles = []
        for actions_tuple in product(*[p.actions for p in players]):
            action_profile = {p.player_id: action for p, action in zip(players, actions_tuple)}
            self.action_profiles.append(ActionProfile(action_profile))
        
        self.induced_pre_order_for_players = {}
        for player in players:
            self.induced_pre_order_for_players[player.player_id] = self.induced_preorder_action_profiles(player.player_id)
    
    def get_player_by_id(self, player_id: str) -> Player:
        """Get a player by ID."""
        return self.player_dict[player_id]
    
    def evaluate_metrics(self, player_id: str, action_profile: ActionProfile) -> Dict[str, float]:
        """
        Evaluate all metrics for a player at a given action profile.
        
        Returns:
            Dictionary mapping metric names to their values
        """
        player = self.get_player_by_id(player_id)
        return {
            metric.name: metric.evaluate(action_profile)
            for metric in player.metrics
        }
    
    def induced_preorder_action_profiles(self, player_id: str) -> PreOrder:
        """
        Compute the induced pre-order over action profiles for a player.
        
        Based on Definition 2 in the write-up.
        """
        player = self.get_player_by_id(player_id)
        
        # Create element names from action profiles
        elements = set(self.action_profiles)
        
        relations = set()
        
        for ap1, ap2 in product(self.action_profiles, repeat=2):
            if _check_induced_leq(player, ap1, ap2):
                relations.add((ap1, ap2))
        
        return PreOrder(elements, relations)
    
    def induced_preorder_actions_of_player(self, player_id: str, other_players_actions: ActionProfile) -> PreOrder:
        """
        Compute the induced pre-order over the actions of a player given fixed actions of other players.
        
        Args:
            player_id: ID of the player whose actions are being ordered
            other_players_actions: ActionProfile specifying actions of other players. Allowed to have this player's action missing or arbitrary.
        Returns:
            PreOrder over the ActionProfile, allowing only the specified player's actions to vary.
        """
        all_otherplayer_ids = [pid for pid in self.player_ids if pid != player_id]
        all_allowed_action_profiles = set([])

        action_dict_of_other_players = {pid: other_players_actions[pid] for pid in all_otherplayer_ids}
        for action in self.get_player_by_id(player_id).actions:
            this_action_profile_dict = deepcopy(action_dict_of_other_players)
            this_action_profile_dict[player_id] = action
            all_allowed_action_profiles.add(ActionProfile(this_action_profile_dict))
        
        return self.induced_preorder_action_profiles(player_id).build_sub_preorder(all_allowed_action_profiles)


def game_from_preference_dict(base_game: PosetalGame, preference_dict: Dict[str, PartialOrder]) -> PosetalGame:
    """
    Create a new PosetalGame from a base game and a dictionary of preferences for each player.
    
    Args:
        base_game: The original PosetalGame
        preference_dict: Dictionary mapping player IDs to their new PartialOrder preferences
    Returns:
        A new PosetalGame with updated player preferences
    """
    new_players = []
    for player in base_game.players:
        if player.player_id not in preference_dict:
            print(f"Warning: Player {player.player_id} not in preference_dict; using original preference.")
            new_players.append(player)
        else:
            new_pref = preference_dict[player.player_id]
            new_player = Player(
                player_id=player.player_id,
                actions=player.actions,
                metrics=player.metrics,
                preference=new_pref
            )
            new_players.append(new_player)
    return PosetalGame(new_players)

def best_response(player: Player, other_players_actions: ActionProfile, game: PosetalGame) -> Set[str]:
    """
    Compute the best response actions for a player given other players' actions.
    
    Args:
        player: The player for whom to compute best responses
        other_players_actions: ActionProfile specifying actions of other players
        game: The PosetalGame instance
    Returns:
        Set of best response actions for the player
    """
    induced_preorder_over_action_profiles = game.induced_preorder_actions_of_player(player.player_id, other_players_actions)
    # Find maximal elements in the induced pre-order
    maximal_aps = maximal_elements(induced_preorder_over_action_profiles)
    # Extract the actions of the player from the maximal action profiles
    best_responses = set()
    for ap in maximal_aps:
        best_responses.add(ap[player.player_id])
    return best_responses

def _check_induced_leq(player: Player, ap1: ActionProfile, ap2: ActionProfile) -> bool:
    """
    Check if action profile ap1 <= ap2 according to the induced pre-order.
    """
    # Build a DiGraph representing metric comparisons
    LESS = 'LESS'
    EQUAL = 'EQUAL'
    GREATER = 'GREATER'
    metric_comparison_results: Dict[str, str] = {}
    for metric in player.metrics:
        val1 = metric.evaluate(ap1)
        val2 = metric.evaluate(ap2)
        if val1 < val2:
            metric_comparison_results[metric.name] = LESS
        elif val1 > val2:
            metric_comparison_results[metric.name] = GREATER
        else:
            metric_comparison_results[metric.name] = EQUAL
    G = nx.DiGraph()
    # Add nodes, with the comparison results as attributes
    for metric_name, comparison_result in metric_comparison_results.items():
        G.add_node(metric_name, comparison_result=comparison_result)
    # Add edges according to player's preference
    for a, b in product(player.preference.elements, repeat=2):
        if player.preference.less(a, b):
            G.add_edge(a, b)

    # Now check if there is a path from any GREATER node to any LESS node
    greater_nodes = set(n for n, attr in G.nodes(data=True) if attr['comparison_result'] == GREATER)
    less_nodes = set(n for n, attr in G.nodes(data=True) if attr['comparison_result'] == LESS)
    ancestors_of_less = set()
    for ln in less_nodes:
        ancestors_of_less.update(nx.ancestors(G, ln))
    # If any greater node is an ancestor of one of the less nodes, then ap1 <= ap2 holds
    if greater_nodes.issubset(ancestors_of_less):
        return True
    return False
