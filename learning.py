"""
Learning framework for preference learning in posetal games.
"""
from copy import deepcopy
import random
import warnings
from typing import Dict, List, Set, Tuple, Optional, Callable
from itertools import product


from .game import PosetalGame, Player, ActionProfile, best_response
from .orders import PreOrder, PartialOrder
from .nash_finder import find_admissible_nash_equilibria_with_preferences


class PreferenceBelief:
    """
    Represents a belief distribution over possible preferences for a player.
    """
    belief: Dict[PartialOrder, float]
    
    def __init__(self, prior_belief: Dict[PartialOrder, float]):
        """
        Initialize a uniform belief over possible preferences.
        
        Args:
            possible_preferences: Set of possible partial orders over metrics
        """
        self.belief = prior_belief

        assert self.belief, "Belief cannot be empty"
        for prob in self.belief.values():
            assert 0.0 <= prob <= 1.0, "Probabilities must be in [0, 1]"
        total = sum(self.belief.values())
        assert total > 0, "Total probability must be positive"

        # Normalize
        if total != 1.0:
            warnings.warn("Normalizing prior belief")
            for pref in self.belief:
                self.belief[pref] /= total
    
    def sample(self) -> PartialOrder:
        """Sample a preference from the belief distribution."""
        prefs = list(self.belief.keys())
        probs = [self.belief[p] for p in prefs]
        return random.choices(prefs, weights=probs, k=1)[0]
    
    def most_likely(self) -> PartialOrder:
        """Return the most likely preference."""
        return max(self.belief.items(), key=lambda item: item[1])[0]

    def __hash__(self):
        return hash(tuple(sorted(
            (pref, prob)
            for pref, prob in self.belief.items()
        )))
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PreferenceBelief):
            return False
        return self.belief == value.belief
    
    def keys(self) -> Set[PartialOrder]:
        """Return the set of preferences in the belief."""
        return set(self.belief.keys())

def update_belief(
    prior: PreferenceBelief,
    likelihoods: Dict[PartialOrder, float]
) -> PreferenceBelief:
    """
    Update the belief based on observed action likelihoods.
    
    Args:
        prior: Prior belief over preferences
        likelihoods: Likelihood of observed action given each preference
    
    Returns:
        Updated PreferenceBelief
    """
    updated_belief = {}
    
    for pref, prior_prob in prior.belief.items():
        try:
            likelihood = likelihoods[pref]
        except KeyError:
            warnings.warn(f"Preference {pref} not in likelihoods; assuming zero likelihood")
            likelihood = 0.0
        updated_belief[pref] = prior_prob * likelihood
    
    # Normalize
    total = sum(updated_belief.values())
    if total == 0:
        raise ValueError("Total probability after update is zero; check likelihoods")
    
    for pref in updated_belief:
        updated_belief[pref] /= total
    
    return PreferenceBelief(updated_belief)

class IndividualLearningAlgorithm:
    """
    Base class for learning algorithms in posetal games.
    """
    
    def compute_action_distribution(
            self,
            my_player_id: str,
            my_preference: PartialOrder,
            beliefs: Dict[str, PreferenceBelief],
            base_game: PosetalGame,
            ne_finder: Callable[[PosetalGame, Dict[str, PartialOrder]], Set[ActionProfile]],
        ) -> Dict[str, float]:
        """
        Compute a distribution over actions given beliefs about others' preferences.
        
        Args:
            beliefs: Dictionary mapping other player IDs to their preference beliefs
            ne_finder: Function to find Nash equilibria given a game and beliefs
        Returns:
            Dictionary mapping actions to probabilities
        """
        raise NotImplementedError


class WeightedVotingPureNEProbability(IndividualLearningAlgorithm):
    """
    Weighted voting algorithm based on pure Nash equilibria (Equation (1) in Section 3.2.1).
    
    For each action a_i of player i, computes weight as the sum over all preference profiles
    of other players of: P(preference_profile) * indicator(a_i in admissible NE).
    """
    def compute_action_distribution(
            self, 
            my_player_id: str,
            my_preference: PartialOrder,
            beliefs: Dict[str, PreferenceBelief],
            base_game: PosetalGame,
            ne_finder: Callable[[PosetalGame, Dict[str, PartialOrder]], Set[ActionProfile]],
        ) -> Dict[str, float]:
        """
        Compute action distribution via weighted voting (Eq. 1 in write-up).
        
        π_i(t)(a_i) ∝ Σ_{P_{-i}} μ_{-i}(t-1)(P_{-i}) × 𝟙{a_i ∈ AdmissibleNE(M_i, P_i^true, P_{-i})}
        """
        my_actions = base_game.get_player_by_id(my_player_id).actions
        action_weights = {action: 0.0 for action in my_actions}
        
        # Get all other player IDs
        other_player_ids = [pid for pid in beliefs.keys() if pid != my_player_id]
        
        if not other_player_ids:
            # Single player case: uniform distribution
            uniform_prob = 1.0 / len(my_actions)
            return {action: uniform_prob for action in my_actions}
            
        # Get all preference combinations
        other_prefs_lists = [list(beliefs[pid].belief.keys()) for pid in other_player_ids]
        
        for other_prefs_tuple in product(*other_prefs_lists):
            # Build preference dict for this combination
            pref_dict = {pid: pref for pid, pref in zip(other_player_ids, other_prefs_tuple)}
            pref_dict[my_player_id] = my_preference  # Add our true preference
            
            # Compute joint probability of this preference profile
            joint_prob = 1.0
            for pid, pref in zip(other_player_ids, other_prefs_tuple):
                joint_prob *= beliefs[pid].belief[pref]
            
            # Find admissible NEs for this preference profile
            admissible_nes = ne_finder(base_game, pref_dict)
            
            # For each action, check if it appears in any admissible NE
            for action in my_actions:
                action_in_ne = any(
                    ap[my_player_id] == action 
                    for ap in admissible_nes
                )
                if action_in_ne:
                    action_weights[action] += joint_prob
        
        # Normalize to get distribution
        total_weight = sum(action_weights.values())
        
        if total_weight == 0:
            # No admissible NEs found; return uniform distribution
            warnings.warn("No admissible NEs found; returning uniform distribution.")
            uniform_prob = 1.0 / len(my_actions)
            return {action: uniform_prob for action in my_actions}
        
        return {action: weight / total_weight for action, weight in action_weights.items()}


class WeightedVotingPureNEMax(IndividualLearningAlgorithm):
    """
    Weighted voting algorithm based on pure Nash equilibria (Equation (1) or (2) in Section 3.2.1).
    
    For each action a_i of player i, computes weight as the maximum over all preference profiles
    of other players of: P(preference_profile) * indicator(a_i in admissible NE).
    """
    def compute_action_distribution(
            self, 
            my_player_id: str,
            my_preference: PartialOrder,
            beliefs: Dict[str, PreferenceBelief],
            base_game: PosetalGame,
            ne_finder: Callable[[PosetalGame, Dict[str, PartialOrder]], Set[ActionProfile]],
        ) -> Dict[str, float]:
        """
        Compute action distribution via max weighted voting (Eq. 2 in write-up).
        
        π_i(t)(a_i) ∝ max_{P_{-i}} μ_{-i}(t-1)(P_{-i}) × 𝟙{a_i ∈ AdmissibleNE(M_i, P_i^true, P_{-i})}
        """
        my_actions = base_game.get_player_by_id(my_player_id).actions
        action_max_weights = {action: 0.0 for action in my_actions}
        
        # Get all other player IDs
        other_player_ids = [pid for pid in beliefs.keys() if pid != my_player_id]
        
        if not other_player_ids:
            # Single player case: uniform distribution
            uniform_prob = 1.0 / len(my_actions)
            return {action: uniform_prob for action in my_actions}
        
        # Get all preference combinations
        other_prefs_lists = [list(beliefs[pid].belief.keys()) for pid in other_player_ids]
        
        for other_prefs_tuple in product(*other_prefs_lists):
            # Build preference dict for this combination
            pref_dict = {pid: pref for pid, pref in zip(other_player_ids, other_prefs_tuple)}
            pref_dict[my_player_id] = my_preference  # Add our true preference
            
            # Compute joint probability of this preference profile
            joint_prob = 1.0
            for pid, pref in zip(other_player_ids, other_prefs_tuple):
                joint_prob *= beliefs[pid].belief[pref]
            
            # Find admissible NEs for this preference profile
            admissible_nes = ne_finder(base_game, pref_dict)
            
            # For each action, check if it appears in any admissible NE and update max
            for action in my_actions:
                action_in_ne = any(
                    ap[my_player_id] == action 
                    for ap in admissible_nes
                )
                if action_in_ne:
                    action_max_weights[action] = max(action_max_weights[action], joint_prob)
        
        # Normalize to get distribution
        total_weight = sum(action_max_weights.values())
        
        if total_weight == 0:
            # No admissible NEs found; return uniform distribution
            warnings.warn("No admissible NEs found; returning uniform distribution.")
            uniform_prob = 1.0 / len(my_actions)
            return {action: uniform_prob for action in my_actions}
        
        return {action: weight / total_weight for action, weight in action_max_weights.items()}


class PreferenceProfile:
    """
    Represents a preference profile for all players.
    """
    preferences: Dict[str, PartialOrder]
    
    def __init__(self, preferences: Dict[str, PartialOrder]):
        self.preferences = preferences
    
    def __hash__(self):
        return hash(tuple(sorted(
            (pid, pref)
            for pid, pref in self.preferences.items()
        )))

    def __eq__(self, other):
        if not isinstance(other, PreferenceProfile):
            return False
        return self.preferences == other.preferences

class LearningFramework:
    """
    Framework for simulating learning in posetal games.
    """
    true_game: PosetalGame # The posetal game with true preferences
    base_game: PosetalGame # The original game with trivial preferences
    prior_beliefs: Dict[str, PreferenceBelief]
    algorithms: Dict[str, IndividualLearningAlgorithm]

    belief_history: List[Dict[str, PreferenceBelief]]
    action_history: List[ActionProfile]

    _ne_cache: Dict[PreferenceProfile, Set[ActionProfile]]
    
    def __init__(
            self, 
            true_game: PosetalGame,
            prior_beliefs: Dict[str, PreferenceBelief],
            algorithms: Dict[str, IndividualLearningAlgorithm]
    ):
        """
        Initialize the learning framework.
        
        Args:
            true_game: The posetal game with true preferences
            prior_beliefs: Initial beliefs for each player
        """
        self.true_game = true_game
        self.prior_beliefs = prior_beliefs
        self.algorithms = algorithms

        self.base_game = PosetalGame([
            Player(
                player.player_id,
                actions=player.actions,
                metrics=player.metrics,
                preference=PartialOrder(
                    {m.name for m in player.metrics},
                    {(m.name, m.name) for m in player.metrics}
                )
            )
            for player in true_game.players
        ])

        self.belief_history = [prior_beliefs]
        self.action_history = []
        self._ne_cache = {}
    
    def current_beliefs(self) -> Dict[str, PreferenceBelief]:
        """Return the latest beliefs for all players."""
        return self.belief_history[-1]
    
    def get_action_distribution(
            self,
            player_id: str,
            proposed_preference: PartialOrder,
            other_beliefs: Dict[str, PreferenceBelief]) -> Dict[str, float]:
        """Get action distribution for a player given proposed preference and others' beliefs."""
        action_dist = self.algorithms[player_id].compute_action_distribution(
            player_id,
            proposed_preference,
            other_beliefs,
            self.base_game,
            self._ne_finder,
        )
        return action_dist

    def run_iteration(self) -> ActionProfile:
        """
        Run one iteration: each player chooses an action according to their algorithm
        using current beliefs; then update beliefs based on observed actions.
        Returns the played action profile as a tuple ordered by self.true_game.players.
        """
        beliefs = self.current_beliefs()
        chosen_actions: Dict[str, str] = {}

        for player in self.true_game.players: # here for each player, they know their own true preference
            action_dist = self.get_action_distribution(
                player.player_id,
                self.true_game.get_player_by_id(player.player_id).preference,
                {pid: belief for pid, belief in beliefs.items() if pid != player.player_id}
            )
            actions = list(action_dist.keys())
            probs = [action_dist[a] for a in actions]
            chosen_action = random.choices(actions, weights=probs, k=1)[0]
            chosen_actions[player.player_id] = chosen_action

        # Build ActionProfile and update beliefs
        ap = ActionProfile(chosen_actions)
        # Update beliefs based on observed actions
        new_beliefs = self._update_beliefs_with_profile_and_distributions(ap)
        self.belief_history.append(new_beliefs)

        self.action_history.append(ap)

        # Return tuple of actions in players order
        return ap

    def _update_beliefs_with_profile_and_distributions(self, ap: ActionProfile) -> Dict[str, PreferenceBelief]:
        """
        Update beliefs given observed action profile ap and action distributions.
        For any player, compute likelihoods of her choosen this specific action under each possible preference, and update via Bayes' rule.
        """
        current_beliefs = self.current_beliefs()
        new_beliefs = deepcopy(current_beliefs)
        for player in self.true_game.players:
            pid = player.player_id
            prior_belief = current_beliefs[pid]
            likelihoods: Dict[PartialOrder, float] = {}
            
            for pref in prior_belief.keys():
                # Build Belief dict for other players
                other_beliefs = {other_pid: current_beliefs[other_pid]
                                 for other_pid in current_beliefs.keys() if other_pid != pid}
                # Get action distribution for this player under this preference
                action_dist = self.algorithms[pid].compute_action_distribution(
                    pid,
                    pref,
                    other_beliefs,
                    self.base_game,
                    self._ne_finder,
                )
                chosen_action = ap[pid]
                try:
                    likelihoods[pref] = action_dist[chosen_action]
                except KeyError:
                    warnings.warn(f"Action {chosen_action} not in action distribution for player {pid} under preference {pref}; assuming zero likelihood")
                    likelihoods[pref] = 0.0

            # Update belief via Bayes' rule
            updated_belief = update_belief(prior_belief, likelihoods)
            # Update belief history
            new_beliefs[pid] = updated_belief
        
        return new_beliefs

    def simulate(self, num_iterations: int = 100) -> List[ActionProfile]:
        """Run multiple iterations and return the action trajectory."""
        traj: List[ActionProfile] = []
        for _ in range(num_iterations):
            traj.append(self.run_iteration())
        return traj

    def get_converged_beliefs(self) -> Dict[str, PartialOrder]:
        """Return most likely preference per player from latest beliefs."""
        return {pid: belief.most_likely() for pid, belief in self.current_beliefs().items()}
    
     # NE finder with manual caching using the hashable PreferenceProfile key
    def _ne_finder(self, base_game: PosetalGame, prefs: Dict[str, PartialOrder]) -> Set[ActionProfile]:
        key = PreferenceProfile(prefs)
        cached = self._ne_cache.get(key)
        if cached is not None:
            return cached
        # Finder is expected to be non-cached now; call directly
        result = find_admissible_nash_equilibria_with_preferences(base_game, prefs)
        self._ne_cache[key] = result
        return result

