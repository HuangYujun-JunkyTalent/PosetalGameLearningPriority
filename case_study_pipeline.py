import random
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict

from LearningPriority.game import PosetalGame, Player, Metric, ActionProfile
from LearningPriority.orders import PartialOrder
from LearningPriority.learning import (
    PreferenceBelief,
    WeightedVotingPureNEProbability,
    WeightedVotingPureNEMax,
    LearningFramework,
)
from LearningPriority.order_of_priority import all_partial_orders


def generate_random_metrics(target_player_id: str, actions: Dict[str, set], num_metrics):
    """
    Generate a list of random metrics. Each metric is a function from action profile to a float in [0,1].
    """
    metrics = []
    player_ids = list(actions.keys())
    all_profiles = list(itertools.product(*[actions[pid] for pid in player_ids]))

    for m in range(num_metrics):
        payoff_map = {}
        for ap in all_profiles:
            payoff_map[ap] = random.uniform(0, 1)
        def make_metric_func(payoff_map):
            return lambda ap: payoff_map[tuple(ap[pid] for pid in player_ids)]
        metrics.append(Metric(f"{target_player_id}M{m+1}", make_metric_func(payoff_map)))
    return metrics


def enumerate_preferences(metric_names):
    """
    Enumerate all possible partial orders over the set of metric names.
    """
    return list(all_partial_orders(set(metric_names)))


def build_single_player(player_id, actions: Dict[str, set], num_metrics, max_preferences):
    """
    Build a single player with random metrics and a random true preference.
    """
    metrics = generate_random_metrics(player_id, actions, num_metrics)
    metric_names = [m.name for m in metrics]
    possible_prefs = enumerate_preferences(metric_names)
    if len(possible_prefs) > max_preferences:
        possible_prefs = random.sample(possible_prefs, max_preferences)
    true_preference = random.choice(possible_prefs)
    player = Player(player_id, actions=set(actions[player_id]), metrics=set(metrics), preference=true_preference)
    return player, true_preference, possible_prefs


def run_case_study(I=3, A=2, M=2, max_preferences=10, steps=20, seed=42):
    random.seed(seed)
    player_ids = [f"P{i+1}" for i in range(I)]
    players = []
    true_prefs = {}
    possible_prefs = {}
    prior = {}

    # Build players with different metric sets and preferences
    for pid in player_ids:
        actions = {pid: {f"{pid}A{j+1}" for j in range(A)}}
        player, true_pref, prefs = build_single_player(pid, actions, M, max_preferences)
        true_prefs[pid] = true_pref
        possible_prefs[pid] = prefs
        players.append(player)
        # randomly generate prior beliefs, just ensure non of them are zero
        probs = [random.uniform(0.3, 1.0) for _ in prefs]
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]
        prior[pid] = PreferenceBelief({p: prob for p, prob in zip(prefs, probs)})

    game = PosetalGame(players)

    # Try both algorithms
    results = {}
    for alg_name, AlgClass in [
        ("probability", WeightedVotingPureNEProbability),
        ("max", WeightedVotingPureNEMax),
    ]:
        algorithms = {pid: AlgClass() for pid in player_ids}
        lf = LearningFramework(game, prior, algorithms)
        belief_trajectories = {pid: {pref: [prior[pid].belief.get(pref, 0.0)] for pref in possible_prefs[pid]} for pid in player_ids}
        for step in range(steps):
            lf.run_iteration()
            for pid in player_ids:
                belief = lf.current_beliefs()[pid]
                for pref in possible_prefs[pid]:
                    belief_trajectories[pid][pref].append(belief.belief.get(pref, 0.0))
        results[alg_name] = (belief_trajectories, true_prefs, possible_prefs)
    return results, player_ids, game


def plot_belief_trajectories(results, player_ids, steps=20, print_title=True, print_legend=True, simple_tilte=False, simple_legend=False, algo_name=None):
    """
    Plot belief trajectories per player for each algorithm.
    Adapts to per-player preference sets and variable trajectory lengths.
    """
    if algo_name:
        results = {algo_name: results[algo_name]}
    for alg_name, (belief_trajectories, true_prefs, possible_prefs) in results.items():
        fig, axes = plt.subplots(len(player_ids), 1, figsize=(10, 3.2*len(player_ids)), sharex=True)
        if len(player_ids) == 1:
            axes = [axes]
        for i, pid in enumerate(player_ids):
            ax = axes[i]
            # Determine actual number of recorded steps (robust to mismatch)
            recorded_steps = 0
            if possible_prefs[pid]:
                # Use the maximum length among existing trajectories for this player
                recorded_steps = max(
                    (len(belief_trajectories[pid].get(pref, [])) for pref in possible_prefs[pid]),
                    default=0,
                )
            x = list(range(recorded_steps or steps))

            # Plot each preference trajectory for this player
            for pref in possible_prefs[pid]:
                # Safely retrieve trajectory; default to zeros if missing
                y = belief_trajectories[pid].get(pref, [0.0] * (recorded_steps or steps))
                if not simple_legend:
                    ax.plot(x[:len(y)], y, label=str(pref))
                else:
                    ax.plot(x[:len(y)], y, label=f"Pref {possible_prefs[pid].index(pref)+1}")
            if print_title:
                if not simple_tilte:
                    ax.set_title(f"Player {pid} (True pref: {true_prefs[pid]})")
                else:
                    ax.set_title(f"Player {pid} (True pref: Pref {possible_prefs[pid].index(true_prefs[pid])+1})")
            ax.set_ylabel("Belief")
            # Place legend outside to avoid clutter when many prefs exist
            if print_legend:
                ax.legend(fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)

        axes[-1].set_xlabel("Step")
        fig.suptitle(f"Belief Trajectories ({alg_name} algorithm)")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
        
    return fig, axes


if __name__ == "__main__":
    results, player_ids, true_game = run_case_study(I=3, A=2, M=2, steps=20, seed=42)
    plot_belief_trajectories(results, player_ids, steps=20)
