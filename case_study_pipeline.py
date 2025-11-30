import random
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict

from LearningPriority.game import PosetalGame, Player, Metric, ActionProfile
from LearningPriority.orders import PartialOrder
from LearningPriority.learning import (
    PreferenceBelief,
    WeightedVotingPureNEProbability,
    WeightedVotingPureNEMax,
    LearningFramework,
)
from LearningPriority.order_of_priority import all_partial_orders


def generate_random_metrics(player_ids, actions, num_metrics):
    """
    Generate a list of random metrics. Each metric is a function from action profile to a float in [0,1].
    """
    metrics = []
    all_profiles = list(itertools.product(*[actions for _ in player_ids]))
    for m in range(num_metrics):
        payoff_map = {}
        for ap in all_profiles:
            payoff_map[ap] = random.uniform(0, 1)
        def make_metric_func(payoff_map):
            return lambda ap: payoff_map[tuple(ap[pid] for pid in player_ids)]
        metrics.append(Metric(f"M{m+1}", make_metric_func(payoff_map)))
    return metrics


def enumerate_preferences(metric_names):
    """
    Enumerate all possible partial orders over the set of metric names.
    """
    return list(all_partial_orders(set(metric_names)))


def build_players(player_ids, actions, metrics, true_prefs):
    players = []
    for pid, true_pref in zip(player_ids, true_prefs):
        players.append(Player(pid, set(actions), metrics, true_pref))
    return players


def uniform_prior_beliefs(player_ids, possible_prefs):
    prior = {}
    for pid in player_ids:
        prefs = possible_prefs[pid]
        prob = 1.0 / len(prefs)
        prior[pid] = PreferenceBelief({p: prob for p in prefs})
    return prior


def run_case_study(I=3, A=2, M=2, steps=20, seed=42):
    random.seed(seed)
    player_ids = [f"P{i+1}" for i in range(I)]
    actions = [f"A{j+1}" for j in range(A)]
    metrics = generate_random_metrics(player_ids, actions, M)
    metric_names = [m.name for m in metrics]
    all_prefs = enumerate_preferences(metric_names)
    # For tractability, limit to first 10 preferences if too many
    if len(all_prefs) > 10:
        all_prefs = all_prefs[:10]
    possible_prefs = {pid: all_prefs for pid in player_ids}
    true_prefs = [random.choice(all_prefs) for _ in player_ids]
    players = build_players(player_ids, actions, metrics, true_prefs)
    game = PosetalGame(players)
    prior = uniform_prior_beliefs(player_ids, possible_prefs)

    # Try both algorithms
    results = {}
    for alg_name, AlgClass in [
        ("probability", WeightedVotingPureNEProbability),
        ("max", WeightedVotingPureNEMax),
    ]:
        algorithms = {pid: AlgClass() for pid in player_ids}
        lf = LearningFramework(game, prior, algorithms)
        belief_trajectories = defaultdict(lambda: defaultdict(list))
        for step in range(steps):
            lf.run_iteration()
            for pid in player_ids:
                belief = lf.current_beliefs()[pid]
                for pref in possible_prefs[pid]:
                    belief_trajectories[pid][pref].append(belief.belief.get(pref, 0.0))
        results[alg_name] = (belief_trajectories, true_prefs, possible_prefs)
    return results, player_ids


def plot_belief_trajectories(results, player_ids, steps=20):
    for alg_name, (belief_trajectories, true_prefs, possible_prefs) in results.items():
        fig, axes = plt.subplots(len(player_ids), 1, figsize=(8, 3*len(player_ids)), sharex=True)
        if len(player_ids) == 1:
            axes = [axes]
        for i, pid in enumerate(player_ids):
            ax = axes[i]
            for pref in possible_prefs[pid]:
                ax.plot(range(steps), belief_trajectories[pid][pref], label=str(pref))
            ax.set_title(f"Player {pid} (True: {true_prefs[i]})")
            ax.set_ylabel("Belief")
            ax.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[-1].set_xlabel("Step")
        fig.suptitle(f"Belief Trajectories ({alg_name} algorithm)")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


if __name__ == "__main__":
    results, player_ids = run_case_study(I=3, A=2, M=2, steps=20, seed=42)
    plot_belief_trajectories(results, player_ids, steps=20)
