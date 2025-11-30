"""
Example: Simple coordination game with posetal preferences.

This example demonstrates:
1. Creating a posetal game with 2 players
2. Finding pure Nash equilibria
3. Running the learning framework
"""

from LearningPriority import (
    PartialOrder, Metric, Player, PosetalGame,
    NashEquilibriumFinder, LearningFramework
)


def create_coordination_game():
    """
    Create a simple 2-player coordination game.
    
    Both players choose between actions A and B.
    Each player has two metrics:
    - efficiency: rewards coordination (both choose same action)
    - fairness: rewards balanced outcomes
    """
    
    # Define metrics for Player 1
    def p1_efficiency(a1, a2):
        """Reward coordination."""
        if a1 == a2:
            return 10.0
        return 0.0
    
    def p1_fairness(a1, a2):
        """Reward 'fair' outcomes (both get same)."""
        if a1 == a2:
            return 5.0
        return 5.0  # All outcomes equally fair in this simple version
    
    # Define metrics for Player 2
    def p2_efficiency(a1, a2):
        """Reward coordination."""
        if a1 == a2:
            return 10.0
        return 0.0
    
    def p2_fairness(a1, a2):
        """Reward 'fair' outcomes."""
        if a1 == a2:
            return 5.0
        return 5.0
    
    # Create metrics
    m1_eff = Metric("efficiency", p1_efficiency)
    m1_fair = Metric("fairness", p1_fairness)
    m2_eff = Metric("efficiency", p2_efficiency)
    m2_fair = Metric("fairness", p2_fairness)
    
    # Create preferences (efficiency > fairness for both players)
    pref1 = PartialOrder(
        elements={"efficiency", "fairness"},
        relations={("efficiency", "fairness")}
    )
    
    pref2 = PartialOrder(
        elements={"efficiency", "fairness"},
        relations={("efficiency", "fairness")}
    )
    
    # Create players
    player1 = Player(
        player_id="P1",
        actions={"A", "B"},
        metrics={m1_eff, m1_fair},
        preference=pref1
    )
    
    player2 = Player(
        player_id="P2",
        actions={"A", "B"},
        metrics={m2_eff, m2_fair},
        preference=pref2
    )
    
    # Create game
    game = PosetalGame([player1, player2])
    
    return game


def main():
    """Run the example."""
    print("=" * 60)
    print("Example: Coordination Game with Posetal Preferences")
    print("=" * 60)
    
    # Create the game
    game = create_coordination_game()
    print(f"\n{game}")
    print(f"Number of action profiles: {len(game.action_profiles)}")
    print(f"Action profiles: {game.action_profiles}")
    
    # Find Nash equilibria
    print("\n" + "-" * 60)
    print("Finding Nash Equilibria...")
    print("-" * 60)
    
    ne_finder = NashEquilibriumFinder(game)
    nash_equilibria = ne_finder.find_pure_nash_equilibria()
    
    print(f"\nPure Nash Equilibria found: {len(nash_equilibria)}")
    for i, ne in enumerate(nash_equilibria, 1):
        print(f"  NE {i}: {ne}")
        
        # Check if it's strict
        if ne_finder.is_strict_nash_equilibrium(ne):
            print(f"         (strict)")
    
    # Find admissible Nash equilibria
    admissible_ne = ne_finder.find_admissible_nash_equilibria()
    print(f"\nAdmissible Nash Equilibria: {len(admissible_ne)}")
    for i, ne in enumerate(admissible_ne, 1):
        print(f"  Admissible NE {i}: {ne}")
    
    # Evaluate metrics at equilibria
    print("\n" + "-" * 60)
    print("Metric Values at Nash Equilibria:")
    print("-" * 60)
    
    for ne in nash_equilibria:
        print(f"\nAction profile: {ne}")
        for player in game.players:
            metrics = game.evaluate_metrics(player.player_id, ne)
            print(f"  {player.player_id}: {metrics}")
    
    # Demonstrate learning framework
    print("\n" + "=" * 60)
    print("Learning Framework Simulation")
    print("=" * 60)
    
    # Create alternative preferences for learning
    # Alternative: fairness > efficiency
    alt_pref1 = PartialOrder(
        elements={"efficiency", "fairness"},
        relations={("fairness", "efficiency")}
    )
    
    alt_pref2 = PartialOrder(
        elements={"efficiency", "fairness"},
        relations={("fairness", "efficiency")}
    )
    
    # True preference (efficiency > fairness)
    true_pref = PartialOrder(
        elements={"efficiency", "fairness"},
        relations={("efficiency", "fairness")}
    )
    
    possible_prefs = {
        "P1": {true_pref, alt_pref1},
        "P2": {true_pref, alt_pref2}
    }
    
    learning = LearningFramework(game, possible_prefs)
    
    print("\nRunning learning simulation for 20 iterations...")
    trajectory = learning.simulate(num_iterations=20)
    
    print("\nAction trajectory:")
    for i, action_profile in enumerate(trajectory[:10], 1):
        print(f"  Round {i}: {action_profile}")
    
    if len(trajectory) > 10:
        print(f"  ... ({len(trajectory) - 10} more rounds)")
    
    # Show converged beliefs
    print("\n" + "-" * 60)
    print("Converged Beliefs:")
    print("-" * 60)
    
    converged = learning.get_converged_beliefs()
    for player_id, pref in converged.items():
        print(f"\n{player_id} most likely preference:")
        print(f"  Elements: {set(pref.elements)}")
        
        # Show which metric is preferred
        if pref.greater("efficiency", "fairness"):
            print(f"  Prefers: efficiency > fairness")
        elif pref.greater("fairness", "efficiency"):
            print(f"  Prefers: fairness > efficiency")
        else:
            print(f"  Incomparable preferences")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
