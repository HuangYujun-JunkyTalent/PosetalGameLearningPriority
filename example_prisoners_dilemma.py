"""
Example: Prisoner's Dilemma with posetal preferences.

This example shows how the classic Prisoner's Dilemma can be modeled
with posetal preferences, where players care about both personal payoff
and fairness/cooperation metrics.
"""

from LearningPriority import (
    PartialOrder, Metric, Player, PosetalGame,
    NashEquilibriumFinder
)


def create_prisoners_dilemma_posetal():
    """
    Create a Prisoner's Dilemma with posetal preferences.
    
    Players can Cooperate (C) or Defect (D).
    Each player has three metrics:
    - personal_payoff: standard PD payoff
    - cooperation: rewards mutual cooperation
    - fairness: penalizes exploiting the other player
    """
    
    # Standard PD payoffs: (R, S, T, P) = (3, 0, 5, 1)
    # Where R=reward, S=sucker, T=temptation, P=punishment
    
    # Player 1 metrics
    def p1_payoff(a1, a2):
        """Standard PD payoff for player 1."""
        if a1 == "C" and a2 == "C":
            return 3.0  # Mutual cooperation
        elif a1 == "C" and a2 == "D":
            return 0.0  # Sucker
        elif a1 == "D" and a2 == "C":
            return 5.0  # Temptation
        else:  # Both defect
            return 1.0  # Punishment
    
    def p1_cooperation(a1, a2):
        """Rewards mutual cooperation."""
        if a1 == "C" and a2 == "C":
            return 10.0
        elif a1 == "C":
            return 5.0  # I cooperated
        else:
            return 0.0
    
    def p1_fairness(a1, a2):
        """Penalizes unfair outcomes (exploiting other)."""
        if a1 == "D" and a2 == "C":
            return 0.0  # Exploiting is unfair
        else:
            return 5.0
    
    # Player 2 metrics (symmetric)
    def p2_payoff(a1, a2):
        """Standard PD payoff for player 2."""
        if a1 == "C" and a2 == "C":
            return 3.0
        elif a1 == "D" and a2 == "C":
            return 0.0  # Sucker
        elif a1 == "C" and a2 == "D":
            return 5.0  # Temptation
        else:
            return 1.0
    
    def p2_cooperation(a1, a2):
        """Rewards mutual cooperation."""
        if a1 == "C" and a2 == "C":
            return 10.0
        elif a2 == "C":
            return 5.0
        else:
            return 0.0
    
    def p2_fairness(a1, a2):
        """Penalizes unfair outcomes."""
        if a1 == "C" and a2 == "D":
            return 0.0
        else:
            return 5.0
    
    # Create metrics
    m1_payoff = Metric("payoff", p1_payoff)
    m1_coop = Metric("cooperation", p1_cooperation)
    m1_fair = Metric("fairness", p1_fairness)
    
    m2_payoff = Metric("payoff", p2_payoff)
    m2_coop = Metric("cooperation", p2_cooperation)
    m2_fair = Metric("fairness", p2_fairness)
    
    return {
        "metrics": {
            "P1": {m1_payoff, m1_coop, m1_fair},
            "P2": {m2_payoff, m2_coop, m2_fair}
        },
        "elements": {"payoff", "cooperation", "fairness"}
    }


def main():
    """Run the Prisoner's Dilemma example."""
    print("=" * 70)
    print("Example: Prisoner's Dilemma with Posetal Preferences")
    print("=" * 70)
    
    game_data = create_prisoners_dilemma_posetal()
    
    # Create three different preference profiles to compare
    scenarios = {
        "Selfish": {
            "description": "Players prioritize payoff > fairness > cooperation",
            "pref1": PartialOrder(
                elements=game_data["elements"],
                relations={("payoff", "fairness"), ("payoff", "cooperation"),
                          ("fairness", "cooperation")}
            ),
            "pref2": PartialOrder(
                elements=game_data["elements"],
                relations={("payoff", "fairness"), ("payoff", "cooperation"),
                          ("fairness", "cooperation")}
            )
        },
        "Cooperative": {
            "description": "Players prioritize cooperation > fairness > payoff",
            "pref1": PartialOrder(
                elements=game_data["elements"],
                relations={("cooperation", "fairness"), ("cooperation", "payoff"),
                          ("fairness", "payoff")}
            ),
            "pref2": PartialOrder(
                elements=game_data["elements"],
                relations={("cooperation", "fairness"), ("cooperation", "payoff"),
                          ("fairness", "payoff")}
            )
        },
        "Fair": {
            "description": "Players prioritize fairness > cooperation > payoff",
            "pref1": PartialOrder(
                elements=game_data["elements"],
                relations={("fairness", "cooperation"), ("fairness", "payoff"),
                          ("cooperation", "payoff")}
            ),
            "pref2": PartialOrder(
                elements=game_data["elements"],
                relations={("fairness", "cooperation"), ("fairness", "payoff"),
                          ("cooperation", "payoff")}
            )
        },
        "Mixed": {
            "description": "P1 is selfish, P2 is cooperative",
            "pref1": PartialOrder(
                elements=game_data["elements"],
                relations={("payoff", "fairness"), ("payoff", "cooperation"),
                          ("fairness", "cooperation")}
            ),
            "pref2": PartialOrder(
                elements=game_data["elements"],
                relations={("cooperation", "fairness"), ("cooperation", "payoff"),
                          ("fairness", "payoff")}
            )
        }
    }
    
    # Analyze each scenario
    for scenario_name, scenario in scenarios.items():
        print("\n" + "=" * 70)
        print(f"Scenario: {scenario_name}")
        print(f"{scenario['description']}")
        print("=" * 70)
        
        # Create players with these preferences
        player1 = Player(
            player_id="P1",
            actions={"C", "D"},
            metrics=game_data["metrics"]["P1"],
            preference=scenario["pref1"]
        )
        
        player2 = Player(
            player_id="P2",
            actions={"C", "D"},
            metrics=game_data["metrics"]["P2"],
            preference=scenario["pref2"]
        )
        
        game = PosetalGame([player1, player2])
        finder = NashEquilibriumFinder(game)
        
        # Find Nash equilibria
        nash_eq = finder.find_pure_nash_equilibria()
        admissible_ne = finder.find_admissible_nash_equilibria()
        
        print(f"\nAction profiles: {game.action_profiles}")
        print(f"\nPure Nash Equilibria: {len(nash_eq)}")
        for ne in nash_eq:
            is_strict = finder.is_strict_nash_equilibrium(ne)
            is_admissible = ne in admissible_ne
            print(f"  {ne} {'(strict)' if is_strict else ''} {'[admissible]' if is_admissible else ''}")
        
        # Show metrics for each Nash equilibrium
        if nash_eq:
            print("\nMetric values at Nash Equilibria:")
            for ne in nash_eq:
                print(f"\n  Action profile: {ne}")
                for player in game.players:
                    metrics = game.evaluate_metrics(player.player_id, ne)
                    print(f"    {player.player_id}: {metrics}")
        
        # Show metrics for comparison at all action profiles
        if scenario_name in ["Selfish", "Cooperative"]:
            print("\n  All action profiles for comparison:")
            for ap in game.action_profiles:
                print(f"    {ap}:")
                for player in game.players:
                    metrics = game.evaluate_metrics(player.player_id, ap)
                    print(f"      {player.player_id}: {metrics}")
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("""
1. In the 'Selfish' scenario, players prioritize personal payoff.
   The Nash equilibrium is mutual defection (D,D) - classic PD result.

2. In the 'Cooperative' scenario, players prioritize cooperation.
   The Nash equilibrium shifts to mutual cooperation (C,C).

3. In the 'Fair' scenario, players prioritize fairness.
   This also supports cooperation as exploiting others is considered unfair.

4. In the 'Mixed' scenario, outcomes depend on the interplay of preferences.
   The asymmetry in preferences can lead to different equilibria.

This demonstrates how posetal preferences can model ethical considerations
and social preferences that go beyond simple payoff maximization.
    """)
    
    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
