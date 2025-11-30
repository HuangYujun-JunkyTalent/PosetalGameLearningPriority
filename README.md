# LearningPriority: Posetal Games Learning Framework

This package provides a framework for modeling, simulating, and learning in posetal games—games where player preferences are represented by partial orders over multiple metrics.

## Features
- **Order Theory Structures:** PreOrder, PartialOrder, Hasse diagrams, minimal/maximal elements.
- **Game Modeling:** Players, metrics, action profiles, posetal game construction.
- **Nash Equilibrium Finders:** Brute-force pure NE and admissible NE search.
- **Learning Algorithms:** Weighted voting (probability and max) for preference learning.
- **Case Study Pipeline:** Easily simulate and visualize learning dynamics in random games.

## Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
- Run the case study pipeline and plot belief evolution:
  ```bash
  python case_study_pipeline.py
  ```
- See `tests/` for unit tests and examples of core functionality.

## Project Structure
- `orders.py` — Order theory classes and utilities
- `order_of_priority.py` — Partial order enumeration
- `game.py` — Game, Player, Metric, ActionProfile
- `nash_finder.py` — Nash equilibrium search
- `learning.py` — Learning framework and algorithms
- `case_study_pipeline.py` — Example pipeline for experiments and plotting
- `tests/` — Unit tests

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## License
MIT
