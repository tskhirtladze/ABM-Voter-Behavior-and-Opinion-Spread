# Voter Interaction Simulation

## Overview
This project simulates voter behavior using an agent-based model where each voter interacts with randomly chosen neighbors to influence their voting preferences. The simulation captures how social interactions affect opinion formation and vote adoption in a voting scenario.

## Features
- **Agent-Based Modeling**: Each voter agent adopts their vote based on interactions with others.
- **Candidate Representation**: Voters start with initial votes influenced by pre-defined candidate ratings.
- **Interaction Recording**: The model tracks and visualizes interactions between agents.
- **Network Visualization**: Interactions are displayed as a network graph, with different colors representing different candidates.
- **Reproducibility**: Fixed random seeds ensure consistent results across multiple runs.

## Requirements
- Python 3.x
- Mesa
- NumPy
- Matplotlib
- NetworkX
- Pandas

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/tskhirtladze/ABM-Voter-Behavior-and-Opinion-Spread.git
   cd ABM-Voter-Behavior-and-Opinion-Spread
    ```
   
2. Install the required packages:
    ```bash
   pip install mesa numpy matplotlib networkx pandas
   ```
   
## Usage
Run the simulation script:
```python
python vote.py
```

## Results
The simulation will generate a network graph visualizing the interactions and voting behavior over time. The results will also be printed in a DataFrame format, summarizing the final vote counts for each candidate.
