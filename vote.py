import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)


# Define the agent
class VoterAgent(Agent):
    def __init__(self, unique_id, model, initial_vote):
        super().__init__(unique_id, model)
        self.vote = initial_vote

    def step(self):
        neighbors = self.model.get_random_neighbors(5)
        if neighbors:
            neighbor = random.choice(neighbors)
            previous_vote = self.vote
            self.vote = neighbor.vote
            self.model.record_interaction(self.unique_id, neighbor.unique_id, previous_vote, self.vote)


# Define the model
class VotingModel(Model):
    def __init__(self, num_voters, interaction_fraction=0.01):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.running = True
        self.interaction_fraction = interaction_fraction

        candidate_ratings = {
            '41': 34,
            '5': 18,
            '4': 14,
            '25': 11,
            '9': 10,
            '10': 5,
            '36': 4,
            '8': 3.6,
            '100': 1
        }

        self.candidates = list(candidate_ratings.keys())
        total_ratings = sum(candidate_ratings.values())
        self.probabilities = np.array([candidate_ratings[c] for c in self.candidates]) / total_ratings

        self.vote_counts = defaultdict(int)
        self.interaction_history = []
        self.datacollector = DataCollector(
            agent_reporters={"Vote": "vote"},
            model_reporters={"VoteCounts": "count_votes"}
        )

        for i in range(num_voters):
            initial_vote = np.random.choice(self.candidates, p=self.probabilities)
            agent = VoterAgent(i, self, initial_vote)
            self.schedule.add(agent)
            self.vote_counts[initial_vote] += 1

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def get_random_neighbors(self, num_neighbors):
        all_agents = list(self.schedule.agents)
        return random.sample(all_agents, min(num_neighbors, len(all_agents)))

    def count_votes(self):
        return dict(self.vote_counts)

    def record_interaction(self, voter_id, neighbor_id, previous_vote, new_vote):
        self.interaction_history.append((voter_id, neighbor_id, previous_vote, new_vote))


def run_simulation(num_voters, interaction_fraction, steps):
    model = VotingModel(num_voters, interaction_fraction)
    vote_history = []

    for _ in range(steps):
        model.step()
        vote_counts = model.count_votes()
        vote_history.append(vote_counts)

    return vote_history, model.interaction_history


# Parameters for batch run
model_params = {
    "num_voters": [1000],
    "interaction_fraction": [0.01]
}


def visualize_network(interaction_history, run_index, model):
    G = nx.Graph()

    # Add nodes with candidate names
    for agent in model.schedule.agents:
        G.add_node(agent.unique_id, vote=agent.vote)

    # Add edges based on interactions
    for voter_id, neighbor_id, prev_vote, new_vote in interaction_history:
        G.add_edge(voter_id, neighbor_id)

    # Check for edges
    if G.number_of_edges() == 0:
        print("No edges found in the graph.")
    else:
        print(f"Number of edges in the graph: {G.number_of_edges()}")

    # Set node labels to candidate names
    labels = {node: G.nodes[node]['vote'] for node in G.nodes()}

    # Create a color map for candidates
    candidate_colors = plt.cm.get_cmap('Set3', len(model.candidates))
    color_map = [candidate_colors(model.candidates.index(G.nodes[node]['vote'])) for node in G.nodes()]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=color_map, node_size=300, font_size=10)
    plt.title(f"Agent Interaction Network")
    plt.show()


if __name__ == "__main__":
    iterations = 3
    steps = 10

    # Use ThreadPoolExecutor for parallel execution of simulations
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, num_voters, interaction_fraction, steps)
                   for num_voters, interaction_fraction in
                   zip(model_params["num_voters"], model_params["interaction_fraction"])]

        results = []
        for future in futures:
            results.append(future.result())

    # Aggregate results and visualize networks
    aggregated_vote_counts = defaultdict(int)

    for run_index, (vote_history, interactions) in enumerate(results):
        # Aggregate votes
        for vote_counts in vote_history:
            for candidate, count in vote_counts.items():
                aggregated_vote_counts[candidate] += count

        # Visualize the interaction network for this run
        visualize_network(interactions, run_index, VotingModel(1000, 0.01))

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(aggregated_vote_counts, orient='index', columns=['Vote Count']).reset_index()
    results_df.rename(columns={'index': 'Candidate'}, inplace=True)

    # Display the results
    print("Voting Results DataFrame:")
    print(results_df)
