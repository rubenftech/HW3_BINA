from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy


actions_to_idx = {"UP": 0, "DOWN": 1, "RIGHT": 2, "LEFT": 3}
idx_to_actions = ["UP", "DOWN", "RIGHT", "LEFT"]


def transition(mdp: MDP, s_from: Tuple[int, int], s_to: Tuple[int, int], action: Action) -> float:
    if s_from in mdp.terminal_states:
        return 0.0
    action = Action(action)
    prob = 0.0

    for wanted_action, move in mdp.actions.items():
        new_state = mdp.step(s_from , wanted_action)
        if new_state == s_to:
            action_idx = list(mdp.actions.keys()).index(wanted_action)
            prob += mdp.transition_function[action][action_idx]
    return prob


def total_utility(mdp, U, state, action):
    states = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            states.append((i, j))
    utilities = [transition(mdp, state, s_to, action) * U[s_to[0]][s_to[1]] for s_to in states]
    return sum(utilities)


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float = 10 ** (-3)) -> np.ndarray:
    U = deepcopy(U_init)
    delta = float('inf')

    while delta > epsilon * (1 - mdp.gamma) / mdp.gamma:
        delta = 0
        U_prev = deepcopy(U)

        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == "WALL":
                    continue

                if (i, j) in mdp.terminal_states:
                    U[i][j] = float(mdp.get_reward((i, j)))
                    continue

                max_utility = float('-inf')
                for a in mdp.actions:
                    action_probabilities = mdp.transition_function[a]
                    expected_utility = 0

                    for k, dir_action in enumerate(mdp.actions):
                        next_state = mdp.step((i, j), dir_action)
                        probability_next_state = action_probabilities[k]
                        utility_next_state = U[next_state[0]][next_state[1]]
                        expected_utility += probability_next_state * utility_next_state

                    max_utility = max(max_utility, expected_utility)

                U[i][j] = float(mdp.get_reward((i, j))) + mdp.gamma * max_utility
                delta = max(delta, abs(U[i][j] - U_prev[i][j]))

    return U


def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    policy = np.full((mdp.num_row, mdp.num_col), None)

    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue

            if (i, j) in mdp.terminal_states:
                policy[i][j] = 'TERMINAL'
                continue

            state = (i, j)
            best_action = None
            max_utility = float('-inf')

            for action in mdp.actions:
                expected_utility = 0
                action_probabilities = mdp.transition_function[action]

                for k, dir_action in enumerate(mdp.actions):
                    next_state = mdp.step(state, dir_action)
                    probability_next_state = action_probabilities[k]
                    utility_next_state = U[next_state[0]][next_state[1]]
                    expected_utility += probability_next_state * utility_next_state

                if expected_utility > max_utility:
                    max_utility = expected_utility
                    best_action = action.value

            policy[i][j] = best_action

    return policy



def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:
    states = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            states.append((i, j))

    rewards_list = []
    for s in states:
        reward_value = float(mdp.board[s[0]][s[1]])
        rewards_list.append(reward_value)
    rewards = np.array(rewards_list)

    transitions_matrix = []
    for s_from in states:
        row = []
        for s_to in states:
            action = policy[s_from[0]][s_from[1]]
            transition_prob = 0 if s_from in mdp.terminal_states else transition(mdp, s_from, s_to, action)
            row.append(transition_prob)
        transitions_matrix.append(row)
    transitions_matrix = np.array(transitions_matrix)

    identity_matrix = np.eye(len(rewards))
    discounted_transition_matrix = mdp.gamma * transitions_matrix
    matrix_to_invert = identity_matrix - discounted_transition_matrix
    inverse_matrix = np.linalg.inv(matrix_to_invert)
    utility = inverse_matrix @ rewards

    U = deepcopy(policy)
    for s, u in zip(states, utility.tolist()):
        i, j = s
        U[i][j] = u

    return U


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:
    policy = deepcopy(policy_init)
    changed = True
    states = []
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if mdp.board[i][j] == "WALL":
                continue
            states.append((i, j))

    while changed:
        U = policy_evaluation(mdp, policy)
        changed = False

        for s in states:
            utilities = [total_utility(mdp, U, s, a) for a in actions_to_idx]
            total = total_utility(mdp, U, s, policy[s[0]][s[1]])
            if max(utilities) > total and not np.isclose(max(utilities), total, rtol=1e-6):
                i, j = s
                policy[i][j] = idx_to_actions[utilities.index(max(utilities))]
                changed = True
    optimal_policy = policy
    return optimal_policy


def adp_algorithm(
        sim: Simulator,
        num_episodes: int,
        num_rows: int = 3,
        num_cols: int = 4,
        actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid,
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.

    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions.
    """

    reward_matrix = np.zeros((num_rows, num_cols), dtype=float)
    transition_probs = {action: {a: 0.0 for a in actions} for action in actions}
    num_of_action = {a: 0 for a in actions}

    for episode_index, episode_gen in enumerate(sim.replay(num_episodes)):
        for step_index, step in enumerate(episode_gen):
            state, reward, action, actual_action = step
            reward_matrix[state] = reward
            if action is None:
                break
            transition_probs[action][actual_action] += 1
            num_of_action[action] += 1

    for from_action, to_actions in transition_probs.items():
        for to_action in to_actions:
            if num_of_action[from_action] != 0:
                tmp = (transition_probs[from_action][to_action] / num_of_action[from_action])
                transition_probs[from_action][to_action] = tmp

    return reward_matrix, transition_probs
