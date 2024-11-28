# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
from collections import deque



# Parameters that help determine Pacman's decision
wall_reward = None
food_reward = 10
ghost_penalty = -1000
capsule_reward = 20
empty_location_reward = -0.05
edible_ghost_reward = 20
danger = 50
gamma = 0.7
max_iterations = 20
convergence_threshold = 0.01


#MDPAgent
# An Agent that uses a Markov Decision Process (MDP) to control Pacman's movement

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py

    def __init__(self):
        print ("Starting up MDPAgent!")
        name = "Pacman"

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    # Accesses api.py to retrieve corners and walls
    # Sets the height, width and map
    def registerInitialState(self, state):
        print ("Running registerInitialState for MDPAgent!")
        print ("I'm at:")
        print (api.whereAmI(state))
        self.corners = api.corners(state)
        self.height = getLayoutHeight(self.corners)
        self.width = getLayoutWidth(self.corners)
        self.walls = api.walls(state)
        self.map = self.empty_map_maker()
        
    # Runs at the end. 
    # Initialises the map, walls and corners after every run 
    def final(self, state):
        print ("Looks like the game just ended!")
        self.map = None
        self.walls = None
        self.corners = None

    

    # Pacman moves according to the best action calculated with value iteration
    # Returns the bect move Pacman can make
    def getAction(self, state):
        # Initialise legal moves
        legal = api.legalActions(state)

        # Remove STOP from legal moves
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
    
        # Initialise the map to an empty map
        if self.map is None:
            self.registerInitialState(state) 

        
        # After every action retrieve the current position of pacman
        pacman = api.whereAmI(state)

        # Call value iteration function to populate the map with values
        self.map = self.value_iteration(state,self.map)
        
        # Create a dictionary to look up an action and its value to choose the best one
        action_values = {}
        
        # Possible moves pacman can make
        moves = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }
        
        # Check each possible action
        for action in legal:

            # Get the values of each move
            dx, dy = moves[action]

            # Add the current location of pacman with the values of each move to obtain the next direction cell number
            new_x, new_y = pacman[0] + dx, pacman[1] + dy
            
            # Map each action to its value in the map
            action_values[action] = self.map[new_x][new_y]

        # Looks up the maximum value of each move and returns the move with the highest value
        best_action = max(action_values, key=action_values.get)

        # Return the best move Pacman can make
        return api.makeMove(best_action, legal)    

    
    # Value iteration MDP solver function that has an input of an empty map 
    # Returns a map with new utilities
    def value_iteration(self, state, empty_map):
        
        # Initialises food, capsules and ghost_states in the current state
        food = api.food(state)
        capsules = api.capsules(state)
        ghost_states = api.ghostStates(state)

        # Assigns a reward to each position in the map
        rewarded_map = self.assign_rewards(food, capsules, ghost_states)

        # Updates the rewarded map with decaying values around the ghost
        self.ghost_halo(rewarded_map, ghost_states)

        # Initialises iterations to 0
        iterations = 0

        # Starts iterating - until convergence or maximum iterations reached
        while iterations < max_iterations:

            # Creates a copy of an empty map for the iterating values
            U_copy = self.empty_map_maker()

            # Initialises delta to check for convergence
            delta = 0

            # For each cell in the grid
            for x in range(self.width):
                for y in range(self.height):

                    # Calculate the utility value for current state using Bellman equation
                    state_utility = self.bellman_equation(empty_map, (x, y), rewarded_map[x][y])
                    
                    # Store the new utility value in the copy map
                    U_copy[x][y] = state_utility

                    # If the current position is not None (a wall) then 
                    # Update delta to track the maximum change in utility values
                    if rewarded_map[x][y] is not None:
                        delta = abs(state_utility - rewarded_map[x][y])

            # Update the empty map with the new utility values
            empty_map = U_copy

            # Increment iteration counter
            iterations += 1

            # Check for convergence - if maximum change is below threshold, stop iterating
            if delta < convergence_threshold:
                break
        
        # Return the final map with computed utility values 
        return empty_map


    # Bellman Equation Function that takes a map, the current position and the reward of the current position
    # Returns the Bellman equation result: current reward + discounted future reward
    def bellman_equation(self, map, curr_pos, reward):
        # If current position is a wall (reward is None), return None
        if reward is None:  
            return None

        # Define possible actions as movements: North, South, East, West
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
       
        # Initialize the maximum value to negative infinity for comparison
        max_value = float('-inf')

        # Iterate through each possible action
        for action in actions:
            # Initialise the value for the current action
            value = 0

            # Get the movement coordinates for each action
            dx, dy = action

            # Calculate the new position if intended action is taken
            intended_x, intended_y = curr_pos[0] + dx, curr_pos[1] + dy

            # Check if intended move is valid (within bounds and not a wall)
            # If valid, add 80% of the utility value for intended direction
            # If move is invalid, stay in current position
            if (0 <= intended_x < self.width and 
                0 <= intended_y < self.height and 
                map[intended_x][intended_y] is not None):
                value += 0.8 * map[intended_x][intended_y]
            else:
                value += 0.8 * map[curr_pos[0]][curr_pos[1]] 


            # Calculate perpendicular movements (representing possible drift)
            # For each intended action, check both perpendicular directions
            for perpendicular in [(-dy, dx), (dy, -dx)]:

                # Get the movement coordinates for perpendicular direction
                px, py = perpendicular

                # Calculate new position for perpendicular movement
                new_x, new_y = curr_pos[0] + px, curr_pos[1] + py

                # Check if perpendicular move is valid
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height and 
                    map[new_x][new_y] is not None):
                    # Add 10% of the utility value for perpendicular direction
                    value += 0.1 * map[new_x][new_y]
                else:
                    # If move is invalid, stay in current position
                    value += 0.1 * map[curr_pos[0]][curr_pos[1]]  

            # Update max_value if current action yields better result
            max_value = max(max_value, value)
        bellman = float(reward + gamma * max_value)
        # Calculate and return the Bellman equation result  
        return bellman

    # A function that assigns rewards to an empty map
    # Returns a map with the assigned values
    def assign_rewards(self, food, capsules, ghost_states):

        # Initialise an empty map
        empty_map = self.empty_map_maker()

        # Iterate through all the coordinates in the map
        for x in range(self.width):
            for y in range(self.height):
                
                # Assign rewards to the coordinates 
                # if they are in the wall, have food, have a capsule
                if (x, y) in self.walls:
                    empty_map[x][y] = wall_reward
                elif (x, y) in food:
                    empty_map[x][y] = food_reward
                elif (x, y) in capsules:
                    empty_map[x][y] = capsule_reward
                
                # Assign large negative values to the entrance of the cave that spawns ghosts and has no food
                elif (x, y) == (10, 6):
                    empty_map[x][y] = -500
                elif (x, y) == (9, 6):
                    empty_map[x][y] = -500
                
                #  If not in any of the values above it is either a ghost or an empty location
                else:  
                    # Iterate through all the ghosts             
                    for ghost_state in ghost_states:
                        # Find the the ghost position
                        ghost_pos = (int(ghost_state[0][0]), int(ghost_state[0][1]))

                        # If the coordinate is a ghost and is edible, assign an edible ghost reward
                        # else assign a negative ghost penalty
                        if (x, y) == ghost_pos:
                            if ghost_state[1] > 0:  # Edible ghost
                                empty_map[x][y] = edible_ghost_reward
                            else:
                                empty_map[x][y] = ghost_penalty
                            break
                    else:  
                        empty_map[x][y] = empty_location_reward

        # Return a rewarded map
        return empty_map
        
    # Function that creates decaying values around non-edible ghosts and a reward for edible ones
    # Returns an updated map with the new values        
    def ghost_halo(self, rewarded_map, ghost_states):

        # Iterate through all the ghosts 
        for ghost_state in ghost_states:
            # Initialise the position of the ghost
            ghost_pos = (int(ghost_state[0][0]), int(ghost_state[0][1]))
            # Initialise the state of the ghost (edible == 1, non-edible == 0)
            edible = ghost_state[1]
            
            # Create a distance map using breadth first search of the ghosts
            distance_map = self.breadth_first_search(ghost_pos)
            
            # Iterate through all of the coordinates in the map
            for x in range(self.width):
                for y in range(self.height):

                    # Check if the reward in the particular location is not None (a wall)
                    if rewarded_map[x][y] is not None:

                        # Look up the reward of the particular coordinate from the distance map
                        distance = distance_map.get((x, y))

                        # If the ghost is not edible and the distance is greater than zero (its not the ghost)
                        # Create decreasing penalty based on distance further from ghost smaller penalty
                        if edible == 0:  # Dangerous ghost
                            if distance > 0:
                                danger_value = danger / (distance * distance)
                                rewarded_map[x][y] -= danger_value
                        
                        
                        # else if the ghost is edible make ghost and adjacent positions attractive
                        # except for spawn points (9,6) and (10,6)
                        else:  # Edible ghost
                            if distance <= 1:

                                rewarded_map[x][y] = edible_ghost_reward if (x,y) not in [(9,6), (10,6)] else rewarded_map[x][y]

        # return an updated map of the rewarded map  
        return rewarded_map

    
    # Breadth-First Search (BFS) function that starts from the given position `start` 
    # to calculate the shortest distances from `start` to all reachable positions 
    # within a maximum allowable distance.
    # Returns a dictionary of the reachable distances from the start position  
    def breadth_first_search(self, start):
        
        # Initialize a dictionary to store the distance of each position from the start
        distances = {}

        # Initialize the queue with the starting position and a distance of 0
        queue = deque([(start, 0)])
        
        # Use a set to keep track of positions that have already been explored
        explored = {start}

        # Define a maximum search distance, typically a third of the total grid size
        max_distance = (self.height * self.width) // 3

        # Begin the Breadth-First Search (BFS) loop
        while queue:

            # Dequeue the next position and its distance from the queue
            pos, dist = queue.popleft()

            # Record the current position and its distance in the distances dictionary
            distances[pos] = dist

            # If the maximum allowable distance is reached, stop processing further for this branch
            if dist >= max_distance:
                continue

            # Explore the neighbors of the current position
            for neighbor in self.neighbours(pos):          

                # Check if the neighbor is valid and has not been explored yet
                if neighbor and neighbor not in explored:
                    
                    # Add the neighbor to the queue with an incremented distance
                    queue.append((neighbor, dist + 1))
                    
                    # Mark the neighbor as explored
                    explored.add(neighbor)

        # Return the disctionary of distances
        return distances
        
    # Neighbours function that takes a cell and returns its neighbours
    # Returns the allowed neighbours the input cell can go
    def neighbours(self, pos):
        # Initialize empty list to store neighboring positions
        neighbors = []

        # Define the four possible movement directions
        directions = [
            (0, 1),   # North
            (0, -1),  # South
            (1, 0),   # East
            (-1, 0)   # West
        ]
        
        # Extract x and y coordinates from the input position
        x, y = pos

        # Check each possible direction
        for dx, dy in directions:
            # Calculate new position coordinates
            new_x = x + dx
            new_y = y + dy
            
            # Check if within grid bounds
            if (0 < new_x < self.width and 
                0 < new_y < self.height):
                
                # If new position is a wall, append None
                if (new_x, new_y) in self.walls:
                    neighbors.append(None)
                # If new position is valid, append the coordinates
                else:
                    neighbors.append((new_x, new_y))
            
            # If new position is out of bounds, append None
            else:
                neighbors.append(None)
        
        # Return list of neighboring positions (including None for invalid moves)        
        return neighbors

    # Function that creates and initializes a 2D grid representing the game map
    # Returns a map where walls are None and all other locations have empty_location_reward
    def empty_map_maker(self):
        # Create a 2D list (width x height) filled with empty_location_reward values
        # Using list comprehension to initialize all positions with default reward
        empty_map = [[empty_location_reward for _ in range(self.height)] for _ in range(self.width)]

        # Set wall positions to None
        for wall_pos in self.walls:
            empty_map[wall_pos[0]][wall_pos[1]] = None

        # Return the initialized map with walls marked as None and other cells with default rewards
        return empty_map
        
        

# Function taken from MapAgent.py from King's College London Week 5 practical material
# Finds the height of the Pacman layout
def getLayoutHeight(corners):
    height = -1
    for i in range(len(corners)):
        if corners[i][1] > height:
            height = corners[i][1]
    return height + 1

# Function taken from MapAgent.py from King's College London Week 5 practical material
# Finds the width of the Pacman layout
def getLayoutWidth(corners):
    width = -1
    for i in range(len(corners)):
        if corners[i][0] > width:
            width = corners[i][0]
    return width + 1

