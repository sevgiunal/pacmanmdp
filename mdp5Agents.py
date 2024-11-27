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
import random
import game
import util
from collections import deque





wall_reward = None
food_reward = 10
ghost_reward = -1000
capsule_reward = 20
empty_location_reward = -0.05
edible_ghost_reward = 20
danger = 50
gamma = 0.9
MAX_ITERATIONS = 20
CONVERGENCE_THRESHOLD = 0.01


class MDP5Agent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print ("Starting up MDPAgent!")
        name = "Pacman"
        self.map = None
        self.walls = None
        self.corners = None


    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print ("Running registerInitialState for MDPAgent!")
        print ("I'm at:")
        print (api.whereAmI(state))
        self.corners = api.corners(state)
        self.height = getLayoutHeight(self.corners)
        self.width = getLayoutWidth(self.corners)
        self.walls = api.walls(state)
        self.map = self.empty_map_maker()
        # self.visited_cells = set()
        
    def final(self, state):
        print ("Looks like the game just ended!")

    def getAction(self, state):
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
    
        if self.map is None:
            self.registerInitialState(state) 

        
        pacman = api.whereAmI(state)
        ghosts = api.ghosts(state)
        
        self.map = self.value_iteration(state,self.map, ghosts )

        [scores, actions] = self.get_action_scores(legal, self.map, pacman[0], pacman[1])
        max_score_index = scores.index(max(scores))
        choice = actions[max_score_index]

        # printMap(self.map, pacman, ghosts, self.height, self.width)

        return api.makeMove(choice, legal)    
    

    def get_action_scores(self, legal, pacman_map, x, y):
        scores = []
        actions = []
        moves = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }
        
        for action in legal:
            dx, dy = moves[action]
            new_x, new_y = x + dx, y + dy
            
            # Check bounds and walls
            if (0 <= new_x < self.width and 
                0 <= new_y < self.height and 
                pacman_map[new_x][new_y] is not None):
                scores.append(pacman_map[new_x][new_y])
                actions.append(action)

        return scores, actions
    

    def value_iteration(self, state, empty_map, ghosts):
        iterations = 0

        food = api.food(state)

        capsules = api.capsules(state)
        ghost_states = api.ghostStates(state)

        rewarded_map = self.assign_rewards(food, capsules, ghost_states)


        pacman = api.whereAmI(state)
        pacman = (pacman[1], pacman[0])
        self.ghost_halo(pacman, rewarded_map, ghost_states)

        while iterations < MAX_ITERATIONS:
            

            new_m = self.empty_map_maker()
            delta = 0

            for i in range(self.width):
                for j in range(self.height):
                    
                    r = rewarded_map[i][j]
                    new_value = self.bellman_equation(empty_map, (i, j), r)

                    if r is not None:
                        delta = max(delta, abs(new_value- rewarded_map[i][j]))

                    new_m[i][j] = new_value
            # print("Iteration:", iterations)
            # print("Delta: ", delta)
            empty_map = new_m
            iterations += 1
            if delta < CONVERGENCE_THRESHOLD:
                # print(delta)
                break
        
        

        return empty_map

    def bellman_equation(self, m, cell, r):
        if r is None:  # wall
            return None

        x, y = cell
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        max_value = float('-inf')

        for intended_action in actions:
            value = 0
            ix, iy = intended_action

            # Calculate value for intended action (0.8 probability)
            intended_x, intended_y = x + ix, y + iy
            if (0 <= intended_x < self.width and 
                0 <= intended_y < self.height and 
                m[intended_x][intended_y] is not None):
                value += 0.8 * m[intended_x][intended_y]
            else:
                value += 0.8 * m[x][y]  # Stay in place if would hit wall

            # Calculate values for perpendicular actions (0.1 probability each)
            for perpendicular in [(-iy, ix), (iy, -ix)]:  # Perpendicular directions
                px, py = perpendicular
                new_x, new_y = x + px, y + py
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height and 
                    m[new_x][new_y] is not None):
                    value += 0.1 * m[new_x][new_y]
                else:
                    value += 0.1 * m[x][y]  # Stay in place if would hit wall

                    # Apply penalty for revisiting cells
            # if (x, y) in self.visited_cells:
            #     value *= 0.8  # Reduce the value by 20% if the cell has been visited before


            max_value = max(max_value, value)

            # self.visited_cells.add((x, y))  

        return float(r + gamma * max_value)


    def assign_rewards(self, food, capsules, ghost_states):
        empty_map = self.empty_map_maker()
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                if pos in self.walls:
                    empty_map[x][y] = wall_reward
                elif pos in food:
                    empty_map[x][y] = food_reward
                elif pos in capsules:
                    empty_map[x][y] = capsule_reward
                elif pos == (10, 6):
                    empty_map[x][y] = -500
                elif pos == (9, 6):
                    empty_map[x][y] = -500
                else:


                                    
                    for ghost_state in ghost_states:
                        ghost_pos = (int(ghost_state[0][0]), int(ghost_state[0][1]))
                        if pos == ghost_pos:
                            if ghost_state[1] > 0:  # Edible ghost
                                empty_map[x][y] = edible_ghost_reward
                            else:  # Dangerous ghost
                                empty_map[x][y] = ghost_reward
                            break
                    else:  # No ghost at this position
                        empty_map[x][y] = empty_location_reward


                        

        return empty_map
        
            
    def ghost_halo(self, pacman, rewarded_map, ghost_states):
        for ghost_state in ghost_states:
            ghost_pos = (int(ghost_state[0][0]), int(ghost_state[0][1]))
            edible = ghost_state[1]
            
            # Calculate distances once per ghost
            distance_map = self.breadth_first_search(ghost_pos)
            
            for x in range(self.width):
                for y in range(self.height):
                    if rewarded_map[x][y] is not None:
                        distance = distance_map.get((x, y))
                        if distance is None:
                            continue
                            
                        if edible == 0:  # Dangerous ghost
                            if distance > 0:
                                danger_value = danger / (distance * distance)
                                rewarded_map[x][y] -= danger_value
                        else:  # Edible ghost
                            if distance <= 1:

                                rewarded_map[x][y] = edible_ghost_reward if (x,y) not in [(9,6), (10,6)] else rewarded_map[x][y]

          
        return rewarded_map

    
    
    def breadth_first_search(self, start):
        distances = {}
        queue = deque([(start, 0)])
        explored = {start}
        max_distance = (self.height * self.width) // 3
        # print(max_distance)

        while queue:
            pos, dist = queue.popleft()
            distances[pos] = dist

            if dist >= max_distance:
                continue

            for neighbor in self.neighbours(pos):
                if neighbor and neighbor not in explored:
                    queue.append((neighbor, dist + 1))
                    explored.add(neighbor)

        return distances
        
    def neighbours(self, pos):
        # returns the allowed direction pacman can go, its neighbours
        neighbors = []
        directions = [
            (0, 1),   # North
            (0, -1),  # South
            (1, 0),   # East
            (-1, 0)   # West
        ]
        
        x, y = pos
        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy
            # Check if within grid bounds
            if (0 < new_x < self.width and 
                0 < new_y < self.height):
                if (new_x, new_y) in self.walls:
                    neighbors.append(None)
                else:
                    neighbors.append((new_x, new_y))
            else:
                neighbors.append(None)
                
        return neighbors
    

    def empty_map_maker(self):

        empty_map = [[empty_location_reward for _ in range(self.height)] for _ in range(self.width)]
        for wall_pos in self.walls:
            empty_map[wall_pos[0]][wall_pos[1]] = None
        return empty_map

     
        

# taken from MapAgent.py
def getLayoutHeight(corners):
    height = -1
    for i in range(len(corners)):
        if corners[i][1] > height:
            height = corners[i][1]
    return height + 1

# taken from MapAgent.py
def getLayoutWidth(corners):
    width = -1
    for i in range(len(corners)):
        if corners[i][0] > width:
            width = corners[i][0]
    return width + 1


def printMap(grid, pacman_pos, ghost_positions, height, width):
        """Prints a formatted view of the utilities grid with colored values"""
        # ANSI color codes
        GREEN = '\033[92m'      # Positive values
        RED = '\033[95m'        # Negative values
        BLUE = '\033[94m'       # Zero values
        RESET = '\033[0m'
        PACMAN = '\033[43m'     # Yellow background for Pacman
        GHOST = '\033[101m'     # Bright red background for ghosts
        BOLD = '\033[1m'        # Bold text
        
        
        for y in range(height-1, -1, -1):
            row = ""
            for x in range(width):
                if grid[x][y] is None:  # Wall
                    row += "##########"     # 10 spaces for walls
                else:
                    value = grid[x][y]
                    if value > 0:
                        color = GREEN
                    elif value < 0:
                        color = RED
                    else:
                        color = BLUE
                    
                    # Add highlights for Pacman and ghosts
                    if (x, y) == pacman_pos:
                        row += PACMAN + BOLD + color + "{:8.2f}  ".format(value) + RESET
                    elif (x, y) in ghost_positions:
                        row += GHOST + BOLD + color + "{:8.2f}  ".format(value) + RESET
                    else:
                        row += color + "{:8.2f}  ".format(value) + RESET
            print(row)
        print("-" * (width * 10))


