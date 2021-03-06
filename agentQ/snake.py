import logging
import util

log = logging.getLogger("client.snake")

from collections import deque
from typing import List
from copy import deepcopy

import random
import numpy as np
from ga import GeneticThing
from mlp import MLP
# from overrides import overrides

from base_snake import BaseSnake, Action
from util import Direction, Map, translate_coordinate


log = logging.getLogger("snake")

WIDTH = 46
HEIGHT = 34

# Inputs

I_EMPTY = 0
I_MAX_DIST = 1
I_MIN_DIST = 2
I_FOOD = 3
I_OBSTACLE= 4
I_HEAD = 5
I_BODY = 6
I_TAIL = 7
I_DAZME = 8
NUM_INPUTS = 9

# Determine how far the snake sees these objects
SENSE_EMPTY = 0.5
SENSE_MAX_DIST = 1
SENSE_MIN_DIST = 1
SENSE_FOOD = 0.5
SENSE_OBSTACLE = 1
SENSE_HEAD = 1
SENSE_BODY = 0.5
SENSE_TAIL = 1
SENSE_DAZME = 0.5

mean_inputs = np.array([ [ (HEIGHT * WIDTH) / 10,
                           15,
                           10,
                           1,
                           1,
                           1,
                           2,
                           1.5,
                           2 ]] )

class Cell():
    def __init__(self):

        self.cell_north = None
        self.cell_south = None
        self.cell_west = None
        self.cell_east = None
        self.edges = None

        self.prev = None
        self.dist = 1

        self.empty = 1
        self.foods = 0
        self.obstacles = 0
        self.heads = 0
        self.body = 0
        self.tails = 0
        self.dazme = 0

        self.is_endpoint = False

        self.inputs = [ 0.0 ] * NUM_INPUTS
        
    def clear_tile(self):
        # Sums along shortest path to F, L, R
        self.empty = True
        self.food = False
        self.obstacle = False
        self.head = False
        self.body = False
        self.tail = False
        self.dazme = False
        self.is_endpoint = False

    def clear_sums(self):
        self.prev = None
        for i in range(NUM_INPUTS):
            self.inputs[i] = 0.0
            self.dist = 1

    def set_edges(self, edges):
        self.edges = edges

    def set_food(self):
        self.food = True

    def set_obstacle(self):
        self.obstacle = True
        self.empty = False
        self.is_endpoint = True

    def set_body(self, dazme = 0):
        self.body = True
        self.empty = False
        self.dazme = dazme
        self.is_endpoint = True

    def set_head(self, dazme = 0):
        self.head = True
        self.empty = False
        self.dazme = dazme
        self.is_endpoint = True

    def set_tail(self, dazme = 0):
        self.tail = True
        self.empty = False
        self.dazme = dazme
        self.is_endpoint = True

    def sum_up(self, cell):
        fade = 1.0 / cell.dist
        if self.empty:
            self.inputs[I_EMPTY] += SENSE_EMPTY * fade
        if self.food:
            self.inputs[I_FOOD]  += SENSE_FOOD * fade
        if self.obstacle:
            self.inputs[I_OBSTACLE] += SENSE_OBSTACLE * fade
        if self.head:
            self.inputs[I_HEAD] += SENSE_HEAD * fade
        if self.body:
            self.inputs[I_BODY] += SENSE_BODY * fade
        if self.tail:
            self.inputs[I_TAIL] += SENSE_TAIL * fade
        if self.dazme:
            self.inputs[I_DAZME] += SENSE_DAZME * fade

        self.inputs[I_MAX_DIST] = max(self.inputs[I_MAX_DIST], cell.dist)

class Snake(BaseSnake):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.name = "agentQ-ES"
        self.mlp = mlp
        self.result = None
        self.uid = 0
        self.cells = None

        self.width = WIDTH
        self.height = HEIGHT
        self.board_size = self.width * self.height

        self.count_inputs = 0
        self.sum_inputs = None

    def _set_edges(self, cell, x, y):
        edges = []
        
        # South
        if y < self.height-1:
            cell.cell_south = self.cells[x + (y+1) * self.width]
            edges.append(cell.cell_south)

        # North
        if y > 0:
            cell.cell_north = self.cells[x + (y-1) * self.width]
            edges.append(cell.cell_north)

        # West
        if x > 1:
            cell.cell_west = self.cells[x-1 + y * self.width]
            edges.append(cell.cell_west)

        # East
        if x < self.width-1:
            cell.cell_east = self.cells[x+1 + y * self.width]
            edges.append(cell.cell_east)

        cell.set_edges(edges)

    def _make_cells(self):
        self.cells = [ None ] * self.board_size

        for x in range(self.width):
            for y in range(self.height):
                self.cells[x + y*self.width] = Cell()

        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x + y*self.width]
                self._set_edges(cell, x, y)

    def _load_map(self, gmap: Map):
        if self.cells is None:
            self._make_cells()

        for cell in self.cells:
            cell.clear_tile()

        for snake in gmap.game_map['snakeInfos']:
            positions = snake['positions']
            dazme = 1 if snake['id'] == self.snake_id else 0
            self.cells[positions[0]].set_head(dazme)
            self.cells[positions[-1]].set_tail(dazme)

            for position in positions[1:-1]:
                self.cells[position].set_body(dazme)

        for position in gmap.game_map['obstaclePositions']:
            self.cells[position].set_obstacle()

        for position in gmap.game_map['foodPositions']:
            self.cells[position].set_food()

    def _compute_sums(self, start, gmap: Map):
        for cell in self.cells:
            cell.clear_sums()

        start.prev = start
        frontier = deque([ start ])

        while len(frontier):
            cell = frontier.popleft()
            start.sum_up(cell)

            for vertex in cell.edges:
                if vertex.prev is None:
                    vertex.prev = cell
                    vertex.dist = cell.dist + 1
                    if not vertex.is_endpoint:
                        frontier.append(vertex)

                
    def _get_q_value(self, cell, gmap: Map):

        self._compute_sums(cell, gmap)
        
        inputs = np.array([ cell.inputs ])

        inputs /= mean_inputs

        q_value = self.mlp.recall(inputs)

        return q_value

#    @overrides
    def get_next_action(self, gmap: Map):
        self._load_map(gmap)
        
        myself = gmap.get_snake_by_id(self.snake_id)['positions']
        head = self.cells[myself[0]]

        current_direction = self.get_current_direction()

        cell_l, cell_f, cell_r = [ None, None, None ]
        
        if current_direction == Direction.UP:
            cell_l, cell_f, cell_r = head.cell_west, head.cell_north, head.cell_east
        elif current_direction == Direction.RIGHT:
            cell_l, cell_f, cell_r = head.cell_north, head.cell_east, head.cell_south
        elif current_direction == Direction.LEFT:
            cell_l, cell_f, cell_r = head.cell_south, head.cell_west, head.cell_north
        else:  # DOWN
            cell_l, cell_f, cell_r = head.cell_east, head.cell_south, head.cell_west

        output = np.array([ -1e10, -1e10, -1e10 ])
        
        if cell_l and not cell_l.is_endpoint:
            output[0] = self._get_q_value(cell_l, gmap)

        if cell_f and not cell_f.is_endpoint:
            output[1] = self._get_q_value(cell_f, gmap)

        if cell_r and not cell_r.is_endpoint:
            output[2] = self._get_q_value(cell_r, gmap)

        action = [Action.LEFT, Action.FRONT, Action.RIGHT][output.argmax()]

        return action        


class GeneticSnake(GeneticThing, Snake):
    def __init__(self, r_mutation=0.4, severity=0.05):
        self.uid = None

        # deep snake
        mlp = MLP(NUM_INPUTS, activation="tanh", output="tanh")
        mlp.add_layer(16)
        mlp.add_layer(7)
        mlp.add_layer(1)
        Snake.__init__(self, mlp)

        self.p_alive = 0.0
        self._fitness = 0
        self._r_mutation = r_mutation
        self._severity = severity

    def store_snake(self):
        pass

    @property
    def num_parameters(self):
        N = 0
        for i in range(len(self.mlp.W)):
            w_shape = np.shape(self.mlp.W[i])
            N += w_shape[0] * w_shape[1]
        return N

    def get_parameters(self):
        x = []
        for l in range(len(self.mlp.W)):
            w_shape = np.shape(self.mlp.W[l])
            for j in range(w_shape[0]):
                for k in range(w_shape[1]):
                    x.append(self.mlp.W[l][j][k])
        return np.array(x)

    @property
    def fitness(self):
        return self._fitness

    def set_fitness(self, points, age, p_alive, food_rate):
        winner_bonus = np.power(2, p_alive)
        food_bonus = 1.0 + food_rate
        age_bonus = 1.0 + (age / 750.0)

        self._fitness = points * food_bonus * age_bonus * winner_bonus

    def mutate(self, r_mutation):
        # Add noise to weights. Noise proportional to 0 < r_mutate < 1
        sigma = r_mutation * self._severity
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] = self._get_randomization(self.mlp.W[i], sigma)

    def _get_randomization(self, w, sigma):
        return np.random.normal(w, sigma)
    
    def crosswith(self, that, p_inherit):
        offspring = deepcopy(self)
        offspring.uid = None
        for i in range(len(offspring.mlp.W)):
            w_shape = np.shape(offspring.mlp.W[i])
            mutations = random.randint(1, w_shape[0] * w_shape[1] - 1)
            mutations = int(p_inherit * w_shape[0] * w_shape[1])
            for j in range(mutations):
                k = 0 if w_shape[0] == 1 else random.randint(1, w_shape[0] -1)
                l = 0 if w_shape[1] == 1 else random.randint(1, w_shape[1] -1)
                offspring.mlp.W[i][k][l] = offspring.mlp.W[i][k][l] *  (1-p_inherit) + p_inherit * that.mlp.W[i][k][l]
            # offspring.mlp.W[i] = p_inherit * offspring.mlp.W[i] + (1-p_inherit) * that.mlp.W[i]

        offspring._fitness = self.fitness * (1-p_inherit) + p_inherit * that.fitness
        return offspring

    def distanceto(self, that):
        d = 0
        for i in range(len(self.mlp.W)):
            d += np.sum(np.power(self.mlp.W[i] - that.mlp.W[i], 2))
        return d
    
    def add(self, that):
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] += that.mlp.W[i]

    def subtract(self, that):
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] -= that.mlp.W[i]

    def divide_by(self, divisor):
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] /= divisor

#    @overrides
    def on_game_result(self, player_ranks):
        for player in player_ranks:
            if player['playerName'] == self.name:
                self.is_alive = player['alive']
                self.points = player['points']
                # rank = player['rank']

        self.result = self.uid, self.points, self.age, self.is_alive, self.watch_link
