import logging
import numpy as np
import pickle
# from overrides import overrides

from simulation_controller import SimulationController
from ga import GeneticAlgorithm
from snake import GeneticSnake

log = logging.getLogger("genetic_controller")

class GeneticController(SimulationController):
    def __init__(self, args):
        super().__init__(args)

        self._name = "q_MX_minfit_p15"

        self.max_uid = 0
        self._num_simulations = 5

        population_size = 30
        
        self.ga = GeneticAlgorithm(population_size,
                                   r_mutation = 0.05,
                                   apex_stddev = 0.25)
        # self.ga.generation = 60
        self.result_map = {}
        self.points_map = {}
        self.alive_map = {}

        # population_size = args.population_size
        # args.r_mutation = 0.4

        self._generation_info = []

        for i in range(self.ga.population_size):
            self._add_snake()

        self._load_genomes()

    def _get_num_simulations(self):
        num_simulations = int(self._num_simulations * (1.0 - np.exp(-0.05 * self.ga.generation)))
        if num_simulations < 1:
            num_simulations = 1
        return num_simulations

    def _add_snake(self):
        snake = GeneticSnake(severity = 0.12)
        self.ga.append(snake)

    def _create_batch(self):
        uid = 0
        batch = []                
        for snake in self.ga.population:
            snake.uid = uid
            self.result_map[uid] = (1e4, 0.0, 1e4, 0.0)
            self.points_map[uid] = []
            self.alive_map[uid] = []
            uid += 1
            for s in range(self._get_num_simulations()):
                batch.append(snake)
        

        self.max_uid = uid
        return batch
        
    def initial_batch(self):
        return self._create_batch()

    def create_batch_from_results(self, result_generator):
        # Show the results
        count = 0

        print('[Generation #%d] Results:' % self.ga.generation)
        for uid, points, age, is_alive, watch_link in result_generator:

            self.points_map[uid].append(points)
            self.alive_map[uid].append(0 + is_alive)
            
            count += 1

            if age < 1:
                age = 1
            if points < 1:
                points = 1

            self.result_map[uid] = ( min(self.result_map[uid][0], points),
                                     self.result_map[uid][1] + (0.0 + is_alive),
                                     min(self.result_map[uid][2], age),
                                     self.result_map[uid][3] + ((points - age/3.0)/age) )
            
            print('- #%3d %3s %3d => %s' % (uid, "W" if is_alive else "L", points, watch_link))

        N = self._get_num_simulations()
        
        for snake in self.ga.population:
            points, lives, age, food_rate = self.result_map[snake.uid]

            snake.set_fitness(points,
                              age,
                              lives / N,
                              food_rate / N)
            print('- #%3d %5g, food: %3.2f, lives: %.2f, fitness: %3.2f' % (snake.uid, points, 1.0 + food_rate, lives/N, snake.fitness))

        # Evolve the population and go on with the learning

        self._store_generation_info()
        self._store_distribution()
        
        self.ga.evolve()

        self._store_genomes()

        batch = self._create_batch()

        return batch

    def _store_distribution(self):
        xs = np.array([ self.ga.population[0].get_parameters() ])

        for thing in self.ga.population[1:]:
            x = np.array([thing.get_parameters()])
            xs = np.append(xs, x, axis=0)

        fitness = [ thing.fitness for thing in self.ga.population ]
        sum_fitness = sum(fitness)
        weights = np.array(fitness) / sum_fitness

        print(np.shape(xs))
        print(weights)

        mean = np.mean(xs, axis=0)
        deviations = xs - mean

        weighted_mean = np.average(xs, weights=weights, axis=0)

        C = np.cov(np.transpose(deviations), aweights=weights)

        dist = [ weighted_mean, C]

        filename = "ga_dist_" + self._name + ".obj"
        dump_file = open(filename, "wb")
        pickle.dump(dist, dump_file)
        dump_file.close()
    
    def _store_generation_info(self):
        fitnesses = []
        points = []
        alive = []

        for snake in self.ga.population:
            fitnesses.append(snake.fitness)
            points.append(np.array(self.points_map[snake.uid]))
            alive.append(np.array(self.alive_map[snake.uid]))
            
        self._generation_info.append([ fitnesses, points, alive ])

        dump_file = open(self._name + "_dump.obj", "wb")
        pickle.dump(self._generation_info, dump_file)
        dump_file.close()
        
    def _store_genomes(self):
        genomes = []
        for i in self.ga.population:
            weights = []
            for W in i.mlp.W:
                weights.append(W)

            genomes += [ weights ]

        np.save("population.npy", np.array(genomes))

    def _load_genomes(self):

        try:
            genomes = np.load("population.npy")
            i = 0
            for g in genomes:
                if i >= len(self.ga.population):
                    self._add_snake()
                snake = self.ga.population[i]
                i += 1
                j = 0
                for W in g:
                    snake.mlp.W[j] = W
                    j += 1
        except IOError as e:
            print(e)
