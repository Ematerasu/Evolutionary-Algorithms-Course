import random
import operator
import gymnasium as gym
import numpy as np
from deap import base, creator, tools, gp

# Inicjalizacja środowiska

env = gym.make("MountainCarContinuous-v0", render_mode=None)

def safe_add(x, y):
    return np.clip(x + y, -1e6, 1e6)

def safe_sub(x, y):
    return np.clip(x - y, -1e6, 1e6)

def safe_mul(x, y):
    return np.clip(x * y, -1e6, 1e6)

def safe_div(x, y):
    return np.clip(x / y if y != 0 else 0, -1e6, 1e6)


pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(safe_add, 2)
pset.addPrimitive(safe_sub, 2)
pset.addPrimitive(safe_mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
pset.renameArguments(ARG0="position", ARG1="velocity")

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # maksymalizujemy reward
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)  # bawimy sie tree-based GP

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4) # drzewa maksymalnie glebokosci 4
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset) # zeby potem dostac funkcje do ewaluacji

def evaluate(individual):
    program = toolbox.compile(expr=individual)
    total_reward = 0
    obs, _ = env.reset()
    for _ in range(200):
        position, velocity = obs
        action = np.clip(np.array([program(position, velocity)]), -1, 1)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint) # wymiana poddrzew jako crossover
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) # mutacja poprzez tworzenie poddrzew

# ograniczenie wysokosci nowych drzew
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

def main():
    population = toolbox.population(n=100) 
    ngen = 20  # generations
    cxpb = 0.5  # crossover ppb
    mutpb = 0.2  # mutation ppb

    for gen in range(ngen):
        offspring = tools.selTournament(population, len(population), tournsize=3)
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        print(f"Pokolenie {gen}: Najlepszy fitness: {best_ind.fitness.values[0]}")

    best_ind = tools.selBest(population, 1)[0]
    program = toolbox.compile(expr=best_ind)
    input("Proceed... ")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    done = False
    total_reward = 0
    obs, _ = env.reset()
    while not done:
        position, velocity = obs
        action = np.array([program(position, velocity)])
        print(action)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()
