import numpy as np
from deap import algorithms, base, creator, gp, tools


class DeapFactor:
    def __init__(self, pset, fitness):
        self.pset = pset
        self.fitness = fitness

    def set_gp_params(self):
        # 创建适应度类和个体类，create(类名，继承的类，参数)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 适应度最大化
        creator.create(
            "Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset
        )  # 个体

        # 设置遗传算法的基础配置，register(函数别名，函数，函数需要的参数)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self.toolbox.expr,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        # 适应度计算
        np.random.seed(42)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def set_evaluator(self):
        self.toolbox.register("evaluate", self.fitness)

    def set_pop_size(self, pop_size: int):
        self.pop_size = pop_size
        self.pop = self.toolbox.population(n=self.pop_size)

    def set_stats(self):
        self.hof = tools.HallOfFame(5)  # 保存最优个体
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"

    def set_input(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def run(self, pop_size: int = 300, ngen: int = 40):
        self.set_gp_params()
        self.set_evaluator()
        self.set_pop_size(pop_size)
        self.set_stats()
        self.set_input()

        self.pop, self.logbook = algorithms.eaSimple(
            self.pop,
            self.toolbox,
            cxpb=0.5,
            mutpb=0.1,
            ngen=ngen,
            stats=self.stats,
            halloffame=self.hof,
            verbose=True,
        )
