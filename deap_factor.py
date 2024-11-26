import operator
import random

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
        random.seed(42)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )

        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

    def set_evaluator(self):
        self.toolbox.register(
            "evaluate",
            self.fitness,
            toolbox=self.toolbox,
            X_train=self.X_train,
            y_train=self.y_train,
        )

    def set_pop_size(self, pop_size: int):
        self.pop_size = pop_size
        self.pop = self.toolbox.population(n=self.pop_size)

    def set_stats(self):
        self.hof = tools.HallOfFame(5)  # 保存最优个体
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.stats.register("avg", np.nanmean)
        self.stats.register("std", np.nanstd)
        self.stats.register("min", np.nanmin)
        self.stats.register("max", np.nanmax)

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"

    def set_input(self, data, label):
        setattr(self, label, data)

    def run(self, pop_size: int = 300, ngen: int = 40):
        self.set_gp_params()
        self.set_evaluator()
        self.set_pop_size(pop_size)
        self.set_stats()

        self.pop, self.logbook = algorithms.eaMuPlusLambda(
            self.pop,
            self.toolbox,
            mu=pop_size,
            lambda_=pop_size,
            cxpb=0.5,
            mutpb=0.1,
            ngen=ngen,
            stats=self.stats,
            halloffame=self.hof,
            verbose=True,
        )
