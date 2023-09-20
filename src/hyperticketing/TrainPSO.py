# !/usr/bin/env python
# coding: utf-8


from __future__ import division
import pandas as pd
import numpy as np
import random
import os
import json
from BasePath import base_path
from hyperticketing.EventGroupEval import EventGroupEval
from hyperticketing.EventGroupEval import cost_function
from sklearn.preprocessing import MinMaxScaler

random.seed(0)


class Particle:
    """
    : Particle object for PSO algorithm
    """

    def __init__(self, x0, pso_params, num_dimensions):

        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best individual position
        self.err_best_i = -1  # best individual error
        self.err_i = -1  # individual error
        self.w = pso_params[0]
        self.c1 = pso_params[1]
        self.c2 = pso_params[2]
        self.num_dimensions = num_dimensions

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

        self.window = None
        self.spatial_filter = None
        return

    def evaluate(self, beta, min_max_range, scaler, perc_penalty, association_event_df):
        """
        : Evaluate fitness
        """
        self.get_filters(scaler, min_max_range)
        self.err_i = cost_function(self, beta, perc_penalty, association_event_df)

        # update individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i
        return

    def update_velocity(self, pos_best_g):
        """
        : Update new particle velocity
        """

        w = self.w  # constant inertia
        c1 = self.c1  # cognitive constant
        c2 = self.c2  # social constant

        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social
        return

    def update_position(self, bounds):
        """
        : Update particle position
        """

        for i in range(0, self.num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # update min and max position
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]
        return

    def get_filters(self, scaler, min_max_range):
        """
        : Update window and spatial filter
        """
        filters = ['ip_group', 'location', 'city_code']
        self.window = scaler.inverse_transform(np.array(self.position_i[0]).reshape(1, -1))[0][0]
        self.spatial_filter = filters[int((abs(np.array(self.position_i[1])) - 0.0001) / min_max_range * 3)]
        return


class PSO:
    """
    : Particle swarm optimizer
    """

    def __init__(self, config):

        self.config = config

        association_event_df = pd.read_csv(config.train["filepaths"]["master_df_name"])

        time_range = config.train["parameters"]["range_of_time"]
        time_range = np.arange(time_range[0], time_range[1], time_range[2])
        scaler = MinMaxScaler(feature_range=(-1 * config.train["parameters"]["min_max"], config.train["parameters"]["min_max"]))
        scaler.fit_transform(time_range.reshape(-1, 1))

        beta = config.train["parameters"]["eval_beta"]
        x0 = config.train["parameters"]["init_pos"]
        pso_params = config.train["parameters"]["pso_params"]
        bounds = [config.train["parameters"]["pso_bound"]['1st dim'], config.train["parameters"]["pso_bound"]['2nd dim']]
        min_max_range = config.train["parameters"]["min_max"]
        perc_penalty = config.train["parameters"]["penalty"]
        num_particles = config.train["parameters"]["num_par"]
        maxiter = config.train["parameters"]["max_iteration"]

        num_dimensions = len(x0)
        err_best_g = -1
        pos_best_g = []

        # Save best result for all iterations
        err_best_all = []
        pos_best_all = []

        # uniform initialization of particles
        x = [np.linspace(bounds[i][0], bounds[i][1], num=num_particles) for i in range(num_dimensions)]
        x = list(zip(*x))

        # establish the swarm
        swarm = [Particle(x[i], pso_params, num_dimensions) for i in range(num_particles)]

        # begin optimization loop
        for i in range(maxiter):
            print(i, 'th iteration:')
            for particle in swarm:
                particle.evaluate(beta, min_max_range, scaler, perc_penalty, association_event_df)

                if particle.err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(particle.position_i)
                    pos_best = [particle.window, particle.spatial_filter]
                    err_best_g = float(particle.err_i)

            for particle in swarm:
                particle.update_velocity(pos_best_g)
                particle.update_position(bounds)

            err_best_all.append(err_best_g)
            pos_best_all.append(pos_best)
            print('Best pos:\nwindow: ', pos_best[0], ', spatial_filter:', pos_best[1])
            print("Final Score: {}\n".format(1.0 / err_best_g))

        # Save pso parameters
        self.pso_dict = {
            'err_best_g': err_best_all,
            'window': pos_best[0],
            'spatial_filter': pos_best[1],
            'beta': self.config.train["parameters"]["eval_beta"]
        }

        self.event_swarm_obj = EventGroupEval(association_event_df,
                                              eval_beta=config.train["parameters"]["eval_beta"],
                                              rolling_window=pos_best[0],
                                              spatial_filter=pos_best[1])
        return

    def save_files(self, path):
        """
        : Save pso output
        """

        # make folder for output
        output_folder_name = 'pso_filters_run'
        output_folder_path = path + '' + output_folder_name
        os.makedirs(output_folder_path)
        os.chdir(output_folder_path)
        print('All model files will be saved in ', os.getcwd())

        config_file = 'pso_filter.json'
        f = open(config_file, 'w')
        f.write(json.dumps(self.pso_dict))
        f.close()

        # save the swarm labels
        self.event_swarm_obj.group_df.to_csv('swarm_label.csv', header=True, index=False)
        os.chdir(base_path)
        return output_folder_path