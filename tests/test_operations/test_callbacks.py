# --------------------------------------------------------------------------- #
#                            TEST CALLBACKS                                   #
# --------------------------------------------------------------------------- #
# %%
import datetime
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.operations.callbacks import History, Benchmark
from ml_studio.operations.metrics import Scorer


class HistoryTests:

    @mark.history
    def test_history_on_train_begin(self, get_history):
        history = get_history
        history.on_train_begin()
        assert history.total_epochs == 0, "total_epochs is not zero"
        assert history.total_batches == 0, "total_batches is not zero"
        assert isinstance(
            history.start, datetime.datetime), "start is not a datetime object"
        assert isinstance(history.epochs, list), "history.epochs is not a list"
        assert isinstance(history.epoch_log,
                          dict), "history.epoch_log is not a dictionary object"
        assert isinstance(history.batch_log,
                          dict), "history.batch_log is not a dictionary object"
        assert len(history.epochs) == 0, "history.epochs has a non-zero length"
        assert len(
            history.epoch_log) == 0, "history.epoch_log has a non-zero length"
        assert len(
            history.batch_log) == 0, "history.batch_log has a non-zero length"

    @mark.history
    def test_history_on_batch_end(self, get_history):
        history = get_history
        history.on_train_begin()

        batch = 1
        batch_size = 32
        theta = np.random.normal(size=10)
        train_cost = 100 * np.random.random_sample((1))
        batch_log = {'batch': batch, 'batch_size': batch_size,
                     'theta': theta, 'train_cost': train_cost}
        history.on_batch_end(batch, batch_log)
        assert history.total_batches == 1, "total_batches 1st iteration not equal 1"
        assert history.batch_log['batch'][0] == 1, "batch number 1st iteration not 1"
        assert history.batch_log['batch_size'][0] == 32, "batch_size not correct"
        assert history.batch_log['theta'][0].shape == (
            10,), "theta shape not correct"
        assert isinstance(
            history.batch_log['theta'][0], (list, np.ndarray)), "theta is not a list or ndarray"
        assert history.batch_log['train_cost'][0] == train_cost, "train_cost not valid"

        batch = 2
        batch_size = 32
        theta = np.random.normal(size=10)
        train_cost = 100 * np.random.random_sample((1))
        batch_log = {'batch': batch, 'batch_size': batch_size,
                     'theta': theta, 'train_cost': train_cost}
        history.on_batch_end(batch, batch_log)
        assert history.total_batches == 2, "total_batches 1st iteration not equal 1"
        assert history.batch_log['batch'][1] == 2, "batch number 1st iteration not 1"
        assert history.batch_log['batch_size'][1] == 32, "batch_size not correct"
        assert history.batch_log['theta'][1].shape == (
            10,), "theta shape not correct"
        assert isinstance(
            history.batch_log['theta'][1], (list, np.ndarray)), "theta is not a list or ndarray"
        assert history.batch_log['train_cost'][1] == train_cost, "train_cost not valid"

    @mark.history
    def test_history_on_epoch_end_w_validation(self, get_history):
        # Evaluate batches
        history = get_history
        history.on_train_begin()
        total_costs = 0
        for i in np.arange(1, 11):
            batch = i
            batch_size = 32
            theta = np.random.normal(size=10)
            train_cost = 100 * np.random.random_sample((1))
            total_costs += train_cost
            batch_log = {'batch': batch, 'batch_size': batch_size,
                         'theta': theta, 'train_cost': train_cost}            
            history.on_batch_end(batch, batch_log)
        assert history.total_batches == 10, "number of batches is incorrect"
        assert np.sum(history.batch_log['batch']) == np.sum(
            np.arange(1, 11)), "batch number list incorrect"
        assert len(history.batch_log['batch']) == 10, "batch number is wrong shape"
        assert np.sum(history.batch_log['batch_size']) == np.sum(
            np.repeat(32, 10)), "batch size list incorrect"
        assert len(history.batch_log['batch_size']) == 10, "batch size is wrong shape"
        assert len(history.batch_log['theta']) == 10, "theta is wrong length"
        assert np.isclose(np.sum(history.batch_log['train_cost']), total_costs[0], 10**4), "train costs don't sum"

        assert len(history.batch_log['train_cost']) ==10, "train_cost not correct shape"
        assert isinstance(
            history.batch_log['batch'], list), "batch number is not a list"
        assert isinstance(
            history.batch_log['batch_size'], list), "batch size is not a list"
        assert isinstance(
            history.batch_log['theta'], list), "theta is not an ndarray"
        for theta in history.batch_log['theta']:
               assert isinstance(theta, np.ndarray), "thetas are not np.ndarrays"
        assert isinstance(
            history.batch_log['train_cost'], list), "train_cost is not a list"
        # Evaluate epochs
        epoch = 1
        train_cost = 1000*np.random.random_sample((1))
        train_score = 1000*np.random.random_sample((1))
        val_cost = 1000*np.random.random_sample((1))
        val_score = 1000*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'epoch':epoch, 'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        history.on_epoch_end(epoch, logs=log)
        assert history.epochs[0] == 1, "epochs is not 1 on first iteration"
        assert history.epoch_log['train_cost'][0] == train_cost, "train_cost 1st iteration not correct"
        assert history.epoch_log['train_score'][0] == train_score, "train_score 1st iteration not correct"
        assert history.epoch_log['val_cost'][0] == val_cost, "val_cost 1st iteration not correct"
        assert history.epoch_log['val_score'][0] == val_score, "val_score 1st iteration not correct"
        assert (history.epoch_log['theta'][0]==theta).all(), "theta 1st iteration not correct"
        print(history)
        assert isinstance(
            history.epoch_log['epoch'], list), "epochs is not a list"
        assert isinstance(
            history.epoch_log['train_cost'], list), "train_cost is not a list"
        assert isinstance(
            history.epoch_log['train_score'], list), "train_score is not a list"
        assert isinstance(
            history.epoch_log['val_cost'], list), "val_cost is not a list"
        assert isinstance(
            history.epoch_log['val_score'], list), "val_score is not a list"

        epoch = 2
        train_cost = 1000*np.random.random_sample((1))
        train_score = 1000*np.random.random_sample((1))
        val_cost = 1000*np.random.random_sample((1))
        val_score = 1000*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'epoch':epoch, 'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        history.on_epoch_end(epoch, logs=log)
        assert history.epoch_log['epoch'][1] == 2, "epochs is not 1 on first iteration"
        assert history.epoch_log['train_cost'][1] == train_cost, "train_cost 1st iteration not correct"
        assert history.epoch_log['train_score'][1] == train_score, "train_score 1st iteration not correct"
        assert history.epoch_log['val_cost'][1] == val_cost, "val_cost 1st iteration not correct"
        assert history.epoch_log['val_score'][1] == val_score, "val_score 1st iteration not correct"
        assert (history.epoch_log['theta'][1]==theta).all(),  "theta 2nd iteration not correct"
        assert len(history.epoch_log['epoch']) == 2, "epochs shape not correct on second iteration"
        assert len(history.epoch_log['train_cost']) == 2, "train_cost length not correct on second iteration"
        assert len(history.epoch_log['train_score']) == 2, "train_score shape not correct on second iteration"
        assert len(history.epoch_log['val_cost']) == 2, "val_cost shape not correct on second iteration"
        assert len(history.epoch_log['val_score']) == 2, "val_score shape not correct on second iteration"

    @mark.history
    def test_history_on_train_end(self, get_history):
        history = get_history
        history.on_train_begin()
        history.on_train_end()
        assert isinstance(
            history.end, datetime.datetime), "end is not a datetime object"
        assert isinstance(history.duration, float), "duration is not a float"


class BenchmarkTests:

    @mark.benchmark
    def test_benchmark_on_train_begin(self, get_benchmark):
        # Test when cost is being monitored
        params = {'monitor': 'train_cost', 'metric': 'root_mean_squared_error'}
        benchmark = get_benchmark
        benchmark.set_params(params)
        benchmark.on_train_begin()
        assert benchmark._monitor == 'train_cost', "monitor parameter not correct"
        assert benchmark._metric == 'root_mean_squared_error', "metric parameter not correct"        
        assert isinstance(benchmark.best_model,
                          dict), "best_model not a dictionary object"
        assert benchmark.best_model['monitor'] == 'train_cost', "best_model monitor not valid"
        assert benchmark.best_model['metric'] == 'root_mean_squared_error', "best_model metric not valid"
        assert benchmark._better == np.less, "_better function is not correct"
        assert benchmark._best_performance == np.Inf, "_best performance is not correct"

        # Test when score is being monitored
        params = {'monitor': 'train_score', 'metric': 'neg_mean_squared_error'}
        benchmark = get_benchmark
        benchmark.set_params(params)
        benchmark.on_train_begin()
        assert benchmark._monitor == 'train_score', "monitor parameter not correct"
        assert benchmark._metric == 'neg_mean_squared_error', "metric parameter not correct"
        assert isinstance(benchmark.best_model,
                          dict), "best_model not a dictionary object"
        assert benchmark.best_model['monitor'] == 'train_score', "best_model monitor not valid"
        assert benchmark.best_model['metric'] == 'neg_mean_squared_error', "best_model metric not valid"
        assert benchmark._better == np.greater, "_better function is not correct"
        assert benchmark._best_performance == - \
            np.Inf, "_best performance is not correct"

    @mark.benchmark
    def test_benchmark_on_epoch_end_monitor_cost(self, get_benchmark):
        # Test monitor train cost
        params = {'monitor': 'train_cost', 'metric': 'root_mean_squared_error'}
        benchmark = get_benchmark
        benchmark.set_params(params)
        benchmark.on_train_begin()

        epoch = 1
        train_cost = 1000*np.random.random_sample((1))
        train_score = 1000*np.random.random_sample((1))
        val_cost = 1000*np.random.random_sample((1))
        val_score = 1000*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        benchmark.on_epoch_end(epoch=epoch, logs=log)
        assert benchmark._best_performance == train_cost, "train_cost incorrect 1st iteration"
        assert benchmark.best_model['epoch'] == 1, "best_model[epoch] not equal 1 on 1st iteration"
        assert benchmark.best_model['performance'] == train_cost, "best_model[performance] not correct on 1st iteration"
        assert (benchmark.best_model['theta']==theta).all(), "best_model[theta] not correct on 1st iteration"

        epoch = 2
        train_cost = train_cost * 1.1
        train_score = train_score * 1.1
        val_cost = val_cost * 1.1
        val_score = val_score * 1.1
        theta = np.random.normal(size=10)
        log = {'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        benchmark.on_epoch_end(epoch=epoch, logs=log)
        assert benchmark._best_performance != train_cost, "train_cost incorrect 2nd iteration"
        assert benchmark.best_model['epoch'] == 1, "best_model[epoch] not equal 1 on 2nd iteration"
        assert benchmark.best_model['performance'] != train_cost, "best_model[performance] not correct on 2nd iteration"
        assert (benchmark.best_model['theta'] != theta).all(), "best_model[theta] not correct on 2nd iteration"

        epoch = 3
        train_cost = 100*np.random.random_sample((1))
        train_score = 100*np.random.random_sample((1))
        val_cost = 100*np.random.random_sample((1))
        val_score = 100*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        benchmark.on_epoch_end(epoch=epoch, logs=log)
        assert benchmark._best_performance == train_cost, "train_cost incorrect 3rd iteration"
        assert benchmark.best_model['epoch'] == 3, "best_model[epoch] not equal 1 on 3rd iteration"
        assert benchmark.best_model['performance'] == train_cost, "best_model[performance] not correct on 3rd iteration"
        assert (benchmark.best_model['theta']==theta).all(), "best_model[theta] not correct on 3rd iteration"

    @mark.benchmark
    def test_benchmark_on_epoch_end_monitor_score(self, get_benchmark):
        # Test monitor train cost
        params = {'monitor': 'train_score', 'metric': 'neg_mean_squared_error'}
        benchmark = get_benchmark
        benchmark.set_params(params)
        benchmark.on_train_begin()

        epoch = 1
        train_cost = -1000*np.random.random_sample((1))
        train_score = -1000*np.random.random_sample((1))
        val_cost = -1000*np.random.random_sample((1))
        val_score = -1000*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        benchmark.on_epoch_end(epoch=epoch, logs=log)
        assert benchmark._best_performance == train_score, "train_score incorrect 1st iteration"
        assert benchmark.best_model['epoch'] == 1, "best_model[epoch] not equal 1 on 1st iteration"
        assert benchmark.best_model['performance'] == train_score, "best_model[performance] not correct on 1st iteration"
        assert (benchmark.best_model['theta']==theta).all(), "best_model[theta] not correct on 1st iteration"

        epoch = 2
        train_cost = -2000*np.random.random_sample((1))
        train_score = -2000*np.random.random_sample((1))
        val_cost = -2000*np.random.random_sample((1))
        val_score = -2000*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        benchmark.on_epoch_end(epoch=epoch, logs=log)
        assert benchmark._best_performance != train_score, "train_score incorrect 2nd iteration"
        assert benchmark.best_model['epoch'] == 1, "best_model[epoch] not equal 1 on 2nd iteration"
        assert benchmark.best_model['performance'] != train_score, "best_model[performance] not correct on 2nd iteration"
        assert (benchmark.best_model['theta'] != theta).all(), "best_model[theta] not correct on 2nd iteration"

        epoch = 3
        train_cost = 100*np.random.random_sample((1))
        train_score = 100*np.random.random_sample((1))
        val_cost = 100*np.random.random_sample((1))
        val_score = 100*np.random.random_sample((1))
        theta = np.random.normal(size=10)
        log = {'train_cost': train_cost, 'train_score': train_score,
               'val_cost': val_cost, 'val_score': val_score, 'theta': theta}
        benchmark.on_epoch_end(epoch=epoch, logs=log)
        assert benchmark._best_performance == train_score, "train_score incorrect 3rd iteration"
        assert benchmark.best_model['epoch'] == 3, "best_model[epoch] not equal 1 on 3rd iteration"
        assert benchmark.best_model['performance'] == train_score, "best_model[performance] not correct on 3rd iteration"
        assert (benchmark.best_model['theta']==theta).all(), "best_model[theta] not correct on 3rd iteration"
