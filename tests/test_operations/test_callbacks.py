# --------------------------------------------------------------------------- #
#                            TEST CALLBACKS                                   #
# --------------------------------------------------------------------------- #
# %%
import datetime
import math
import numpy as np
import pytest
from pytest import mark

from ml_studio.supervised_learning.training.monitor import History
from ml_studio.supervised_learning.training.metrics import RegressionMetrics


class HistoryTests:

    @mark.history
    def test_history_on_train_begin(self, get_history):
        history = get_history
        history.on_train_begin()
        assert history.total_epochs == 0, "total_epochs is not zero"
        assert history.total_batches == 0, "total_batches is not zero"
        assert isinstance(
            history.start, datetime.datetime), "start is not a datetime object"        
        assert isinstance(history.epoch_log,
                          dict), "history.epoch_log is not a dictionary object"
        assert isinstance(history.batch_log,
                          dict), "history.batch_log is not a dictionary object"
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
