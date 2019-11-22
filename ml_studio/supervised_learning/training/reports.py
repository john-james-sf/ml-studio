# %%
# =========================================================================== #
#                             PRINT UTILITIES                                 #
# =========================================================================== #
import datetime
import numpy as np
import types

center = 25

def summary(history):
    """Summarizes statistics for model.

    Parameters
    ----------
    history : history object
        history object containing data and statistics from training.
    """
    monitor = history.params.get('monitor')
    metric = history.params.get('metric', "")

    print("\nOptimization Summary")
    print("                  Name: " + history.params.get('name'))
    print("                 Start: " + str(history.start))
    print("                   End: " + str(history.end))
    print("              Duration: " + str(history.duration) + " seconds.")
    print("                Epochs: " + str(history.total_epochs))
    print("               Batches: " + str(history.total_batches))
    print("\n")
    print("   Final Training Loss: " +
          str(history.epoch_log.get('train_cost')[-1]))
    if 'score' in monitor:
        print("  Final Training Score: " + str(history.epoch_log.get('train_score')[-1])
              + " " + history.params.get('metric'))
    if history.early_stop:
        print(" Final Validation Loss: " +
              str(history.epoch_log.get('val_cost')[-1]))
        if history.metric:
            print("Final Validation Score: " + str(history.epoch_log.get('val_score')[-1])
                  + " " + metric)
    print("         Final Weights: " + str(history.epoch_log.get('theta')[-1]))
    print("\nModel Parameters")

    for p, v in history.params.items():
        label_length = len(p)
        spaces = center - label_length
        if isinstance(v, (str, bool, int, list, np.ndarray, types.FunctionType, float)) \
                or v is None:
            p = " " * spaces + p + ": "
            # If v is a function type, it is the lambda function that
            # initializes the regularizer to zeros if the parameter
            # is None. If this is the case, we'll print "None" for
            # this parameter.
            if isinstance(v, types.FunctionType):
                v = None
            print(p + str(v))
        else:
            _recur(p, v)


def _recur(callable_type, callable_object):
    callable_name = callable_object.name
    spaces = center - len(callable_type)
    callable_type = " " * spaces + callable_type + ":"
    print(callable_type, callable_name)
    config = callable_object.get_params()
    if len(config) > 0:
        for k, v in config.items():
            if isinstance(v, (str, bool, int, list, np.ndarray, types.FunctionType, float)) \
                    or v is None:
                spaces = center - len(k)
                k = " " * spaces + k + ": "
                # If v is a function type, it is the lambda function that
                # initializes the regularizer to zeros if the parameter
                # is None. If this is the case, we'll print "None" for
                # this parameter.
                if isinstance(v, types.FunctionType):
                    v = None
                print(k + str(v))
            else:
                _recur(k, v)

        
