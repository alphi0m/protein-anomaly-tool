
import numpy as np
import ruptures as rpt

def calculate_variable_thresholds(errors, pen=1):
    errors = np.array(errors).reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=10)
    algo.fit(errors)
    result = algo.predict(pen=pen)

    variable_thresholds = np.zeros_like(errors)
    start = 0
    for cp in result:
        segment = errors[start:cp]
        threshold = np.mean(segment) + 3 * np.std(segment)
        variable_thresholds[start:cp] = threshold
        start = cp

    return variable_thresholds.flatten(), result
