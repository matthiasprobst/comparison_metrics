import unittest

from comparson_metrics import metrics


class TestMetrics(unittest.TestCase):

    def test_metric(self):
        for metric in metrics.metric_dict:
            print(f'\nChecking metric {metric}')
            _metric = metrics.metric_dict[metric]
            self.assertTrue(_metric.is_fully_defined)
