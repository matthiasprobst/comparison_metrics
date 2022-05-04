import unittest

from comparison_metrics import metrics


class TestMetrics(unittest.TestCase):

    def test_metric_class_setup(self):
        for metric in metrics.metric_dict:
            print(f'\nChecking metric {metric}')
            _metric = metrics.metric_dict[metric]
            self.assertTrue(_metric.is_fully_defined)
            try:
                _metric().get_bibtex()
            except KeyError:
                self.assertTrue(False)
