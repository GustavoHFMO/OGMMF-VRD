"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Cumulative Sum (CUSUM) Method Implementation ***
Paper: Page, Ewan S. "Continuous inspection schemes."
Published in: Biometrika 41.1/2 (1954): 100-115.
URL: http://www.jstor.org/stable/2333009
"""

from detectors.super_detector import SUPER_DETECTOR

class CUSUM(SUPER_DETECTOR):
    """The Cumulative Sum (CUSUM) drift detection method class."""
    def __init__(self, min_instance=30, delta=0.005, lambda_=50):

        super().__init__()

        self.MINIMUM_NUM_INSTANCES = min_instance

        self.m_n = 1
        self.x_mean = 0
        self.sum = 0
        self.delta = delta
        self.lambda_ = lambda_

    def run(self, pr):

        pr = 1 if pr is False else 0

        warning_status = False
        drift_status = False

        # 1. UPDATING STATS
        self.x_mean = self.x_mean + (pr - self.x_mean) / self.m_n
        self.sum = max([0, self.sum + pr - self.x_mean - self.delta])
        self.m_n += 1

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.m_n >= self.MINIMUM_NUM_INSTANCES:
            if self.sum > self.lambda_:
                drift_status = True

        return warning_status, drift_status

    def reset(self):
        self.m_n = 1
        self.x_mean = 0
        self.sum = 0
