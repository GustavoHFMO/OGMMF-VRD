"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Early Drift Detection Method (EDDM) Implementation ***
Paper: Baena-García, Manuel, et al. "Early drift detection method." (2006).
URL: http://www.cs.upc.edu/~abifet/EDDM.pdf
"""

import math
from detectors.super_detector import SUPER_DETECTOR

class EDDM(SUPER_DETECTOR):
    """The Early Drift Detection Method (EDDM) class."""
    
    def __init__(self, min_instance=30, C=1, W=0.5):
        
        
        self.OUT_CONTROL_LEVEL = C
        self.WARNING_LEVEL = W

        self.MINIMUM_NUM_INSTANCES = min_instance
        self.NUM_INSTANCES_SEEN = 0

        self.MINIMUM_NUM_ERRORS = 30
        self.NUM_ERRORS = 0

        self.P = 0.0  # mean
        self.S_TEMP = 0.0
        self.M2S_max = 0

        self.LATEST_E_LOCATION = 0
        self.SECOND_LATEST_E_LOCATION = 0

    def run(self, prediction_status):
        warning_status = False
        drift_status = False

        self.NUM_INSTANCES_SEEN += 1

        if prediction_status is False:

            self.NUM_ERRORS += 1

            self.SECOND_LATEST_E_LOCATION = self.LATEST_E_LOCATION
            self.LATEST_E_LOCATION = self.NUM_INSTANCES_SEEN
            distance = self.LATEST_E_LOCATION - self.SECOND_LATEST_E_LOCATION

            old_p = self.P
            self.P += (distance - self.P) / self.NUM_ERRORS
            self.S_TEMP += (distance - self.P) * (distance - old_p)

            s = math.sqrt(self.S_TEMP / self.NUM_ERRORS)
            m2s = self.P + 2 * s

            if self.NUM_INSTANCES_SEEN > self.MINIMUM_NUM_INSTANCES:
                if m2s > self.M2S_max:
                    self.M2S_max = m2s
                elif self.NUM_ERRORS > self.MINIMUM_NUM_ERRORS:
                    r = m2s / self.M2S_max
                    if r < self.WARNING_LEVEL:
                        warning_status = True
                    if r < self.OUT_CONTROL_LEVEL:
                        drift_status = True

        return warning_status, drift_status

    def reset(self):
        self.P = 0.0
        self.S_TEMP = 0.0
        self.NUM_ERRORS = 0
        self.M2S_max = 0

        self.LATEST_E_LOCATION = 0
        self.SECOND_LATEST_E_LOCATION = 0

        self.NUM_INSTANCES_SEEN = 0

