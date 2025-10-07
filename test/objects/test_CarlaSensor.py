# Copyright (C) 2023 LEIDOS.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by
# applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import unittest
from unittest.mock import MagicMock

import carla
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))

from objects.CarlaSensor import CarlaSensorBuilder


class TestCarlaSensor(unittest.TestCase):

    def setUp(self):
        self.carla_sensor = MagicMock()
        self.carla_sensor.get_location = MagicMock(return_value=carla.Location(1.0, 2.0, 3.0))
        self.carla_sensor.attributes = {
            "points_per_second": 1000,
            "rotation_frequency": 700,
            "horizontal_fov": 360,  # CARLA 0.10.0: replaces upper_fov/lower_fov
            "channels": 32
        }

    def test_builder(self):
        sensor = CarlaSensorBuilder.build_sensor(self.carla_sensor)

        assert sensor.carla_sensor == self.carla_sensor
        assert (sensor.position == np.array([1.0, 2.0, 3.0])).all()
        assert sensor.points_per_second == 1000
        assert sensor.rotation_frequency == 700
        assert sensor.horizontal_fov == np.deg2rad(360)
        # CARLA 0.10.0: Calculate vertical FOV from horizontal FOV for compatibility
        assert sensor.vertical_fov == np.deg2rad(360)  # Updated expectation for CARLA 0.10.0
