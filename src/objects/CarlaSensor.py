# Copyright (C) 2023 LEIDOS.
# Ported to Carla 10 by Will Varner @ UGA MSC Lab 2025
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by
# applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from dataclasses import dataclass

import carla
import numpy as np

from util.CarlaUtils import CarlaUtils


@dataclass(frozen=True)
class CarlaSensor:
    """Wrapper class for the carla.Sensor."""
    carla_sensor: carla.Sensor
    position: np.ndarray
    points_per_second: float
    rotation_frequency: float
    horizontal_fov: float
    vertical_fov: float
    number_of_channels: int


class CarlaSensorBuilder:
    @staticmethod
    def build_sensor(carla_sensor):

        # CARLA 0.10.0 and newer versions of the Lidar sensor blueprint
        # only provide a 'horizontal_fov' attribute and have removed 'upper_fov'
        # and 'lower_fov'.
        if "horizontal_fov" in carla_sensor.attributes:
            horizontal_fov = float(carla_sensor.attributes["horizontal_fov"])
            # In the absence of 'upper_fov' and 'lower_fov', a reasonable assumption is
            # that the vertical FOV is now equal to the horizontal FOV.
            vertical_fov = horizontal_fov
        else:
            horizontal_fov = 360.0
            vertical_fov = 360.0

        return CarlaSensor(carla_sensor,
                           CarlaUtils.vector3d_to_numpy(carla_sensor.get_location()),
                           int(carla_sensor.attributes["points_per_second"]),
                           float(carla_sensor.attributes["rotation_frequency"]),
                           np.deg2rad(horizontal_fov),
                           np.deg2rad(vertical_fov),
                           int(carla_sensor.attributes["channels"]))