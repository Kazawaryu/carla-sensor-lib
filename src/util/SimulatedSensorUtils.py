# Copyright (C) 2023 LEIDOS.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by
# applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import json

import numpy as np
import yaml
import numpy
import os

from objects.DetectedObject import DetectedObject


class DetectedObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DetectedObject):

            dict_return = {
                'type': obj.type,
                'confidence': obj.confidence,
                'projString': obj.projString,
                'objectId': obj.objectId,
                'position':{'x': round(obj.position[0], 3), 'y': round(obj.position[1],3), 'z': round(obj.position[2], 3)},
                'positionCovariance': obj.positionCovariance.tolist(),
                'velocity':{'x': round(obj.velocity[0], 3), 'y': round(obj.velocity[1], 3), 'z': round(obj.velocity[2],3)},
                'velocityCovariance': obj.velocityCovariance.tolist(),
                'angularVelocity': {'x': round(obj.angularVelocity[0], 3), 'y': round(obj.angularVelocity[1],3), 'z': round(obj.angularVelocity[2],3)},
                'angularVelocityCovariance': obj.angularVelocityCovariance.tolist(),
                'size': {'length': round(obj.size[0], 3), 'height': round(obj.size[1],3), 'width': round(obj.size[2],3)},
                'timestamp': (int)((obj.timestamp)*1000), #CDASim expects Int in milliseconds
                'sensorId': obj.sensorId
                #'bounding_box_in_world_coordinate_frame': [array.tolist() for array in obj.bounding_box_in_world_coordinate_frame],
                #'rotation': obj.rotation.tolist(),
                #'carla_actor': str(obj.carla_actor)
            }

            # Convert DetectedObject attributes to dictionary for serialization.
            # np.ndarray objects are converted to lists using the tolist() method.
            return dict_return
        # Handle numpy.ndarray objects
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle objects with __dict__ (serialize all attributes)
        elif hasattr(obj, '__dict__'):
            # Convert object attributes to dictionary, handling numpy arrays
            obj_dict = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, np.ndarray):
                    obj_dict[key] = value.tolist()
                else:
                    obj_dict[key] = value
            return obj_dict
        # Fallback to the base class default method for other types.
        return super(DetectedObjectEncoder, self).default(obj)


class SimulatedSensorUtils:
    """
    Generic utilities.
    """

    @staticmethod
    def load_config_from_file(config_filepath):
        """
        Load a yaml config file.
        :param config_filepath: Path to the config file.
        :return: Dictionary containing the configuration.
        """
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            return config

    @staticmethod
    def serialize_to_json(obj):
        """
        Serialize any object to JSON. Specialized handling is provided for fields of type numpy.ndarray. Generalized
        deserialization is not possible.

        :param obj: Object to serialize.
        :return: JSON string.
        """

        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            data = [SimulatedSensorUtils.serialize_to_json(item) for item in obj]
            return "[" + str.join(",", data) + "]"
        else:
            return json.dumps(obj, cls=DetectedObjectEncoder)
