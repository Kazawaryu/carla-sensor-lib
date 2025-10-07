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
from scipy.spatial.transform import Rotation
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))

from util.CarlaUtils import CarlaUtils


class TestCarlaUtils(unittest.TestCase):

    def setUp(self):
        self.carla_actor = MagicMock()

    def test_vector3d_to_numpy(self):
        vec = MagicMock(x=1.0, y=2.0, z=3.0)
        result = CarlaUtils.vector3d_to_numpy(vec)
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0, 3.0])))

    def test_vector2d_to_numpy(self):
        vec = MagicMock(x=1.0, y=2.0)
        result = CarlaUtils.vector2d_to_numpy(vec)
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0])))

    def test_get_actor_angular_velocity(self):
        carla_actor = MagicMock(get_angular_velocity=MagicMock(return_value=MagicMock(x=1.0, y=2.0, z=3.0)))
        result = CarlaUtils.get_actor_angular_velocity(carla_actor)
        self.assertTrue(np.array_equal(result, np.deg2rad(np.array([1.0, 2.0, 3.0]))))

    def test_get_actor_rotation_matrix(self):
        # Replicate the rotation
        rotation_angles_deg = np.array([45.0, 30.0, 90.0])
        rotation_angles = np.deg2rad(rotation_angles_deg)
        rotation_matrix = Rotation.from_euler('xyz', rotation_angles)

        # Get the rotation matrix
        carla_actor = MagicMock(
            get_transform=MagicMock(return_value=MagicMock(rotation=carla.Rotation(30.0, 90.0, 45.0))))
        result = CarlaUtils.get_actor_rotation_matrix(carla_actor)

        assert np.allclose(result, rotation_matrix.as_matrix())

    def test_get_actor_bounding_box_points(self):
        carla_actor = MagicMock(
            bounding_box=MagicMock(get_world_vertices=MagicMock(return_value=[carla.Location(1.0, 2.0, 3.0),
                                                                              carla.Location(4.0, 5.0, 6.0),
                                                                              carla.Location(7.0, 8.0, 9.0),
                                                                              carla.Location(10.0, 11.0, 12.0)])))
        result = CarlaUtils.get_actor_bounding_box_points(carla_actor)
        assert (result[0] == np.array([1.0, 2.0, 3.0])).all()
        assert (result[1] == np.array([4.0, 5.0, 6.0])).all()
        assert (result[2] == np.array([7.0, 8.0, 9.0])).all()
        assert (result[3] == np.array([10.0, 11.0, 12.0])).all()

    def test_determine_object_type(self):
        # CARLA 0.10.0: Test with vehicle actor (type_id starts with "vehicle.")
        carla_actor = MagicMock(type_id="vehicle.tesla.model3", attributes={'base_type': 'car'})
        assert "CAR" == CarlaUtils.determine_object_type(carla_actor, ["PEDESTRIAN", "CAR"])

        # Multiple types
        assert "CAR" == CarlaUtils.determine_object_type(carla_actor, ["NONE", "CAR"])

        # No allowed type
        assert "NONE" == CarlaUtils.determine_object_type(carla_actor, ["NONE"])

        # Test with pedestrian actor
        carla_actor = MagicMock(type_id="walker.pedestrian.0001", attributes={})
        result = CarlaUtils.determine_object_type(carla_actor, ["CAR", "BUILDINGS"])
        self.assertEqual(result, "NONE")

    def test_get_semantic_tag_name(self):
        # CARLA 0.10.0 tag IDs
        assert CarlaUtils.get_semantic_tag_name(0) == "NONE"
        assert CarlaUtils.get_semantic_tag_name(1) == "Roads"
        assert CarlaUtils.get_semantic_tag_name(2) == "Sidewalks"
        assert CarlaUtils.get_semantic_tag_name(3) == "Buildings"
        assert CarlaUtils.get_semantic_tag_name(6) == "Poles"
        assert CarlaUtils.get_semantic_tag_name(11) == "Sky"
        assert CarlaUtils.get_semantic_tag_name(12) == "Pedestrians"
        assert CarlaUtils.get_semantic_tag_name(14) == "Car"
        assert CarlaUtils.get_semantic_tag_name(20) == "Static"
        assert CarlaUtils.get_semantic_tag_name(22) == "Other"
        assert CarlaUtils.get_semantic_tag_name(23) == "Water"
        assert CarlaUtils.get_semantic_tag_name(24) == "RoadLines"
        assert CarlaUtils.get_semantic_tag_name(27) == "RailTrack"

    def test_get_semantic_tag_id(self):
        # CARLA 0.10.0 tag IDs
        assert CarlaUtils.get_semantic_tag_id("NONE") == 0
        assert CarlaUtils.get_semantic_tag_id("Roads") == 1
        assert CarlaUtils.get_semantic_tag_id("Sidewalks") == 2
        assert CarlaUtils.get_semantic_tag_id("Buildings") == 3
        assert CarlaUtils.get_semantic_tag_id("Poles") == 6
        assert CarlaUtils.get_semantic_tag_id("Sky") == 11
        assert CarlaUtils.get_semantic_tag_id("Pedestrians") == 12
        assert CarlaUtils.get_semantic_tag_id("Car") == 14
        assert CarlaUtils.get_semantic_tag_id("Static") == 20
        assert CarlaUtils.get_semantic_tag_id("Other") == 22
        assert CarlaUtils.get_semantic_tag_id("Water") == 23
        assert CarlaUtils.get_semantic_tag_id("RoadLines") == 24
        assert CarlaUtils.get_semantic_tag_id("RailTrack") == 27
        # Legacy "Vehicles" should map to "Car"
        assert CarlaUtils.get_semantic_tag_id("Vehicles") == 14
