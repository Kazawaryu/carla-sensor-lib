# Copyright (C) 2023 LEIDOS.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by
# applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import csv
from dataclasses import replace
from unittest.mock import MagicMock

import carla

from objects.DetectedObject import DetectedObjectBuilder


class SimulatedSensorTestUtils:
    TOLERANCE = 1e-4

    @staticmethod
    def generate_simulated_sensor_config():
        return {
            "prefilter": {
                "allowed_semantic_tags": ["CAR", "PEDESTRIAN", "VAN", "TRUCK", "MOTORCYCLE", "CYCLIST"],
                "max_distance_meters": 42
            },
            "detection_threshold_scaling_formula": {
                "nominal_hitpoint_detection_ratio_threshold": 0.6,
                "hitpoint_detection_ratio_threshold_per_meter_change_rate": -0.0033,
                "adjustable_threshold_scaling_parameters": {
                    "dropoff_rate": 0.01
                }
            },
            "geometry_reassociation": {
                # CARLA 0.10 API: Updated from old "sample_count": 3 to new sampling parameters
                "min_sample_count": 1,
                "max_sample_count": 10,
                "downsample_ratio": 2,
                "geometry_association_max_dist_in_meters": 2.0,
            },
            "use_sensor_centric_frame": True
        }

    @staticmethod
    def generate_lidar_sensor_config():
        return {
            "horizontal_fov": 360.0,  # CARLA 0.10.0: horizontal FOV
            "lower_fov": -30.0,  # Vertical FOV lower bound (still used for calculations)
            "upper_fov": 10.0,   # Vertical FOV upper bound (still used for calculations)
            "channels": 60,
            "range": 100.0,
            "rotation_period": 0.05,
            "points_per_second": 10000,
            "projection_string": "EPSG:3857"
        }

    @staticmethod
    def generate_noise_model_config():
        return {
            "noise_model_name": "GaussianNoiseModel",
            "std_deviations": {
                "position_in_meters": [0.8, 0.8, 0.8],
                "orientation_in_radians": [0.1, 0.1, 0.1],
                "linear_velocity_in_ms": [0.1, 0.1, 0.1],
                "angular_velocity_in_rs": [0.1, 0.1, 0.1],
            },
            "stages": {
                "position_noise": True,
                "orientation_noise": True,
                "type_noise": True,
                "list_inclusion_noise": True,
                "position_covariance_noise": True,
                "orientation_covariance_noise": True,
                "linear_velocity_noise": True,
                "angular_velocity_noise": True
            },
            "type_noise": {
                "allowed_semantic_tags": [
                    "Buildings",
                    "Fences",
                    "Ground",
                    "GuardRail",
                    "Pedestrians",
                    "Poles",
                    "RoadLines",
                    "Roads",
                    "Sidewalks",
                    "Sky",
                    "Terrain",
                    "TrafficLight",
                    "TrafficSigns",
                    "Vegetation",
                    "Vehicles",
                    "Walls",
                    "Water"
                ]
            }
        }

    @staticmethod
    def generate_carla_sensor():
        """
        Generate a mock sensor.lidar.ray_cast_semantic.
        :return:
        """
        carla_sensor = MagicMock()
        sensor_config = MagicMock()
        carla_sensor.attributes = {
            "points_per_second": 1000,
            "rotation_frequency": 10.0,
            "horizontal_fov": 360,  # CARLA 0.10.0: replaces upper_fov/lower_fov  
            "channels": 32
        }
        sensor_config.position = carla.Location(1.0, 1.0, 0.0)
        rotation = carla.Rotation(0, 0, 0)
        transform = carla.Transform(carla.Location(1.0, 1.0, 0.0), rotation)
        sensor_config.transform = MagicMock(return_value=transform)
        carla_sensor.get_transform = MagicMock(return_value=transform)
        carla_sensor.get_location = MagicMock(return_value=carla.Location(1.0, 1.0, 0.0))

        return carla_sensor

    @staticmethod
    def generate_test_data_detected_objects():

        # Mock the carla.Actor class
        carla_actor = MagicMock()
        carla_actor.id = 0
        carla_actor.attributes = {'base_type': 'car'}  # CARLA 0.10.0 base_type attribute
        carla_actor.semantic_tags = [int(carla.CityObjectLabel.Car)]  # Add semantic tags for CARLA 0.10.0
        carla_actor.is_alive = True
        carla_actor.parent = None
        carla_actor.type_id = "vehicle.ford.mustang"

        extent = carla.Vector3D(2.94838892768239, 1.69796758051459, 1.0)
        location = carla.Location(20, 34.6410161513775, 0.0)
        rotation = carla.Rotation(3.0, 1.4, 4.0)
        
        # Mock bounding box with get_world_vertices method
        mock_bbox = MagicMock()
        mock_bbox.extent = extent
        mock_bbox.location = location
        mock_bbox.rotation = rotation
        mock_bbox.get_world_vertices = MagicMock(return_value=[
            carla.Location(1.0, 2.0, 3.0),
            carla.Location(4.0, 5.0, 6.0),
            carla.Location(7.0, 8.0, 9.0),
            carla.Location(10.0, 11.0, 12.0)
        ])
        carla_actor.bounding_box = mock_bbox

        carla_actor.get_acceleration = MagicMock(return_value=carla.Vector3D(0.0, 0.0, 0.0))
        carla_actor.get_angular_velocity = MagicMock(return_value=carla.Vector3D(0.0, 0.0, 0.005))
        carla_actor.get_location = MagicMock(return_value=location)
        carla_actor.get_transform = MagicMock(
            return_value=carla.Transform(carla.Location(10.0, 15.0, 7.0), rotation))
        carla_actor.get_velocity = MagicMock(return_value=carla.Vector3D(100.0, 1.0, 0.0))
        carla_actor.get_world = MagicMock(return_value=carla.World)

        # Construct the DetectedObject
        simulated_sensor_config = SimulatedSensorTestUtils.generate_simulated_sensor_config()
        detected_object = DetectedObjectBuilder.build_detected_object(
            carla_actor, 
            simulated_sensor_config["prefilter"]["allowed_semantic_tags"],
            "EPSG:3857",  # projection_string_config
            "test_sensor_1"  # sensor_Id
        )

        # Construct additional DetectedObject by adjustment
        return [
            replace(detected_object, objectId=0, type="CAR"),
            replace(detected_object, objectId=1, type="PEDESTRIAN"),
            replace(detected_object, objectId=2, type="PEDESTRIAN"),
            replace(detected_object, objectId=3, type="PEDESTRIAN"),
            replace(detected_object, objectId=4, type="CAR"),
            replace(detected_object, objectId=5, type="CAR")
        ]

    @staticmethod
    def generate_test_data_hitpoints(self):
        # Read raw data
        test_points = []
        with open("data/test_data_hitpoints.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                theta, x, y, z = map(float, row)
                test_points.append((theta, x, y, z))

        # Chunk into data collection chunks
        num_chunks = 3
        semantic_lidar_measurements = []
        for n in num_chunks:
            chunk = test_points[n * len(test_points) // num_chunks: (n + 1) * len(test_points) // num_chunks]
            semantic_lidar_measurements.append(SimulatedSensorTestUtils.generate_semantic_lidar_measurement(chunk))
        return semantic_lidar_measurements

    @staticmethod
    def generate_semantic_lidar_measurement(test_points_chunk):
        raw_data = []
        for point in test_points_chunk:
            location = carla.Location(x=point[1], y=point[2], z=point[3])
            semantic_lidar_measurement = carla.SemanticLidarMeasurement(point=location, object_idx=123)
            raw_data.append(semantic_lidar_measurement)

        # Assign one representative angle for this measurement. This is a very rough approximation of the real collection data.
        return carla.SemanticLidarMeasurement(channels=1, horizontal_angle=test_points_chunk[0][0], raw_data=raw_data)
