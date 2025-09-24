# Copyright (C) 2023 LEIDOS.
# Ported to Carla 10 by Will Varner @ UGA MSC Lab 2025
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by
# applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from collections import Counter
from dataclasses import replace
import carla
import numpy as np
from scipy.spatial import distance
import math
from objects.DetectedObject import DetectedObjectBuilder
from sensor.SimulatedSensor import SimulatedSensor
from util.CarlaUtils import CarlaUtils

prev_objects = {}

class SemanticLidarSensor(SimulatedSensor):
    """
    Wrapper for the CARLA Semantic Lidar Sensor, with additional post-processing logic to compute a list of
    detected objects in the scene.

    Data is reported as the carla.SemanticLidarMeasurement type through the callback registered to
    carla.Sensor.listen(self, callback):

    https://carla.readthedocs.io/en/latest/python_api/#carla.Sensor.listen
    """

    def __init__(self, infrastructure_id, sensor_id, simulated_sensor_config, carla_sensor_config, carla_world, sensor,
                 data_collector, noise_model, parent_id):
        """
        Constructor.

        :param infrastructure_id: Infrastructure ID with which the sensor is associated.
        :param sensor_id: Sensor ID.
        :param simulated_sensor_config: Configuration dictionary containing parameters specific to the wrapped sensor logic.
        :param carla_sensor_config: Parameters relevant to the CARLA sensor configuration.
        :param carla_world: Reference to the CARLA world object.
        :param sensor: CarlaSensor object wrapping the CARLA sensor actor.
        :param data_collector: DataCollector object handling collection and caching of raw sensor data from the CARLA simulation.
        :param noise_model: Noise model available to be used for noise application to the output data.
        """

        # Configuration
        super().__init__(infrastructure_id, sensor_id)
        self.__simulated_sensor_config = simulated_sensor_config
        self.__carla_sensor_config = carla_sensor_config

        # CARLA connection
        self.__carla_world = carla_world

        # Internal objects
        self.__sensor = sensor
        self.__data_collector = data_collector
        self.__noise_model = noise_model
        self.__parent_id = parent_id

        self.__rng = np.random.default_rng()

        # Object cache
        self.__detected_objects = []

    # ------------------------------------------------------------------------------
    # Primary functions
    # ------------------------------------------------------------------------------

    def compute_detected_objects(self):
        """
        Main function used to query the currently-detected objects. Upon calling, the latest raw data cache is
        retrieved and sent through the processing pipeline to produce a list of DetectedObject objects.

        :return: List of DetectedObject objects serialized in JSON form.
        """

        # Get detected_object truth states from simulation
        detected_objects = self.get_scene_detected_objects()

        # Prefilter
        detected_objects, object_ranges = self.prefilter(detected_objects)

        # Get LIDAR hitpoints with Actor ID associations
        timestamp, hitpoints = self.__data_collector.get_carla_lidar_hitpoints()

        detected_objects = [replace(obj, timestamp=timestamp) for obj in detected_objects]

        # Compute data needed for occlusion operation
        actor_angular_extents = self.compute_actor_angular_extents(detected_objects)
        detection_thresholds = self.compute_adjusted_detection_thresholds(detected_objects, object_ranges)

        # Instantaneous geometry association
        min_sample_size = self.__simulated_sensor_config["geometry_reassociation"]["min_sample_count"]
        max_sample_size = self.__simulated_sensor_config["geometry_reassociation"]["max_sample_count"]
        downsample_ratio = self.__simulated_sensor_config["geometry_reassociation"]["downsample_ratio"]

        downsampled_hitpoints = self.sample_hitpoints(hitpoints, min_sample_size, max_sample_size, downsample_ratio)
        hitpoints_without_ids = []

        for hit_id, hitpoint_list in downsampled_hitpoints.items():
            for hitpoint in hitpoint_list:
                hitpoints_without_ids.append(hitpoint)

        hitpoints = self.compute_instantaneous_actor_id_association(hitpoints_without_ids, detected_objects)

        detected_objects = self.apply_occlusion(detected_objects, actor_angular_extents, hitpoints,
                                               detection_thresholds)

        #compare ground truth for position and rotation for SUMO vehicle
        self.store_groundtruth_pose_angular_diff(detected_objects)

        #due to sumo vehicle not having velocity and angular velocity in CARLA, added a patch to calculate those parameters
        detected_objects = self.update_velocity_angularVelocity(detected_objects)

        # Apply noise
        detected_objects = self.apply_noise(detected_objects)

        # Update reference frame, and detection time
        detected_objects = self.update_object_frame_and_timestamps(detected_objects, timestamp)

        self.__detected_objects = detected_objects

        return detected_objects

    def get_detected_objects(self):
        """Returns the latest detected objects."""
        return self.__detected_objects

    # ------------------------------------------------------------------------------
    # CARLA Scene DetectedObject Retrieval
    # ------------------------------------------------------------------------------

    def get_scene_detected_objects(self):
        """
        Retrieve the current objects in scene. This collection is considered the "truth state" as it contains the set
        of objects in the world, as well as their positions, orientations, and various other state data.

        :return: DetectedObject wrappers objects referring to the actors.
        """
        actors = self.__carla_world.get_actors()

        scene_objects = [DetectedObjectBuilder.build_detected_object(actor,
                                                            self.__simulated_sensor_config["prefilter"]["allowed_semantic_tags"],
                                                            self.__carla_sensor_config["projection_string"],
                                                            self._sensor_id)
                for actor in actors]

        # Remove invalid objects
        scene_objects = list(filter(lambda obj: obj is not None, scene_objects))

        # Remove sensor's parent object if detected (LIDAR sensor detecting car to which it is attached)
        scene_objects = list(filter(lambda obj: obj.objectId != self.__parent_id, scene_objects))

        return scene_objects

    # ------------------------------------------------------------------------------
    # Prefilter
    # ------------------------------------------------------------------------------

    def prefilter(self, detected_objects):
        """
        Filter the truth state for objects of the desired detection type, and within range of the prefilter distance.

        :param detected_objects: List of objects currently considered for detection.
        :return: Filtered set of objects, and associated object ranges in a lookup structure.
        """
        #

        # Filter by detected_object type Actor.type_id and Actor.semantic_tags are available for determining type;
        # semantic_tags effectively specifies the type of detected_object Possible types are listed in the CARLA
        # documentation: https://carla.readthedocs.io/en/0.9.10/ref_sensors/#semantic-segmentation-camera

        detected_objects = list(
            filter(lambda obj: obj is not None, detected_objects))

        detected_objects = list(
            filter(lambda obj: obj.type in self.__simulated_sensor_config["prefilter"]["allowed_semantic_tags"],
                   detected_objects))

        # Compute ranges
        sensor_location = self.__sensor.carla_sensor.get_location()
        sensor_location_np = np.array([sensor_location.x, sensor_location.y, sensor_location.z])
        object_ranges = dict(
            [(obj.objectId, np.linalg.norm(obj.position - sensor_location_np)) for obj in detected_objects])

        # Filter by radius
        detected_objects = list(
            filter(
                lambda obj: object_ranges[obj.objectId] <= self.__simulated_sensor_config["prefilter"]["max_distance_meters"],
                detected_objects))

        return detected_objects, object_ranges
# ------------------------------------------------------------------------------
    # Computations
    # ------------------------------------------------------------------------------

    def compute_actor_angular_extents(self, detected_objects):
        """
        Compute horizontal and vertical angular FOV extent from sensor perspective for each detected object.

        :param detected_objects: List of objects currently considered for detection.
        :return: Dictionary mapping actor ID to tuple of (horizontal, vertical) angular extents.
        """
        return dict([(detected_object.objectId,
                      self.compute_actor_angular_extent(detected_object)) for detected_object in
                     detected_objects])

    def compute_actor_angular_extent(self, detected_object):
        """
        Compute the horizontal and vertical angular FOV extent from sensor perspective for the given object.

        :param detected_object: Object under consideration.
        :return: Tuple containing the maximal horizontal and
        vertical angular extents consumed by the object, from the sensor's perspective.
        """
        corner_vectors_in_world_frame = detected_object.bounding_box_in_world_coordinate_frame
        theta = [self.compute_horizontal_angular_offset(vec) for vec in corner_vectors_in_world_frame]
        phi = [self.compute_vertical_angular_offset(vec) for vec in corner_vectors_in_world_frame]
        horizontal_fov = max(theta) - min(theta)
        vertical_fov = max(phi) - min(phi)
        return horizontal_fov, vertical_fov

    def compute_horizontal_angular_offset(self, vec):
        """Compute horizontal angle of a vector in relation to the sensor, as measured from the x axis."""
        sensor_location = self.__sensor.carla_sensor.get_location()
        p = vec - np.array([sensor_location.x, sensor_location.y, sensor_location.z])  # Position vector relative to sensor
        return np.arctan2(p[1], p[0])

    def compute_vertical_angular_offset(self, vec):
        """Compute the vertical angle of a vector in relation to the sensor, as measured from the x-y plane."""
        sensor_location = self.__sensor.carla_sensor.get_location()
        p = vec - np.array([sensor_location.x, sensor_location.y, sensor_location.z])
        return np.arcsin(p[2] / np.linalg.norm(p))

    def compute_adjusted_detection_thresholds(self, detected_objects, object_ranges):
        """
        Compute the adjusted detection threshold for each object, based on the object's range. This threshold varies
        with distance to accommodate ray spreading which naturally leads to lower hitpoint counts, risking object
        misdetections.
        """
        return dict([(detected_object.objectId,
                      self.compute_adjusted_detection_threshold(object_ranges[detected_object.objectId]))
                     for
                     detected_object in detected_objects])

    def compute_adjusted_detection_threshold(self, range):
        """Compute the detection threshold appropriate for the given range."""
        dt_dr = self.__simulated_sensor_config["detection_threshold_scaling_formula"][
            "hitpoint_detection_ratio_threshold_per_meter_change_rate"]
        t_nominal = self.__simulated_sensor_config["detection_threshold_scaling_formula"][
            "nominal_hitpoint_detection_ratio_threshold"]

        return dt_dr * range * t_nominal

    # ------------------------------------------------------------------------------
    # Geometry Re-Association: Sampling
    # ------------------------------------------------------------------------------

    def sample_hitpoints(self, hitpoints, min_sample_size, max_sample_size, downsample_ratio):
        """Down sample each object's hitpoint list

        Randomly sample points inside each object's set of LIDAR
        hitpoints. This is done to reduce size of the data being sent
        through the distance computation. If the hitpoint population of
        a specific object is less than the sample size, the whole
        hitpoint population will be used (i.e., the object's list of
        hitpoints will remain unchanged).
        NOTE: sizes and ratio in inputs affect individual object_idx in CARLA 0.9.10
        which is same as object_tag, but not actor_id. For example, if there are 2 vehicle.carlamotors.carlacola,
        the size and ratios will apply to total hitpoints of vehicle.carlamotors.carlacola, but not individually.

        :param hitpoints: lidar points associated with each object
        :param min_sample_size: minimum size to sample from hitpoint list for each objects
        :param max_sample_size: maximum size to sample from hitpoint list for each objects
        :param downsample_ratio: int number determining the factor to which the size of the sample is reduced
        :return: downsampled hitpoints
        """

        def clamp(value):
            return max(min_sample_size, min(value, max_sample_size))

        return {
            id_: self.__rng.choice(
                points, min(len(points), clamp(math.ceil(len(points) / downsample_ratio))), replace=False
            )
            for id_, points in hitpoints.items()
        }

    # ------------------------------------------------------------------------------
    # Geometry Re-Association: Instantaneous Association
    # ------------------------------------------------------------------------------

    def compute_instantaneous_actor_id_association(self, hitpoints, scene_objects):
        """
        Compute the association of each sampled hitpoint to each actor, based on distance of the hitpoint to the
        actor's position.

        :param hitpoints: List of hitpoints.
        :param scene_objects: List of objects currently considered for detection.
        :return: Actor ID association applicable to this time step based on geometry-based association algorithm,
            in the form of a dictionary mapping {hitpoint ID -> actor ID}.
        """

        # Compute nearest neighbor for each hitpoint
        hitpoints_in_map_frame = []

        for hitpoint in hitpoints:
            # The hitpoint argument is now a carla.SemanticLidarDetection object
            hitpoint_in_lidar_frame = hitpoint.point
            # transform function translates a 3D point from local to global
            # coordinates using the current transformation as frame of reference
            hitpoint_in_map_frame = self.__sensor.carla_sensor.get_transform().transform(hitpoint_in_lidar_frame)
            new_pos = np.array([hitpoint_in_map_frame.x, hitpoint_in_map_frame.y, hitpoint_in_map_frame.z])
            hitpoints_in_map_frame.append(new_pos)

        matching_nearest_neighbor_ids = self.compute_closest_object_id_list(hitpoints_in_map_frame, scene_objects,
                                                          self.__simulated_sensor_config["geometry_reassociation"][
                                                              "geometry_association_max_dist_in_meters"])

        association = zip(hitpoints_in_map_frame, matching_nearest_neighbor_ids)

        grouped_data = dict()
        for hitpoint, actor_id in association:
            if actor_id is None:
                continue
            # some hitpoints may not get association due to associator's range, which the library ignores
            if actor_id not in grouped_data:
                grouped_data[actor_id] = [hitpoint]
            else:
                grouped_data[actor_id].append(hitpoint)

        # Filter unassociated hitpoints
        return grouped_data

    def compute_closest_object_id_list(self, hitpoint_list, scene_objects, geometry_association_max_dist_in_meters):

        """Get the closest objects to each hitpoint."""
        return [self.compute_closest_object_id(hitpoint, scene_objects, geometry_association_max_dist_in_meters) for
                hitpoint in hitpoint_list]

    def compute_closest_object_id(self, hitpoint, scene_objects, geometry_association_max_dist_in_meters):
        """
        Compute the closest object to this hitpoint, which lies within a maximum distance threshold.

        The threshold prevents association between a point and object which are very far apart.
        """
        if len(scene_objects) == 0:
            return None
        object_positions = [obj.position for obj in scene_objects]

        distances_list = distance.cdist([hitpoint], object_positions)

        if len(distances_list) <= 0 or len(distances_list[0]) <= 0:
            return None

        distances = distances_list[0]
        closest_index = np.argmin(distances)

        # May result from bad or empty data
        if closest_index is None or closest_index < 0:
            return None

        # Due to vehicles being a large object compared to pedestrians, more buffer maybe required
        geometry_association_threshold_buffer = 0.0
        if scene_objects[closest_index].type in ("VAN", "TRUCK"):
            geometry_association_threshold_buffer = 3.0
        elif scene_objects[closest_index].type == "CAR":
            geometry_association_threshold_buffer = 2.0

        # Observe a maximum object distance to preclude association with far-away objects
        if distances[closest_index] > geometry_association_max_dist_in_meters + geometry_association_threshold_buffer:
            return None

        # Return closest object's ID
        closest_object = scene_objects[closest_index]
        return closest_object.objectId

    def vote_most_frequent_id(self, object_id_list):
        """Determine the object with the highest number of votes as determined by the nearest-neighbor search."""

        if object_id_list is None or len(object_id_list) == 0:
            return None

        # Do not consider incorrectly associated ID's
        object_id_list = list(filter(lambda x: x is not None, object_id_list))

        # Identify the most frequently occurring ID
        most_frequent_pair = Counter(object_id_list).most_common(1)
        if most_frequent_pair is not None and len(most_frequent_pair) > 0:
            return most_frequent_pair[0][0]
        else:
            return None

    # ------------------------------------------------------------------------------
    # Occlusion Filter
    # ------------------------------------------------------------------------------

    def apply_occlusion(self, detected_objects, actor_angular_extents, hitpoints, detection_thresholds):
        """
        Filter detected objects by occlusion. This is determined by comparing the number of hitpoints expected to hit
        the object as computed from the problem geometry, with the number of hitpoints detected.

        :param detected_objects: List of objects currently considered for detection.
        :param actor_angular_extents: Dictionary containing angular extent pairs referenced by object ID.
        :param detection_thresholds: Thresholds computed for each object based on distance from the sensor.
        :return: List of objects filtered by occlusion.
        """
        return list(filter(
            lambda obj: self.is_visible(actor_angular_extents.get(obj.objectId), hitpoints.get(obj.objectId),
                                        detection_thresholds.get(obj.objectId)),
            detected_objects))

    def is_visible(self, actor_angular_extents, object_hitpoints, detection_threshold_ratio):
        """
        Compute if an object is visible based on the ratio of actual to expected hitpoints.

        :param actor_angular_extents: Tuple containing the horizontal and vertical angular extents of the object.
        :param object_hitpoints: List of hitpoints associated with the object.
        :param detection_threshold_ratio: Threshold ratio computed for this object based on distance from the sensor.
        :return: True if the object is visible, False otherwise.
        """

        if actor_angular_extents is None or object_hitpoints is None or detection_threshold_ratio is None:
            return False

        actor_horizontal_angular_extent = actor_angular_extents[0]
        actor_vertical_angular_extent = actor_angular_extents[1]

        # Compute threshold hitpoint count for this object
        num_expected_hitpoints = self.compute_expected_num_hitpoints(actor_horizontal_angular_extent, actor_vertical_angular_extent)

        # number of expected hitpoint count is reduced by sampling
        downsample_ratio = self.__simulated_sensor_config["geometry_reassociation"]["downsample_ratio"]
        num_expected_hitpoints = num_expected_hitpoints / downsample_ratio

        min_hitpoint_count = detection_threshold_ratio * num_expected_hitpoints

        # Compare hitpoint count
        num_hitpoints = len(object_hitpoints)

        return num_hitpoints >= min_hitpoint_count

    def compute_expected_num_hitpoints(self, actor_horizontal_angular_extent, actor_vertical_angular_extent):
        """
        Compute the expected number of hitpoints for the given field of view without accounting for distance. Imagine even distribution of hitpoints within some radius.
        NOTE: This result is heavily determined by the CARLA sensor configuration.

        :param actor_horizontal_angular_extent: Actor's detected horizontal field of view in radians.
        :param actor_vertical_angular_extent: Actor's detected vertical field of view in radians.
        :return: Expected number of hitpoints in a scan across the specified field of view.
        """
        # Replaced removed sensor attributes with values from the configuration dictionary
        num_horizontal_points_per_scan = (self.__carla_sensor_config["points_per_second"] / self.__carla_sensor_config["rotation_period"]) / self.__carla_sensor_config["channels"]
        horizontal_angular_resolution = math.radians(self.__carla_sensor_config["horizontal_fov"]) / num_horizontal_points_per_scan

        num_vertical_points_per_scan = self.__carla_sensor_config["channels"]
        vertical_angular_resolution = math.radians(self.__carla_sensor_config["upper_fov"] - self.__carla_sensor_config["lower_fov"]) / num_vertical_points_per_scan

        return (actor_horizontal_angular_extent / horizontal_angular_resolution) * (actor_vertical_angular_extent / vertical_angular_resolution)

    # ------------------------------------------------------------------------------
    # Noise Filter
    # ------------------------------------------------------------------------------

    def apply_noise(self, detected_objects):
        """
        Apply noise to the detected objects.

        :param detected_objects: List of objects currently considered for detection.
        :return: Objects with noise applied.
        """
        detected_objects = self.__noise_model.apply_position_noise(detected_objects)
        detected_objects = self.__noise_model.apply_orientation_noise(detected_objects)
        detected_objects = self.__noise_model.apply_type_noise(detected_objects)
        detected_objects = self.__noise_model.apply_list_inclusion_noise(detected_objects)
        detected_objects = self.__noise_model.apply_position_covariance_noise(detected_objects)
        detected_objects = self.__noise_model.apply_orientation_covariance_noise(detected_objects)
        detected_objects = self.__noise_model.apply_linear_velocity_covariance_noise(detected_objects)
        detected_objects = self.__noise_model.apply_angular_velocity_covariance_noise(detected_objects)

        return detected_objects

    # ------------------------------------------------------------------------------
    # Post Processing
    # ------------------------------------------------------------------------------

    def update_object_frame_and_timestamps(self, detected_objects, timestamp):
        """
        Update object metadata including detection timestamp in seconds and adjusting the coordinates to the
        sensor-centric frame.

        :param detected_objects: List of objects currently considered for detection.
        :param timestamp: Timestamp of the current frame (seconds).
        :return: List of objects with updated metadata.
        """

        return [self.update_object_frame_and_timestamps_from_hitpoint(obj, timestamp)
                for obj in detected_objects]

    def update_object_frame_and_timestamps_from_hitpoint(self, obj, timestamp):
        """
        Update the object metadata timestamp and coordinates .

        :param obj: Detected object.
        :param timestamp: Timestamp of the current frame (seconds).
        :return: Updated object.
        """

        # If enabled, convert coordinates to sensor-centric frame
        new_position = obj.position
        new_velocity = obj.velocity
        new_angularVelocity = obj.angularVelocity
        new_rotation = obj.rotation

        if self.__simulated_sensor_config["use_sensor_centric_frame"]:
            inv_T = np.array(self.__sensor.carla_sensor.get_transform().get_inverse_matrix())
            pos_in_map = np.array([new_position[0],
                                    new_position[1],
                                    new_position[2],
                                    1.0])
            new_position = np.matmul(inv_T, pos_in_map)
            sensor_rotation = self.__sensor.carla_sensor.get_transform().rotation
            new_rotation[0] -= math.radians(sensor_rotation.roll)
            new_rotation[1] -= math.radians(sensor_rotation.pitch)
            new_rotation[2] -= math.radians(sensor_rotation.yaw)

        # The bug where the y-axis was negated in CARLA 0.9.10 is fixed in 0.10.0 and later.
        # The location provided by the XML-RPC client is now correct, so no negation is needed.

        return replace(obj,
                       timestamp=timestamp,
                       position=new_position,
                       velocity = new_velocity,
                       rotation = new_rotation,
                       angularVelocity = new_angularVelocity
                       )

    def update_velocity_angularVelocity(self, detected_objects):
        for detected_object in detected_objects:
            if detected_object.objectId in prev_objects and 'timestamp' in prev_objects[detected_object.objectId]:
                time_diff = detected_object.timestamp - prev_objects[detected_object.objectId]['timestamp']
                if  time_diff and np.all(detected_object.velocity==0) and not np.all(prev_objects[detected_object.objectId]['pose_diff']==0):
                    detected_object.velocity[0] = prev_objects[detected_object.objectId]['pose_diff'][0] / time_diff
                    detected_object.velocity[1] = prev_objects[detected_object.objectId]['pose_diff'][1] / time_diff
                    detected_object.velocity[2] = prev_objects[detected_object.objectId]['pose_diff'][2] / time_diff

                #Calculation based on an assumption that the object's orientation is the direction of travel
                if time_diff and np.all(detected_object.angularVelocity==0) and not np.all(prev_objects[detected_object.objectId]['rotation_diff']==0):
                    detected_object.angularVelocity[0] = prev_objects[detected_object.objectId]['rotation_diff'][0]/time_diff
                    detected_object.angularVelocity[1] = prev_objects[detected_object.objectId]['rotation_diff'][1]/time_diff
                    detected_object.angularVelocity[2] = prev_objects[detected_object.objectId]['rotation_diff'][2]/time_diff


            prev_objects[detected_object.objectId]['timestamp'] = detected_object.timestamp


        return detected_objects

    def store_groundtruth_pose_angular_diff(self, detected_objects):
        for detected_object in detected_objects:
            if detected_object.objectId in prev_objects:
                pose_diff_x = detected_object.position[0] - prev_objects[detected_object.objectId]['pose'][0]
                pose_diff_y = detected_object.position[1] - prev_objects[detected_object.objectId]['pose'][1]
                pose_diff_z = detected_object.position[2] - prev_objects[detected_object.objectId]['pose'][2]

                prev_objects[detected_object.objectId]['pose_diff'] = np.array([pose_diff_x, pose_diff_y, pose_diff_z])

                rotation_diff_roll = math.radians(detected_object.rotation[0] - prev_objects[detected_object.objectId]['rotation'][0])
                rotation_diff_pitch = math.radians(detected_object.rotation[1] - prev_objects[detected_object.objectId]['rotation'][1])
                rotation_diff_yaw = math.radians(detected_object.rotation[2] - prev_objects[detected_object.objectId]['rotation'][2])

                prev_objects[detected_object.objectId]['rotation_diff'] = np.array([rotation_diff_roll, rotation_diff_pitch, rotation_diff_yaw])

            else:
                prev_objects[detected_object.objectId] = {}

            prev_objects[detected_object.objectId]['pose'] = detected_object.position.copy()
            prev_objects[detected_object.objectId]['rotation'] = detected_object.rotation.copy()