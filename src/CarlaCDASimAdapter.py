# Copyright (C) 2024 LEIDOS.
# Ported for Carla 10 by Will Varner @ UGA MSC Lab 2025
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by
# applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
import argparse
import threading
import logging
import datetime
from xmlrpc.server import SimpleXMLRPCServer
import sys
sys.path.append('../')
from CarlaCDASimAPI import CarlaCDASimAPI
from util.SimulatedSensorUtils import SimulatedSensorUtils
class CarlaCDASimAdapter:
    # Holds Sensor configuration object
    sensor_config = None
    # Holds noise model configuration objst
    noise_model_config = None
    # Holds detection cycle delay seconds
    detection_cycle_delay_seconds = None
    
    def __init__(self, sensor_api):
        """
        CarlaCDASimAdapter constructor.
        :param sensor_api: The API object exposing CARLA connection.
        """
        self.__api = sensor_api
        
    def start_xml_rpc_server(self, xmlrpc_server_host, xmlrpc_server_port, sensor_config_file, noise_model_config_file, detection_cycle_delay_seconds, blocking=True):
        """
        Starts the XML-RPC server for the sensor data service.
        :param xmlrpc_server_host: The server host.
        :param xmlrpc_server_port: The server port.
        :param blocking: Toggles server starting on main thread. If False the server will be started on a new thread, and control will be returned to the caller.
        :return: Thread object the server was started on, or blocked if the server was started on the main thread.
        """
        # Create an XML-RPC server
        logging.info("Starting sensorlib XML-RPC server.")
        # Silence XMLRPC server logging if not in debug
        log_requests = logging.getLogger().isEnabledFor(logging.DEBUG)
        server = SimpleXMLRPCServer((xmlrpc_server_host, xmlrpc_server_port), logRequests=log_requests)
        server.register_introspection_functions()
        server.register_function(self.__create_simulated_semantic_lidar_sensor,
                                 "create_simulated_semantic_lidar_sensor")
        server.register_function(self.__get_simulated_sensor, "get_simulated_sensor")
        server.register_function(self.__get_detected_objects, "get_detected_objects")
        server.register_function(self.__connect, "connect")
        self.sensor_config = SimulatedSensorUtils.load_config_from_file(sensor_config_file)
        self.noise_model_config = SimulatedSensorUtils.load_config_from_file(noise_model_config_file)
        self.detection_cycle_delay_seconds = detection_cycle_delay_seconds
        # Start, with blocking option
        if blocking:
            return server.serve_forever()
        else:
            rpc_server_thread = threading.Thread(target=server.serve_forever)
            rpc_server_thread.start()
            return rpc_server_thread
            
    def __create_simulated_semantic_lidar_sensor(self,
                                                 infrastructure_id, sensor_id,
                                                 sensor_position, sensor_rotation):
        logging.info(f"Received request to create sensor at {sensor_position}")
        # The bug where the y-axis was negated in CARLA 0.9.10 is fixed in 0.10.0 and later.
        # The location provided by the XML-RPC client is now correct, so no negation is needed.
        logging.info(f"Updated sensor position to {sensor_position}")
        simulated_sensor = self.__api.create_simulated_semantic_lidar_sensor(self.sensor_config["simulated_sensor"],
                                                                             self.sensor_config["lidar_sensor"],
                                                                             self.noise_model_config,
                                                                             self.detection_cycle_delay_seconds,
                                                                             infrastructure_id, sensor_id,
                                                                             sensor_position, sensor_rotation)
        return str(simulated_sensor.get_id())
                                                     
    def __connect(self):
        logging.info("CARLA CDASim Adapter connected to CDASim CARLA Ambassador!")
        return True
        
    def __get_simulated_sensor(self, infrastructure_id, sensor_id):
        sensor = self.__api.get_simulated_sensor(infrastructure_id, sensor_id)
        logging.debug(f"Retrieved sensor : {sensor.get_id()} for infrastructure {infrastructure_id}")
        return str(sensor.get_id())
        
    def __get_detected_objects(self, infrastructure_id, sensor_id):
        detected_objects = self.__api.get_detected_objects(infrastructure_id, sensor_id)
        return_json = str(SimulatedSensorUtils.serialize_to_json(detected_objects))
        logging.debug(f"Sensor {sensor_id} detected objects {return_json}")
        return return_json
        
if __name__ == "__main__":
    # Parse arguments
    arg_parser = argparse.ArgumentParser(
        description=__doc__)
    
    arg_parser.add_argument(
        "--carla-host",
        default="127.0.0.1",
        type=str,
        help="CARLA host. (default: \"localhost\")")
    
    arg_parser.add_argument(
        "--carla-port",
        default="2000",
        type=str,
        help="CARLA host. (default: \"2000\")")
    
    arg_parser.add_argument(
        "--xmlrpc-server-host",
        default="localhost",
        type=str,
        help="XML-RPC server host. (default: \"localhost\")")
    
    arg_parser.add_argument(
        "--xmlrpc-server-port",
        default=8000,
        type=int,
        help="XML-RPC server port. (default: 8000)")
    
    arg_parser.add_argument(
        "--sensor-config-file",
        default="./config/simulated_sensor_config.yaml",
        type=str,
        help="Path to sensor configuration file. (default: ./config/simulated_sensor_config.yaml)")
    
    arg_parser.add_argument(
        "--noise-model-config-file",
        default="./config/noise_model_config.yaml",
        type=str,
        help="Path to noise mode configuration file. (default: ./config/noise_model_config.yaml)")
    
    arg_parser.add_argument(
        "--detection-cycle-delay-seconds",
        default=0.1,
        type=float,
        help="Time interval between detection reporting. (default: 0.1)")
    
    arg_parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        help="Log Level for service (default: INFO)")
    arg_parser.add_argument(
        "--log-file-path",
        default="/home/CarlaCDASimAdapter/logs/",
        type=str,
        help="Absolute path for log file.")
    
    args = arg_parser.parse_args()
    log_level = logging.getLevelName(args.log_level)
    log_file_name = datetime.datetime.now().strftime("carla_cdasim_adapter_%H_%M_%d_%m_%Y.log")
    FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
    logging.basicConfig(format=FORMAT,  
        handlers=[
            logging.FileHandler(args.log_file_path + log_file_name),
            logging.StreamHandler()
            ], 
        level=log_level)

    sensor_api = CarlaCDASimAPI.build_from_host_spec(args.carla_host, args.carla_port)
    sensor_data_service = CarlaCDASimAdapter(sensor_api)
    sensor_data_service.start_xml_rpc_server(args.xmlrpc_server_host, args.xmlrpc_server_port, args.sensor_config_file, args.noise_model_config_file, args.detection_cycle_delay_seconds, True)