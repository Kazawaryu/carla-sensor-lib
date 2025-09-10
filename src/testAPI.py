import argparse
import logging
import carla
import json
import os
import sys

# Add the parent directory of the API to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CarlaCDASimAPI import CarlaCDASimAPI

def _run_test(client, config):
    """
    Runs the core test logic.
    """
    world = client.get_world()
    
    # Use the existing _setup_vehicle function to spawn a vehicle
    vehicle = _setup_vehicle(world, config)

    print("set up vehicle")
    
    # Instantiate the API and try to create a sensor
    api = CarlaCDASimAPI.build_from_world(world)
    
    # Get the sensor configuration from the stack.json file
    sensor_config = config.get("sensors", [])[0]

    print("Creating simulated sensor...")
    
    # Use the create_simulated_semantic_lidar_sensor function from the API
    simulated_sensor = api.create_simulated_semantic_lidar_sensor(
        simulated_sensor_config={},
        carla_sensor_config=sensor_config["attributes"],
        noise_model_config={"noise_model_name": "identity"},
        detection_cycle_delay_seconds=0.1,
        infrastructure_id="test_infrastructure",
        sensor_id=sensor_config["id"],
        sensor_position=carla.Location(
            x=sensor_config["spawn_point"]["x"], 
            y=sensor_config["spawn_point"]["y"], 
            z=sensor_config["spawn_point"]["z"]
        ),
        sensor_rotation=carla.Rotation(
            roll=sensor_config["spawn_point"]["roll"], 
            pitch=sensor_config["spawn_point"]["pitch"], 
            yaw=sensor_config["spawn_point"]["yaw"]
        ),
        parent_id=vehicle.id
    )

    print("Simulated sensor created")

    # Assert that the sensor was successfully created
    assert simulated_sensor is not None
    logging.info("Test passed: Sensor was created successfully.")

    # Return the spawned actors so they can be destroyed in the main function.
    return vehicle, simulated_sensor

def _setup_vehicle(world, config):
    """
    A copy of the vehicle setup function from the spawn script.
    """
    bp_library = world.get_blueprint_library()
    map_ = world.get_map()
    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))
    bp.set_attribute("ros_name", config.get("id"))
    print('Vehicle attributes set')
    return world.spawn_actor(bp, map_.get_spawn_points()[0], attach_to=None)

def main(args):
    """
    Main function to run the test script.
    """
    client = None
    original_settings = None
    vehicle = None
    simulated_sensor = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        
        world = client.get_world()
        original_settings = world.get_settings()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        with open(args.file) as f:
            config = json.load(f)

        print('Loaded configuration from stack.json')

        # The _run_test function now returns the spawned actors
        vehicle, simulated_sensor = _run_test(client, config)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        # Destroy the actors that were created in the try block, ensuring cleanup on exit.
        if simulated_sensor:
            simulated_sensor.destroy()
        if vehicle:
            vehicle.destroy()
        
        if original_settings:
            world.apply_settings(original_settings)
        if client:
            logging.info("Disconnecting from CARLA server.")
            # client.disconnect()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Test script for CARLACDASimAPI')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', default='stack.json', help='Configuration file to be used (default: stack.json)')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')

    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    main(args)