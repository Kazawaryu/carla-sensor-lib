import argparse
import logging
import carla
import json
import os
import sys
import yaml

# Add the parent directory of the API to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CarlaCDASimAPI import CarlaCDASimAPI

def _run_test(client, carla_config, simulated_sensor_config, noise_config):
    """
    Runs the core test logic.
    """
    world = client.get_world()
    
    # Use the existing _setup_vehicle function to spawn a vehicle
    vehicle = _setup_vehicle(world, carla_config)
    _ = _setup_sensors(world, vehicle, carla_config.get("sensors", []))

    print("set up vehicle")
    
    # Instantiate the API and try to create a sensor
    api = CarlaCDASimAPI.build_from_world(world)
    
    # Get the sensor configuration from the stack.json file
    sensor_config = carla_config.get("sensors", [])[1]

    print("Creating simulated sensor...")
    print(sensor_config)
    
    # Use the create_simulated_semantic_lidar_sensor function from the API
    simulated_sensor = api.create_simulated_semantic_lidar_sensor(
        simulated_sensor_config=simulated_sensor_config,
        carla_sensor_config=sensor_config["attributes"],
        noise_model_config=noise_config,
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

    print("Entering world tick loop")

    vehicle.set_autopilot(True)
    world.tick()
    try:
        while True:
            _ = world.tick()
    except KeyboardInterrupt:
        return vehicle, simulated_sensor

def _setup_vehicle(world, config):
    logging.debug("Spawning vehicle: {}".format(config.get("type")))

    bp_library = world.get_blueprint_library()
    map_ = world.get_map()

    bp = bp_library.filter(config.get("type"))[0]
    bp.set_attribute("role_name", config.get("id"))
    bp.set_attribute("ros_name", config.get("id")) 

    return  world.spawn_actor(
        bp,
        map_.get_spawn_points()[0],
        attach_to=None)


def _setup_sensors(world, vehicle, sensors_config):
    bp_library = world.get_blueprint_library()

    sensors = []
    for sensor in sensors_config:
        if sensor.get("id") == "lidar":
            continue # Skip lidar for simulated version
        logging.debug("Spawning sensor: {}".format(sensor))

        bp = bp_library.filter(sensor.get("type"))[0]
        bp.set_attribute("ros_name", sensor.get("id")) 
        bp.set_attribute("role_name", sensor.get("id"))
        for key, value in sensor.get("attributes", {}).items():
            print(f"Setting attribute {key} to {value} for {sensor.get('id')}")
            bp.set_attribute(str(key), str(value))

        wp = carla.Transform(
            location=carla.Location(x=sensor["spawn_point"]["x"], y=-sensor["spawn_point"]["y"], z=sensor["spawn_point"]["z"]),
            rotation=carla.Rotation(roll=sensor["spawn_point"]["roll"], pitch=-sensor["spawn_point"]["pitch"], yaw=-sensor["spawn_point"]["yaw"])
        )

        sensors.append(
            world.spawn_actor(
                bp,
                wp,
                attach_to=vehicle
            )
        )

        sensors[-1].enable_for_ros()

    return sensors


def main(args):

    world = None
    vehicle = None
    sensors = []
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        with open(args.file) as f:
            carla_config = json.load(f)
        with open(args.sensor_config) as f:
            simulated_sensor_config = yaml.safe_load(f)
        with open(args.noise_config) as f:
            noise_config = yaml.safe_load(f)

        # vehicle = _setup_vehicle(world, config=carla_config)
        # sensors = _setup_sensors(world, vehicle, carla_config.get("sensors", []))

        _ = world.tick()

        logging.info("Running...")

        vehicle, simulated_sensor = _run_test(client, carla_config, simulated_sensor_config.get("simulated_sensor", {}), noise_config)
        

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:
        if original_settings:
            world.apply_settings(original_settings)

        for sensor in sensors:
            sensor.destroy()

        if vehicle:
            vehicle.destroy()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Test script for CARLACDASimAPI')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f1', '--file', default='stack.json', help='Configuration file to be used (default: stack.json)')
    argparser.add_argument('-f2', '--sensor_config', default='../config/simulated_sensor_config.yaml', help='Simulated sensor configuration file to be used (default: config/simulated_sensor_config.yaml)')
    argparser.add_argument('-f3', '--noise_config', default='../config/noise_model_config.yaml', help='Noise model config .yaml file')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')

    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    main(args)