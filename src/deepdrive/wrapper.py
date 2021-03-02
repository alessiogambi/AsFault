import cv2
import time
import math
import numpy

from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging
from beamngpy.sensors import Camera, GForces, Electrics, Damage
from docutils.nodes import transition

from shapely.geometry import Point

import speed_dreams as sd


def preprocess(img, brightness):

    # Elaborate Frame from BeamNG
    pil_image = img.convert('RGB')
    open_cv_image = numpy.array(pil_image)

    # Convert RGB to BGR. This is important
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # decrease_brightness and resize
    hsv = cv2.cvtColor(cv2.resize(open_cv_image, (280, 210)), cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * brightness
    preprocessed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Check that we are passing to the network values between 0 and 1
    return preprocessed


def translate_steering(original_steering_value):
    # Using a quadratic function might be too much
    # newValue = -1.0 * (0.4 * pow(original_steering_value, 2) + 0.6 * original_steering_value + 0)
    # This seems to over shoot. Maybe it's just a matter of speed and not amount of steering
    newValue = -1.0 * original_steering_value;
    linear_factor = 0.6
    # Dump the controller to compensate oscillations in gentle curve
    if abs(original_steering_value) < 1:
        newValue = linear_factor * newValue

    # print("Steering", original_steering_value, " -> ", newValue)
    return newValue


def main(MAX_SPEED):
    setup_logging()

    # Gains to port TORCS actuators to BeamNG
    # steering_gain = translate_steering()
    acc_gain = 0.5  # 0.4
    brake_gain = 1.0
    # BeamNG images are too bright for DeepDrive
    brightness = 0.4


    # Set up first vehicle
    # ! A vehicle with the name you specify here has to exist in the scenario
    vehicle = Vehicle('egovehicle')

    # Set up sensors
    resolution = (280, 210)
    # Original Settings
    #pos = (-0.5, 1.8, 0.8)  # Left/Right, Front/Back, Above/Below
    # 0.4 is inside
    pos = (0, 2.0, 0.5)  # Left/Right, Front/Back, Above/Below
    direction = (0, 1, 0)
    # direction = (180, 0, 180)

    # FOV 60, MAX_SPEED 100, 20 (3) Hz fails
    # FOV 60, MAX_SPEED 80, 20 (3) Hz Ok
    # FOV 60, MAX_SPEED 80, 12 Hz Ok-ish Oscillations
    # FOV 60, MAX_SPEED 80, 10 Hz Ok-ish Oscillations
    # FOV 40, MAX_SPEED 50, 12 Hz Seems to be fine but drives slower
    # FOV 40, MAX_SPEED 80, 10 Hz Seems to be fine but drives slower

    fov = 60
    # MAX_SPEED = 70
    MAX_FPS = 60
    SIMULATION_STEP = 6
    # Running the controller at 20 hz makes experiments 3 to 4 times slower ! 5 minutes of simulations end up sucking 20 minutes !
    #

    # WORKING SETTINGS: 20 Freq, 90 FOV.
    front_camera = Camera(pos, direction, fov, resolution,
                          colour=True, depth=True, annotation=True)
    electrics = Electrics()

    vehicle.attach_sensor('front_cam', front_camera)
    vehicle.attach_sensor('electrics', electrics)

    # Setup the SHM with DeepDrive
    # Create shared memory object
    Memory = sd.CSharedMemory(TargetResolution=[280, 210])
    # Enable Pause-Mode
    Memory.setSyncMode(True)

    Memory.Data.Game.UniqueRaceID = int(time.time())
    print("Setting Race ID at ", Memory.Data.Game.UniqueRaceID)

    # Setting Max_Speed for the Vehicle.
    # TODO What's this? Maybe some hacky way to pass a parameter which is not supposed to be there...

    Memory.Data.Game.UniqueTrackID = int(MAX_SPEED)
    # Speed is KM/H
    print("Setting speed at ", Memory.Data.Game.UniqueTrackID)
    # Default for AsFault
    Memory.Data.Game.Lanes = 1
    # By default the AI is in charge
    Memory.Data.Control.IsControlling = 1

    deep_drive_engaged = True
    STATE = "NORMAL"

    Memory.waitOnRead()
    if Memory.Data.Control.Breaking == 3.0 or Memory.Data.Control.Breaking == 2.0:
        print("\n\n\nState not reset ! ", Memory.Data.Control.Breaking)
        Memory.Data.Control.Breaking = 0.0
        # Pass the computation to DeepDrive
        # Not sure this will have any effect
        Memory.indicateWrite()

    Memory.waitOnRead()
    if Memory.Data.Control.Breaking == 3.0 or Memory.Data.Control.Breaking == 2.0:
        print("\n\n\nState not reset Again! ", Memory.Data.Control.Breaking)
        Memory.Data.Control.Breaking = 0.0
        # Pass the computation to DeepDrive
        Memory.indicateWrite()


    # Connect to running beamng
    beamng = BeamNGpy('localhost', 64256, home='C://Users//Alessio//BeamNG.research_unlimited//trunk')
    bng = beamng.open(launch=False)
    try:
        bng.set_deterministic()  # Set simulator to be deterministic
        bng.set_steps_per_second(MAX_FPS)  # With 60hz temporal resolution
        # Connect to the existing vehicle (identified by the ID set in the vehicle instance)
        bng.connect_vehicle(vehicle)
        # Put simulator in pause awaiting further inputs
        bng.pause()

        assert vehicle.skt

        # Road interface is not available in BeamNG.research yet
        # Get the road map from the level
        # roads = bng.get_roads()
        # # find the actual road. Dividers lane markings are all represented as roads
        # theRoad = None
        # for road in enumerate(roads):
        #     # ((left, centre, right), (left, centre, right), ...)
        #     # Compute the width of the road
        #     left = Point(road[0][0])
        #     right = Point(road[0][1])
        #     distance = left.distance( right )
        #     if distance < 2.0:
        #         continue
        #     else:
        #         theRoad = road;
        #         break
        #
        # if theRoad is None:
        #     print("WARNING Cannot find the main road of the map")

        while True:
            # Resume the execution
            # 6 steps correspond to 10 FPS with a resolution of 60FPS
            # 5 steps 12 FPS
            # 3 steps correspond to 20 FPS
            bng.step(SIMULATION_STEP)

            # Retrieve sensor data and show the camera data.
            sensors = bng.poll_sensors(vehicle)
            # print("vehicle.state", vehicle.state)

            # # TODO: Is there a way to query for the speed directly ?
            speed = math.sqrt(vehicle.state['vel'][0] * vehicle.state['vel'][0] + vehicle.state['vel'][1] * vehicle.state['vel'][1])
            # Speed is M/S ?
            # print("Speed from BeamNG is: ", speed, speed*3.6)

            imageData = preprocess(sensors['front_cam']['colour'], brightness)

            Height, Width = imageData.shape[:2]
            # print("Image size ", Width, Height)
            # TODO Size of image should be right since the beginning
            Memory.write(Width, Height, imageData, speed)

            # Pass the computation to DeepDrive
            Memory.indicateWrite()

            # Wait for the control commands to send to the vehicle
            # This includes a sleep and will be unlocked by writing data to it
            Memory.waitOnRead()

            # TODO Assumption. As long as the car is out of the road for too long this value stays up
            if Memory.Data.Control.Breaking == 3.0:
                if STATE != "DISABLED":
                    print("Abnormal situation detected. Disengage DeepDrive and enable BeamNG AI")
                    vehicle.ai_set_mode("manual")
                    vehicle.ai_drive_in_lane(True)
                    vehicle.ai_set_speed(MAX_SPEED)
                    vehicle.ai_set_waypoint("waypoint_goal")
                    deep_drive_engaged = False
                    STATE = "DISABLED"
            elif Memory.Data.Control.Breaking == 2.0:
                if STATE != "GRACE":
                    print("Grace period. Deep Driving still disengaged")
                    vehicle.ai_set_mode("manual")
                    vehicle.ai_set_waypoint("waypoint_goal")
                    # vehicle.ai_drive_in_lane(True)
                    STATE = "GRACE"
            else:
                if STATE != "NORMAL":
                    print("DeepDrive re-enabled")
                    # Disable BeamNG AI driver
                    vehicle.ai_set_mode("disabled")
                    deep_drive_engaged = True
                    STATE = "NORMAL"

            # print("State ", STATE, "Memory ",Memory.Data.Control.Breaking )
            if STATE == "NORMAL":
                vehicle.ai_set_mode("disabled")
                # Get commands from SHM
                # Apply Control - not sure cutting at 3 digit makes a difference
                steering = round(translate_steering(Memory.Data.Control.Steering), 3)
                throttle = round(Memory.Data.Control.Accelerating * acc_gain, 3)
                brake = round(Memory.Data.Control.Breaking * brake_gain, 3)

                # Apply commands
                vehicle.control(throttle=throttle, steering=steering, brake=brake)
                #
                # print("Suggested Driving Actions: ")
                # print(" Steer: ", steering)
                # print(" Accel: ", throttle)
                # print(" Brake: ", brake)

    finally:
        bng.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-speed', type=int, default=70, help='Speed Limit in KM/H')
    args = parser.parse_args()
    print("Setting max speed to", args.max_speed)
    main(args.max_speed)
