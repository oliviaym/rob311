import sys
import threading
import time
import numpy as np
from threading import Thread
from MBot.Messages.message_defs import mo_states_dtype, mo_cmds_dtype, mo_pid_params_dtype
from MBot.SerialProtocol.protocol import SerialProtocol
from DataLogger import dataLogger
from ps4_controller_api import ROB311BTController
from rtplot import client
from scipy.signal import butter, lfilter, filtfilt
from simple_pid import PID
from pyPS4Controller.controller import Controller
import board
import adafruit_dotstar as dotstar
from enum import Enum
from collections import deque
import ROB311_filtering_demo as filt

# ---------------------------------------------------------------------------
"""
ROB 311 - Ball-bot Stability Controller Walkthrough [Kp]

This program uses a soft realtime loop to enforce loop timing. Soft real time loop is a  class
designed to allow clean exits from infinite loops with the potential for post-loop cleanup operations executing.

The Loop Killer object watches for the key shutdown signals on the UNIX operating system (which runs on the PI)
when it detects a shutdown signal, it sets a flag, which is used by the Soft Realtime Loop to stop iterating.
Typically, it detects the CTRL-C from your keyboard, which sends a SIGTERM signal.

the function_in_loop argument to the Soft Realtime Loop's blocking_loop method is the function to be run every loop.
A typical usage would set function_in_loop to be a method of an object, so that the object could store program state.
See the 'ifmain' for two examples.

Authors: Senthur Raj, Gray Thomas, Yves Nazon and Elliott Rouse 
Neurobionics Lab / Locomotor Control Lab
"""

import signal
import time
from math import sqrt

PRECISION_OF_SLEEP = 0.0001

# Version of the SoftRealtimeLoop library
__version__ = "1.0.0"

class LoopKiller:
    def __init__(self, fade_time=0.0):
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGHUP, self.handle_signal)
        self._fade_time = fade_time
        self._soft_kill_time = None

    def handle_signal(self, signum, frame):
        self.kill_now = True

    def get_fade(self):
        # interpolates from 1 to zero with soft fade out
        if self._kill_soon:
            t = time.time() - self._soft_kill_time
            if t >= self._fade_time:
                return 0.0
            return 1.0 - (t / self._fade_time)
        return 1.0

    _kill_now = False
    _kill_soon = False

    @property
    def kill_now(self):
        if self._kill_now:
            return True
        if self._kill_soon:
            t = time.time() - self._soft_kill_time
            if t > self._fade_time:
                self._kill_now = True
        return self._kill_now

    @kill_now.setter
    def kill_now(self, val):
        if val:
            if self._kill_soon:  # if you kill twice, then it becomes immediate
                self._kill_now = True
            else:
                if self._fade_time > 0.0:
                    self._kill_soon = True
                    self._soft_kill_time = time.time()
                else:
                    self._kill_now = True
        else:
            self._kill_now = False
            self._kill_soon = False
            self._soft_kill_time = None

class SoftRealtimeLoop:
    def __init__(self, dt=0.001, report=False, fade=0.0):
        self.t0 = self.t1 = time.time()
        self.killer = LoopKiller(fade_time=fade)
        self.dt = dt
        self.ttarg = None
        self.sum_err = 0.0
        self.sum_var = 0.0
        self.sleep_t_agg = 0.0
        self.n = 0
        self.report = report

    def __del__(self):
        if self.report:
            print("In %d cycles at %.2f Hz:" % (self.n, 1.0 / self.dt))
            print("\tavg error: %.3f milliseconds" % (1e3 * self.sum_err / self.n))
            print(
                "\tstddev error: %.3f milliseconds"
                % (
                    1e3
                    * sqrt((self.sum_var - self.sum_err**2 / self.n) / (self.n - 1))
                )
            )
            print(
                "\tpercent of time sleeping: %.1f %%"
                % (self.sleep_t_agg / self.time() * 100.0)
            )

    @property
    def fade(self):
        return self.killer.get_fade()

    def run(self, function_in_loop, dt=None):
        if dt is None:
            dt = self.dt
        self.t0 = self.t1 = time.time() + dt
        while not self.killer.kill_now:
            ret = function_in_loop()
            if ret == 0:
                self.stop()
            while time.time() < self.t1 and not self.killer.kill_now:
                if signal.sigtimedwait(
                    [signal.SIGTERM, signal.SIGINT, signal.SIGHUP], 0
                ):
                    self.stop()
            self.t1 += dt
        print("Soft realtime loop has ended successfully.")

    def stop(self):
        self.killer.kill_now = True

    def time(self):
        return time.time() - self.t0

    def time_since(self):
        return time.time() - self.t1

    def __iter__(self):
        self.t0 = self.t1 = time.time() + self.dt
        return self

    def __next__(self):
        if self.killer.kill_now:
            raise StopIteration

        while (
            time.time() < self.t1 - 2 * PRECISION_OF_SLEEP and not self.killer.kill_now
        ):
            t_pre_sleep = time.time()
            time.sleep(
                max(PRECISION_OF_SLEEP, self.t1 - time.time() - PRECISION_OF_SLEEP)
            )
            self.sleep_t_agg += time.time() - t_pre_sleep

        while time.time() < self.t1 and not self.killer.kill_now:
            if signal.sigtimedwait([signal.SIGTERM, signal.SIGINT, signal.SIGHUP], 0):
                self.stop()
        if self.killer.kill_now:
            raise StopIteration
        self.t1 += self.dt
        if self.ttarg is None:
            # inits ttarg on first call
            self.ttarg = time.time() + self.dt
            # then skips the first loop
            return self.t1 - self.t0
        error = time.time() - self.ttarg  # seconds
        self.sum_err += error
        self.sum_var += error**2
        self.n += 1
        self.ttarg += self.dt
        return self.t1 - self.t0

# ---------------------------------------------------------------------------

def register_topics(ser_dev:SerialProtocol):
    ser_dev.serializer_dict[101] = [lambda bytes: np.frombuffer(bytes, dtype=mo_cmds_dtype), lambda data: data.tobytes()]
    ser_dev.serializer_dict[121] = [lambda bytes: np.frombuffer(bytes, dtype=mo_states_dtype), lambda data: data.tobytes()]

# ---------------------------------------------------------------------------
JOYSTICK_SCALE = 32767

FREQ = 200
DT = 1/FREQ

RW = 0.0048
RK = 0.1210
ALPHA = np.deg2rad(45)

MAX_PLANAR_DUTY = 0.8

# ---------------------------------------------------------------------------
# WEIGHTED AVERAGE FILTER PARAMETERS

WMA_WEIGHTS = np.array([0.1, 0.3, 0.4, 0.8])

WMA_WINDOW_SIZE = len(WMA_WEIGHTS)
WMA_NORM = WMA_WEIGHTS/np.sum(WMA_WEIGHTS)

# ---------------------------------------------------------------------------
###############  THESE WILL NEED TO BE CAREFULLY ADJUSTED ##################
# --------------------------------------------------------------------------

# Proportional gains for the stability controllers (X-Z and Y-Z plane)

KP_THETA_X = 8.0                                   # Adjust until the system balances
KP_THETA_Y = 8.0                                   # Adjust until the system balances

KI_THETA_X = 0.002
KI_THETA_Y = 0.002

KD_THETA_X = 0.1                                   
KD_THETA_Y = 0.1

KD_PHI_X = .1                                   
KD_PHI_Y = .1

# ---------------------------------------------------------------------------
#############################################################################


# Wheel rotation to Ball rotation transformation matrix
J11 = 0
J12 = -np.sqrt(3) * RW/ (3 * RK * np.cos(ALPHA))
J13 = -1 * J12

J21 = -2 * RW/(3 * RK * np.cos(ALPHA))
J22 = RW / (3 * RK * np.cos(ALPHA))
J23 = J22

J31 = RW / (3 * RK * np.sin(ALPHA))
J32 = J31
J33 = J31

J = np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])

# ---------------------------------------------------------------------------

def compute_motor_torques(Tx, Ty, Tz):
    '''
    Parameters:
    ----------
    Tx: Torque along x-axis
    Ty: Torque along y-axis
    Tz: Torque along z-axis

    Returns:
    --------
            Ty
            T1
            |
            |
            |
            . _ _ _ _ Tx
           / \
          /   \
         /     \
        /       \
       T2       T3

    T1: Motor Torque 1
    T2: Motor Torque 2
    T3: Motor Torque 3
    '''

    T1 = (-0.3333) * (Tz - (2.8284 * Ty))
    T2 = (-0.3333) * (Tz + (1.4142 * (Ty + 1.7320 * Tx)))
    T3 = (-0.3333) * (Tz + (1.4142 * (Ty - 1.7320 * Tx)))

    return T1, T2, T3

# ---------------------------------------------------------------------------

def compute_phi(psi_1, psi_2, psi_3):
    '''
    Parameters:
    ----------
    psi_1: Encoder rotation (rad) [MOTOR 1]
    psi_2: Encoder rotation (rad) [MOTOR 2]
    psi_3: Encoder rotation (rad) [MOTOR 3]

    Returns:
    --------
    phi_x: Ball rotation along x-axis (rad)
    phi_y: Ball rotation along y-axis (rad)
    phi_z: Ball rotation along z-axis (rad)
    '''

    # Converting counts to rad
    psi = np.array([[psi_1], [psi_2], [psi_3]])

    # phi = J [3x3] * psi [3x1]
    phi = np.matmul(J, psi)

    # returns phi_x, phi_y, phi_z
    return phi[0][0], phi[1][0], phi[2][0]

if __name__ == "__main__":

    # RTPLOT INITIALIZATION ROUTINE

    imu_states = {'names': ['Roll', 'Pitch'],
                    'title': "Orientation",
                    'ylabel': "rad",
                    'xlabel': "time",
                    'colors' : ["r", "g"],
                    'line_width': [2]*2,
                    'yrange': [-2.0 * np.pi, 2.0 * np.pi]
                    }

    ball_states = {'names': ['Phi', 'dPhi', 'ddPhi'],
                    'title': "Ball States",
                    'ylabel': "rad",
                    'xlabel': "time",
                    'colors' : ["r", "g", "b"],
                    'line_width': [2]*3,
                    }

    stability_controller = {'names': ['P', 'I', 'D'],
                    'title': "Stability Controller",
                    'ylabel': "Terms",
                    'xlabel': "time",
                    'colors' : ["r", "g", "b"],
                    'line_width': [2]*3,
                    }

    plot_config = [imu_states]
    client.initialize_plots(plot_config)
    rtplot_data = []

    # --------------------------------------------------------

    trial_num = int(input('Trial Number? '))
    filename = 'ROB311_Stability_Test_%i' % trial_num
    dl = dataLogger(filename + '.txt')

    ser_dev = SerialProtocol()
    register_topics(ser_dev)

    # Init serial
    serial_read_thread = Thread(target = SerialProtocol.read_loop, args=(ser_dev,), daemon=True)
    serial_read_thread.start()

    # controller init
    rob311_bt_controller = ROB311BTController(interface="/dev/input/js0")
    rob311_bt_controller_thread = threading.Thread(target=rob311_bt_controller.listen, args=(10,))
    rob311_bt_controller_thread.start()

    # Local structs
    commands = np.zeros(1, dtype=mo_cmds_dtype)[0]
    states = np.zeros(1, dtype=mo_states_dtype)[0]

    psi_1 = 0.0
    psi_2 = 0.0
    psi_3 = 0.0

    # Motor torques
    T1 = 0.0
    T2 = 0.0
    T3 = 0.0
   
    # WMA FILTER VARIABLES

    theta_x_window = deque(maxlen=WMA_WINDOW_SIZE) # A sliding window of values
    theta_y_window = deque(maxlen=WMA_WINDOW_SIZE) # A sliding window of values

    for _ in range(WMA_WINDOW_SIZE):
        theta_x_window.append(0.0)
        theta_y_window.append(0.0)

    # Desired theta
    desired_theta_x = 0.0
    desired_theta_y = 0.0

    # Error in theta
    error_x = 0.0
    error_y = 0.0
    error_x_prev = 0.0
    error_y_prev = 0.0

    # Cumulative error
    error_x_sum = 0.0
    error_y_sum = 0.0

    # Desired phi

    desired_phi_x = 0.0
    desired_phi_y = 0.0

    # Error in phi
    error_phi_x = 0.0
    error_phi_y = 0.0
    error_phi_x_prev = 0.0
    error_phi_y_prev = 0.0

    # Cumulative error
    error_phi_x_sum = 0.0
    error_phi_y_sum = 0.0

    commands['kill'] = 0.0

    # Time for comms to sync
    time.sleep(1.0)

    ser_dev.send_topic_data(101, commands)

    print('Beginning program!')
    i = 0

    loop_ct = 1.0
    pulse_val = 1.0
    pulse_increment = 0.1

    for t in SoftRealtimeLoop(dt=DT, report=True):
        try:
            states = ser_dev.get_cur_topic_data(121)[0]
            if i == 0:
                t_start = time.time()
            i = i + 1
        except KeyError as e:
            continue

        t_now = time.time() - t_start

        # Define variables for saving / analysis here - below you can create variables from the available states in message_defs.py
        
        # Motor rotations
        psi_1 = states['psi_1']
        psi_2 = states['psi_2']
        psi_3 = states['psi_3']

        # Body lean angles
        #theta_x = (states['theta_roll'])
        #theta_y = (states['theta_pitch'])
        theta_x_window.append(states['theta_roll'])
        theta_y_window.append(states['theta_pitch'])

        theta_x = filt.wma_filter(theta_x_window)
        theta_y = filt.wma_filter(theta_y_window)
        
        # ball speed
        phi_x, phi_y, phi_z = compute_phi(psi_1, psi_2, psi_3)
        desired_phi_x = phi_x + rob311_bt_controller.tz_demo_2
        desired_phi_y = phi_y + rob311_bt_controller.tz_demo_3

        # Controller error terms
        error_x = desired_theta_x - theta_x
        error_y = desired_theta_y - theta_y

        error_x_sum += error_x
        error_y_sum += error_y

        error_phi_x = desired_phi_x - phi_x
        error_phi_y = desired_phi_y - phi_y

        error_phi_x_prev = error_phi_x
        error_phi_y_prev = error_phi_y

        error_phi_x_sum += error_phi_x
        error_phi_y_sum += error_phi_y

        print([error_phi_x,error_phi_y])

        # ---------------------------------------------------------
        # Compute motor torques (T1, T2, and T3) with Tx, Ty, and Tz

        # Proportional controller
        Tx = KP_THETA_X * error_x + KI_THETA_X * error_x_sum + KD_THETA_X * DT * (error_x-error_x_prev) #+ KD_PHI_X * error_phi_x
        Ty = KP_THETA_Y * error_y + KI_THETA_Y * error_y_sum + KD_THETA_X * DT * (error_x-error_x_prev) #+ KD_PHI_Y * error_phi_y

        Tz = rob311_bt_controller.tz_demo_1

        # ---------------------------------------------------------
        # Saturating the planar torques 
        # This keeps the system having the correct torque balance across the wheels in the face of saturation of any motor during the conversion from planar torques to M1-M3
        if np.abs(Tx) > MAX_PLANAR_DUTY:
            Tx = np.sign(Tx) * MAX_PLANAR_DUTY

        if np.abs(Ty) > MAX_PLANAR_DUTY:
            Ty = np.sign(Ty) * MAX_PLANAR_DUTY

        # ---------------------------------------------------------

        T1, T2, T3 = compute_motor_torques(Tx, Ty, Tz)

        # ---------------------------------------------------------

        # phi_x, phi_y, phi_z = compute_phi(psi_1, psi_2, psi_3)

        # ---------------------------------------------------------

        print("Iteration no. {}, T1: {:.2f}, T2: {:.2f}, T3: {:.2f}".format(i, T1, T2, T3))
        commands['motor_1_duty'] = T1
        commands['motor_2_duty'] = T2
        commands['motor_3_duty'] = T3  

        # rtplot
        rtplot_data = [phi_x,phi_y]
        client.send_array(rtplot_data)

        # Construct the data matrix for saving - you can add more variables by replicating the format below
        data = [i] + [t_now] + [theta_x] + [theta_y] + [T1] + [T2] + [T3] + [phi_x] + [phi_y] + [phi_z] + [psi_1] + [psi_2] + [psi_3]
        dl.appendData(data)

        print("Iteration no. {}, THETA X: {:.2f}, THETA Y: {:.2f}".format(i, theta_x, theta_y))
        ser_dev.send_topic_data(101, commands) # Send motor torques
    
  
    dl.writeOut()

    print("Resetting Motor commands.")
    time.sleep(0.25)
    commands['motor_1_duty'] = 0.0
    commands['motor_2_duty'] = 0.0
    commands['motor_3_duty'] = 0.0
    time.sleep(0.25)
    commands['kill'] = 1.0
    time.sleep(0.25)
    ser_dev.send_topic_data(101, commands)
    time.sleep(0.25)

