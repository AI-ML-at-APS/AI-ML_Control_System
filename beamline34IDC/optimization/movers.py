import numpy as np
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement

def get_movement(movement):
    movement_types = {'relative': Movement.RELATIVE,
                      'absolute': Movement.ABSOLUTE}
    if movement in movement_types:
        return movement_types[movement]
    if movement in movement_types.values():
        return movement
    raise ValueError

def get_motor_move_fn(focusing_system, motor):
    motor_move_fns = {'hkb_4': focusing_system.move_hkb_motor_4_translation,
                      'hkb_3': focusing_system.move_hkb_motor_3_pitch,
                      'hkb_q': focusing_system.change_hkb_shape,
                      'vkb_4': focusing_system.move_vkb_motor_4_translation,
                      'vkb_3': focusing_system.move_vkb_motor_3_pitch,
                      'vkb_q': focusing_system.change_vkb_shape}
    if motor in motor_move_fns:
        return motor_move_fns[motor]
    if motor in motor_move_fns.values():
        return motor
    raise ValueError


def move_motors(focusing_system, motors, translations, movement='relative'):
    movement = get_movement(movement)
    if np.ndim(motors) == 0:
        motors = [motors]
    if np.ndim(translations) == 0:
        translations = [translations]
    for motor, trans in zip(motors, translations):
        motor_move_fn = get_motor_move_fn(focusing_system, motor)
        motor_move_fn(trans, movement=movement)
    return focusing_system


def get_motor_absolute_position_fn(focusing_system, motor):
    motor_get_pos_fns = {'hkb_4': focusing_system.get_hkb_motor_4_translation,
                         'hkb_3': focusing_system.get_hkb_motor_3_pitch,
                         'hkb_q': focusing_system.get_hkb_q_distance,
                         'vkb_4': focusing_system.get_vkb_motor_4_translation,
                         'vkb_3': focusing_system.get_vkb_motor_3_pitch,
                         'vkb_q': focusing_system.get_vkb_q_distance}
    if motor in motor_get_pos_fns:
        return motor_get_pos_fns[motor]
    if motor in motor_get_pos_fns.values():
        return motor
    raise ValueError


def get_absolute_positions(focusing_system, motors):

    if np.ndim(motors) == 0:
        motors = [motors]

    positions = []
    for motor in motors:
        position = get_motor_absolute_position_fn(focusing_system, motor)()
        positions.append(position)
    return positions