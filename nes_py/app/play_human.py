"""A method to play gym environments using human IO inputs."""
import time

import gymnasium as gym
from pyglet import clock

from nes_py._image_viewer import ImageViewer


# the sentinel value for "No Operation"
_NOP = 0


def play_human(env: gym.Env, callback=None) -> None:
    """
    Play the environment using keyboard as a human.

    Args:
        env: the initialized gym environment to play
        callback: a callback to receive output from the environment

    Returns:
        None

    """
    # ensure the observation space is a box of pixels
    assert isinstance(env.observation_space, gym.spaces.box.Box)
    # ensure the observation space is either B&W pixels or RGB Pixels
    obs_s = env.observation_space
    is_bw = len(obs_s.shape) == 2
    is_rgb = len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3]
    assert is_bw or is_rgb
    # get the mapping of keyboard keys to actions in the environment
    if hasattr(env, 'get_keys_to_action'):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, 'get_keys_to_action'):
        keys_to_action = env.unwrapped.get_keys_to_action()
    else:
        raise ValueError('env has no get_keys_to_action method')
    # create the image viewer
    viewer = ImageViewer(
        env.spec.id if env.spec is not None else env.__class__.__name__,
        env.observation_space.shape[0], # height
        env.observation_space.shape[1], # width
        monitor_keyboard=True,
        relevant_keys=set(sum(map(list, keys_to_action.keys()), [])+[ord('8'), ord('9')])
    )
    # create a done flag for the environment
    done = True
    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata['render_fps']
    last_frame_time = 0
    snapshot = None
    # start the main game loop
    try:
        while True:
            current_frame_time = time.time()
            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue
            # save frame beginning time for next refresh
            last_frame_time = current_frame_time
            # clock tick
            clock.tick()
            # reset if the environment is done
            if done:
                done = False
                state = env.reset()
                viewer.show(env.unwrapped.screen)
            # unwrap the action based on pressed relevant keys
            action = _NOP
            if ord('8') in viewer.pressed_keys:
                snapshot = env.dump_state()
                print(f'len(snapshot): {len(snapshot)}')
            elif ord('9') in viewer.pressed_keys:
                env.load_state(snapshot)
            else:
                action = keys_to_action.get(viewer.pressed_keys, _NOP)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            viewer.show(env.unwrapped.screen)
            
            # pass the observation data through the callback
            if callback is not None:
                callback(state, action, reward, done, next_state)
            
            state = next_state
            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                break
    except KeyboardInterrupt:
        pass

    viewer.close()
    env.close()


# explicitly define the outward facing API of the module
__all__ = [play_human.__name__]
