# ruff: noqa

import contextlib
import dataclasses
import faulthandler
import signal
import time
import cv2
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
import zmq

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 100


@dataclasses.dataclass
class Args:
    # Rollout parameters
    max_timesteps: int = 600000
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    use_overview_image: bool = False  # whether to use overview camera image
    use_chunk_delta: bool = False  # whether to use chunk delta
    instruction: str | None = None


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = 5555
    socket.bind(f"tcp://*:{port}")
    print(f"Server started on port {port}")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        instruction = args.instruction
        if instruction is None:
            instruction = input("Enter instruction: ")

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        # Maintain a small open-loop action chunk predicted from the latest policy call
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                req = socket.recv_pyobj()
                if "reset" in req:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY)
                    socket.send_pyobj({})
                    pred_action_chunk = None
                    continue

                obs = req["obs"]
                if "base_pose" in obs.keys():
                    curr_obs = _extract_observation(
                        obs,
                        use_overview_image=args.use_overview_image,
                    )
                else:
                    curr_obs = _extract_observation_2d(
                        obs,
                    )

                # Predict a new chunk if needed
                if pred_action_chunk is None or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    if "base_pose" in obs.keys():
                        request_data = {
                            "observation": {
                                IMAGE_KEYS[0]: image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224),
                                IMAGE_KEYS[1]: image_tools.resize_with_pad(
                                    curr_obs["observation/wrist_image"], 224, 224
                                ),
                                "state": curr_obs["observation/state"],
                            },
                            "prompt": instruction,
                            "batch_size": None,
                        }
                    else:
                        request_data = {
                            "observation": {
                                IMAGE_KEYS[0]: image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224),
                                "state": curr_obs["observation/state"],
                            },
                            "prompt": instruction,
                            "batch_size": None,
                        }
                    if args.use_overview_image:
                        request_data["observation"][IMAGE_KEYS[2]] = image_tools.resize_with_pad(
                            curr_obs["observation/overview_image"], 224, 224
                        )

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # Get response from policy server (may contain actions and/or reasoning)
                        st = time.time()
                        response = policy_client.infer(request_data)
                        # ()

                        # Extract actions from response (either pre-parsed or parse from reasoning)
                        if "actions" in response and response["actions"] is not None:
                            pred_action_chunk = np.asarray(response["actions"])
                            pred_action_chunk = pred_action_chunk.copy()
                            if args.use_chunk_delta:
                                pred_action_chunk[:, 3:10] += curr_obs["observation/state"][3:10]
                        else:
                            raise NotImplementedError

                        et = time.time()
                        inference_time = et - st
                        print(f"Time taken for inference: {inference_time}")

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                if args.use_chunk_delta:
                    action = np.concatenate(
                        [action[:3], action[3:10] - curr_obs["observation/state"][3:10], action[10:11]]
                    )
                actions_from_chunk_completed += 1

                if "base_pose" not in obs.keys():
                    if "overview_image" in obs.keys():
                        rep = {
                            "action": {
                                "robot_actions": action[:11],
                                "inference_time": inference_time,
                            }
                        }
                    else:
                        rep = {
                            "action": {
                                "robot_actions": action[:5],
                                "inference_time": inference_time,
                            }
                        }
                elif "arm_qpos" in obs.keys():
                    assert len(action) == 11, f"Expected action of length 11, got {len(action)}"
                    rep = {
                        "action": {
                            "base_pose": action[:3],
                            "arm_qpos": action[3:10],
                            "gripper_pos": action[10:11],
                        }
                    }
                else:
                    assert len(action) == 11, f"Expected action of length 11, got {len(action)}"
                    rep = {
                        "action": {
                            "base_pose": action[:3],
                            "arm_pos": action[3:6],
                            "arm_quat": action[6:10],
                            "gripper_pos": action[10:11],
                        }
                    }
                socket.send_pyobj(rep)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        answer = input("Do one more eval? (enter y or n) ")
        if "n" in answer.lower():
            break


def quat_to_r6(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to the first 6 elements of rotation matrix:
    [r11, r12, r13, r21, r22, r23].
    """
    assert q.shape[-1] == 4, "Input must be quaternion (w, x, y, z)"

    q = q / np.linalg.norm(q)
    quat_xyzw = np.roll(q, -1)  # scipy expects [x, y, z, w]
    R_mat = R.from_quat(quat_xyzw).as_matrix()

    r6 = np.concatenate([R_mat[0, :], R_mat[1, :]])  # first 2 rows
    return r6


def _extract_observation(obs_dict, use_overview_image=False):
    base_image = cv2.imdecode(obs_dict["base_image"], cv2.IMREAD_COLOR)
    wrist_image = cv2.imdecode(obs_dict["wrist_image"], cv2.IMREAD_COLOR)
    if use_overview_image:
        overview_image = cv2.imdecode(obs_dict["overview_image"], cv2.IMREAD_COLOR)
    if "arm_qpos" in obs_dict.keys():
        state = np.concatenate([obs_dict["base_pose"], obs_dict["arm_qpos"], np.array(obs_dict["gripper_pos"])])
    else:
        state = np.concatenate(
            [
                obs_dict["base_pose"],
                obs_dict["arm_pos"],
                quat_to_r6(obs_dict["arm_quat"]),
                np.array(obs_dict["gripper_pos"]),
            ]
        )

    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "observation/overview_image": overview_image if use_overview_image else None,
    }


def _extract_observation_2d(obs_dict):
    if "overview_image" in obs_dict.keys():
        overview_image = cv2.imdecode(obs_dict["overview_image"], cv2.IMREAD_COLOR)
        base_image = cv2.imdecode(obs_dict["base_image"], cv2.IMREAD_COLOR)
        wrist_image = cv2.imdecode(obs_dict["wrist_image"], cv2.IMREAD_COLOR)
        state = obs_dict["robot_state"]

        return {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/overview_image": overview_image,
            "observation/state": state,
        }
    else:
        image = cv2.imdecode(obs_dict["image"], cv2.IMREAD_COLOR)
        state = obs_dict["robot_state"]

        return {
            "observation/image": image,
            "observation/state": state,
        }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    main(args)
