"""Inspect the training environment with stationary commands; this does not train PPO."""
import argparse
import sys
from pathlib import Path
import gymnasium as gym
import torch
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Oarbot-Iteration1-Mobile-v0')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--steps', type=int, default=0, help='Zero keeps the preview open until Ctrl+C.')
parser.add_argument('--disable_fabric', action='store_true')
add_launcher_args(parser)
args, remaining = setup_preset_cli(parser)
sys.argv = [sys.argv[0]]+remaining
import iteration_1.tasks
from ui_helpers import configure_view, hide_camera_icons

def main():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = resolve_task_config(args.task, '')
    cfg.scene.num_envs = args.num_envs
    if args.disable_fabric:
        cfg.sim.use_fabric = False
    configure_view(cfg)
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        hide_camera_icons()
        try:
            with torch.inference_mode():
                env.reset()
                action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                action[:, 9:] = -1.
                print('OARBOT_PREVIEW_READY: PPO is OFF. Connect WebRTC to 128.113.120.75.', flush=True)
                step = 0
                while not args.steps or step < args.steps:
                    env.step(action)
                    step += 1
                    if step in (16, 64):
                        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                        folder = root/'verification/preview'
                        folder.mkdir(parents=True, exist_ok=True)
                        viewport = get_active_viewport()
                        if viewport:
                            capture_viewport_to_file(viewport, str(folder/f'viewport_{step}.png'))
                        for name in ('room_rgb','room_depth','wrist_rgb','wrist_depth'):
                            sensor = env.unwrapped.scene[name]
                            kind = 'rgb' if name.endswith('rgb') else 'depth'
                            data = sensor.data.output[kind].torch
                            print('CAMERA_PREVIEW', name, list(data.shape), flush=True)
                            from PIL import Image
                            pixels = data[0,...,:3].byte() if kind == 'rgb' else data[0,...,0].nan_to_num(10).clamp(0,10).mul(25.5).byte()
                            Image.fromarray(pixels.cpu().numpy()).save(folder/f'{name}_{step}.png')
        except KeyboardInterrupt:
            print('Preview stopped.', flush=True)
        finally:
            env.close()

if __name__ == '__main__':
    main()
