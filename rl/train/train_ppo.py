"""PPO Training Script for TurtleBot3 LiDAR Navigation.

Specification: docs/Sim2Real_LiDAR_Navigation_Pipeline.pdf (Section 7, 10)
Uses Stable-Baselines3 PPO with MlpPolicy, continuous actions.

Features:
- VecNormalize with norm_obs=False, norm_reward=True
- Periodic checkpoints every 50k steps
- Periodic validation evaluation every 100k steps
- TensorBoard logging with reward components
- Run metadata: config YAML, git hash, pip freeze, seeds
- Kill-and-resume from last checkpoint
"""

from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure as sb3_configure

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from rl.env.tb3_navigation_env import TurtleBot3NavigationEnv
from rl.train.watchdog import GazeboWatchdog


def load_config(config_path: str = "configs/ppo.yaml") -> Dict[str, Any]:
    """Load PPO training configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env_fn(
    rank: int = 0,
    seed: int = 0,
    use_gazebo: bool = False,
    policy_name: str = "P1",
    config: Optional[Dict[str, Any]] = None,
    run_path: Optional[Path] = None,
):
    """Factory creating a picklable environment initializer for DummyVecEnv or SubprocVecEnv."""
    cfg = config or {}
    port = cfg.get('gazebo_port_base', 11345) + rank
    domain_id = cfg.get('ros_domain_id_base', 30) + rank

    def _init() -> gym.Env:
        watchdog = None
        if use_gazebo:
            os.environ['GAZEBO_MASTER_URI'] = f"http://localhost:{port}"
            os.environ['ROS_DOMAIN_ID'] = str(domain_id)
            os.environ['TURTLEBOT3_MODEL'] = 'burger'
            os.environ['GAZEBO_MODEL_DATABASE_URI'] = ''
            os.environ['PYTHONUNBUFFERED'] = '1'

            diag_path = None
            if run_path:
                diag_path = run_path / f"watchdog_diagnostics_w{rank}.json"
            watchdog = GazeboWatchdog(
                timeout_sec=cfg.get('watchdog_timeout_sec', 2.0),
                port=port,
                domain_id=domain_id,
                diagnostics_path=diag_path,
            )
            watchdog.start_simulator()

        worker_seed = seed + rank * 1000
        randomizer = None
        if policy_name == "P2":
            from rl.env.randomizers import SensorRandomizer
            randomizer = SensorRandomizer(seed=worker_seed)
        elif policy_name == "P3":
            from rl.env.randomizers import DynamicsRandomizer
            randomizer = DynamicsRandomizer(seed=worker_seed)
        elif policy_name == "P4":
            from rl.env.randomizers import CombinedRandomizer
            randomizer = CombinedRandomizer(seed=worker_seed)

        env = TurtleBot3NavigationEnv(
            use_gazebo=use_gazebo,
            seed=worker_seed,
            watchdog=watchdog,
            randomizer=randomizer,
        )

        # If watchdog started a simulator, wrap env.close() to ensure simulator termination
        if watchdog is not None:
            orig_close = env.close
            def _close_with_watchdog():
                try:
                    orig_close()
                finally:
                    watchdog.terminate_simulator()
            env.close = _close_with_watchdog

        return env

    return _init


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def get_pip_freeze() -> str:
    """Get pip freeze output."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


class RewardComponentLogger(BaseCallback):
    """Custom callback to log reward components to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_successes = 0
        self._episode_collisions = 0
        self._episode_timeouts = 0
        self._n_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._n_episodes += 1
                self._episode_rewards.append(info['episode']['r'])
                self._episode_lengths.append(info['episode']['l'])

            if info.get('is_success', False):
                self._episode_successes += 1
            if info.get('is_collision', False):
                self._episode_collisions += 1

        # Log every 100 episodes
        if self._n_episodes > 0 and self._n_episodes % 100 == 0:
            if self._episode_rewards:
                self.logger.record("custom/mean_reward_100ep",
                                   np.mean(self._episode_rewards[-100:]))
                self.logger.record("custom/mean_length_100ep",
                                   np.mean(self._episode_lengths[-100:]))
            self.logger.record("custom/total_episodes", self._n_episodes)
            self.logger.record("custom/total_successes", self._episode_successes)
            self.logger.record("custom/total_collisions", self._episode_collisions)
            if self._n_episodes > 0:
                self.logger.record("custom/success_rate",
                                   self._episode_successes / self._n_episodes)

        return True


def train_ppo(
    policy_name: str = "P1",
    seed: int = 0,
    config_path: str = "configs/ppo.yaml",
    run_dir: Optional[str] = None,
    resume: bool = False,
    use_gazebo: bool = False,
    n_workers: Optional[int] = None,
) -> str:
    """Train a PPO policy.

    Args:
        policy_name: One of P1, P2, P3, P4.
        seed: Training seed.
        config_path: Path to PPO config YAML.
        run_dir: Output directory (auto-generated if None).
        resume: If True, resume from last checkpoint.
        use_gazebo: If True, use Gazebo physics.
        n_workers: Number of parallel workers (defaults to config['n_workers'] or 1).

    Returns:
        Path to the final model file.
    """
    config = load_config(config_path)

    # Worker count resolution
    if n_workers is None:
        n_workers = int(config.get('n_workers', 1))

    # Limit torch threads
    torch_threads = config.get('torch_threads', 2)
    torch.set_num_threads(torch_threads)
    os.environ['OMP_NUM_THREADS'] = str(torch_threads)
    os.environ['MKL_NUM_THREADS'] = str(torch_threads)

    # Run directory
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/{policy_name}_s{seed}_{timestamp}"
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    meta = {
        "policy_name": policy_name,
        "seed": seed,
        "n_workers": n_workers,
        "git_hash": get_git_hash(),
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16],
        "start_time": datetime.now().isoformat(),
        "config": config,
    }
    (run_path / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding='utf-8'
    )
    (run_path / "pip_freeze.txt").write_text(get_pip_freeze(), encoding='utf-8')

    # Create vectorized environment
    env_fns = [
        make_env_fn(
            rank=i,
            seed=seed,
            use_gazebo=use_gazebo,
            policy_name=policy_name,
            config=config,
            run_path=run_path,
        )
        for i in range(n_workers)
    ]
    if n_workers > 1:
        base_env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        base_env = DummyVecEnv(env_fns)

    # VecNormalize: norm_obs=False (manual scaling), norm_reward=True
    env = VecNormalize(
        base_env,
        norm_obs=False,
        norm_reward=True,
        gamma=config.get('gamma', 0.99),
    )

    # Evaluation environment (isolated kinematic evaluation)
    eval_fn = make_env_fn(
        rank=99,
        seed=seed + 10_000,
        use_gazebo=False,
        policy_name=policy_name,
        config=config,
        run_path=run_path,
    )
    eval_env = DummyVecEnv([eval_fn])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    try:
        # Policy kwargs
        policy_kwargs = {
            'net_arch': dict(
                pi=config['policy_kwargs']['net_arch']['pi'],
                vf=config['policy_kwargs']['net_arch']['vf'],
            ),
            'activation_fn': torch.nn.Tanh,
            'log_std_init': config['policy_kwargs'].get('log_std_init', -0.5),
        }

        # Check for resume
        checkpoint_path = None
        vecnorm_path = None
        completed_timesteps = 0

        if resume:
            checkpoints = sorted(run_path.glob("checkpoint_*_steps.zip")) + sorted((run_path / "checkpoints").glob("checkpoint_*_steps.zip"))
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])
                vecnorm_path = str(checkpoints[-1]).replace('.zip', '_vecnormalize.pkl')
                if not Path(vecnorm_path).exists():
                    vecnorm_path = None
                print(f"Resuming from checkpoint: {checkpoint_path}")

        if checkpoint_path:
            model = PPO.load(
                checkpoint_path,
                env=env,
                device=config.get('device', 'cpu'),
            )
            if vecnorm_path and Path(vecnorm_path).exists():
                env = VecNormalize.load(vecnorm_path, env)
            completed_timesteps = model.num_timesteps
            print(f"Loaded model num_timesteps = {completed_timesteps}")
        else:
            model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=config.get('learning_rate', 3e-4),
                n_steps=config.get('n_steps', 2048),
                batch_size=config.get('batch_size', 256),
                n_epochs=config.get('n_epochs', 10),
                gamma=config.get('gamma', 0.99),
                gae_lambda=config.get('gae_lambda', 0.95),
                clip_range=config.get('clip_range', 0.2),
                ent_coef=config.get('ent_coef', 0.0),
                vf_coef=config.get('vf_coef', 0.5),
                max_grad_norm=config.get('max_grad_norm', 0.5),
                policy_kwargs=policy_kwargs,
                verbose=config.get('verbose', 1),
                seed=seed,
                device=config.get('device', 'cpu'),
                tensorboard_log=str(run_path / "tensorboard"),
            )

        # Callbacks with frequency mapping for multi-worker environments
        # SB3 callbacks trigger every save_freq *calls* to env.step()
        # With n_workers, each call advances aggregate num_timesteps by n_workers.
        agg_checkpoint_interval = config.get('checkpoint_interval', 50_000)
        agg_eval_interval = config.get('eval_interval', 100_000)
        cb_checkpoint_freq = max(1, agg_checkpoint_interval // n_workers)
        cb_eval_freq = max(1, agg_eval_interval // n_workers)

        checkpoint_callback = CheckpointCallback(
            save_freq=cb_checkpoint_freq,
            save_path=str(run_path / "checkpoints"),
            name_prefix="checkpoint",
            save_vecnormalize=True,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_path / "best_model"),
            log_path=str(run_path / "eval_logs"),
            eval_freq=cb_eval_freq,
            n_eval_episodes=config.get('eval_episodes', 20),
            deterministic=True,
        )

        reward_logger = RewardComponentLogger()
        callbacks = CallbackList([checkpoint_callback, eval_callback, reward_logger])

        # Train with exact remaining timesteps
        total_timesteps = config.get('total_timesteps', 1_000_000)
        remaining_timesteps = max(0, total_timesteps - completed_timesteps)

        print(f"\n{'=' * 60}")
        print(f"Training {policy_name} (seed={seed}, workers={n_workers})")
        print(f"  Target Budget: {total_timesteps:,} steps (Remaining: {remaining_timesteps:,} steps)")
        print(f"  Network: 2x128 tanh MLP")
        print(f"  Device: {config.get('device', 'cpu')}")
        print(f"  Run dir: {run_path}")
        print(f"{'=' * 60}\n")

        if remaining_timesteps <= 0:
            print(f"Target steps ({total_timesteps}) already completed in checkpoint.")
            return str(run_path / "final_model")

        start_time = time.perf_counter()

        model.learn(
            total_timesteps=remaining_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            log_interval=config.get('log_interval', 10),
            progress_bar=True,
        )

        elapsed = time.perf_counter() - start_time
        steps_per_sec = remaining_timesteps / max(elapsed, 0.001)

        # Save final model
        final_model_path = str(run_path / "final_model")
        model.save(final_model_path)
        env.save(str(run_path / "final_vecnormalize.pkl"))

        # Save training summary
        summary = {
            "policy_name": policy_name,
            "seed": seed,
            "n_workers": n_workers,
            "total_timesteps": model.num_timesteps,
            "elapsed_seconds": round(elapsed, 2),
            "steps_per_second": round(steps_per_sec, 2),
            "final_model_path": final_model_path,
            "end_time": datetime.now().isoformat(),
        }
        (run_path / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding='utf-8'
        )

        print(f"\n{'=' * 60}")
        print(f"Training Complete: {policy_name} (seed={seed})")
        print(f"  Elapsed: {elapsed:.1f}s ({steps_per_sec:.1f} steps/s)")
        print(f"  Model: {final_model_path}")
        print(f"{'=' * 60}")

        return final_model_path

    finally:
        env.close()
        eval_env.close()


def main():
    """Main entry point for training."""
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO for TurtleBot3 Navigation")
    parser.add_argument("--policy", type=str, default="P1",
                        choices=["P1", "P2", "P3", "P4"],
                        help="Policy variant (default: P1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Training seed (default: 0)")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml",
                        help="Config file path")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Output run directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--gazebo", action="store_true",
                        help="Use Gazebo physics")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default from config: 1)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total_timesteps")
    args = parser.parse_args()

    if args.steps is not None:
        config = load_config(args.config)
        config['total_timesteps'] = args.steps
        temp_config = Path(args.config).parent / f"ppo_temp_{args.policy}_s{args.seed}.yaml"
        with open(temp_config, 'w') as f:
            yaml.safe_dump(config, f)
        args.config = str(temp_config)

    train_ppo(
        policy_name=args.policy,
        seed=args.seed,
        config_path=args.config,
        run_dir=args.run_dir,
        resume=args.resume,
        use_gazebo=args.gazebo,
        n_workers=args.workers,
    )


if __name__ == '__main__':
    main()
