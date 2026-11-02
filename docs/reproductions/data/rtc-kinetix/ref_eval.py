"""Arm R: official eval_flow.eval on all 12 levels for selected (method, d, s); appends JSONL, skips done rows.

RTC_SOLVED: success read from the terminal step as the last frame-skip substep's GoalR (default; cf7453e, the paper),
``terminal_reward`` (its reward, the max over substeps, is +1) or ``first_terminal`` (the first substep that ends the
episode is a goal: Kinetix v3).  RTC_PHYSICS_PRECISION=highest: f32 dots in the physics (GPU default allows TF32).
usage: python ref_eval.py <rtc_repo_src> <run_path> <out.jsonl> <group> ; group in {core, horizon, bid, bid_h, all}
run_path holds numeric checkpoint dirs (e.g. 7/, 31/) with policies/worlds_l_<level>.pkl; strong = last, weak = first.
"""

import contextlib
import functools
import os
import json
import pathlib
import pickle
import sys
import time

src, run_path, out, group = sys.argv[1:5]
sys.path.insert(0, src)
import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kinetix.environment.env as kenv  # noqa: E402
import kinetix.environment.env_state as kenv_state  # noqa: E402
from jax.experimental import shard_map  # noqa: E402

import kinetix.environment.wrappers as W  # noqa: E402

SOLVED = os.environ.get("RTC_SOLVED", "goal_last_substep")
PRECISION = os.environ.get("RTC_PHYSICS_PRECISION")


def _engine_step(self, env_state, action, env_params):
    """cf7453e's engine_step, plus the first terminal substep's goal and an optional dot precision."""
    with jax.default_matmul_precision(PRECISION) if PRECISION else contextlib.nullcontext():

        def single(state, _):
            state, mfolds = self.physics_engine.step(state, env_params, action)
            reward, info = self.compute_reward_info(state, mfolds)
            return state, (reward, reward != 0, info)

        env_state, (rewards, dones, infos) = jax.lax.scan(
            single, env_state, None, length=self.static_env_params.frame_skip
        )
    env_state = env_state.replace(timestep=env_state.timestep + 1)
    nan = jax.tree.reduce(jnp.logical_or, jax.tree.map(lambda x: jnp.isnan(x).any(), env_state), False)
    done = dones.sum() > 0 | nan  # sic (cf7453e's precedence)
    done |= env_state.timestep >= env_params.max_timesteps
    info = jax.tree.map(lambda x: x[-1], infos)
    info["first_terminal_goal"] = (dones.sum() > 0) & (rewards[dones.argmax()] > 0)
    return self.get_obs(env_state), env_state, rewards.max(), done, info


kenv.BasePhysicsEnv.engine_step = _engine_step


class _LogWrapper(W.LogWrapper):
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = super().step(key, state, action, params)
        if SOLVED == "terminal_reward":
            info["returned_episode_solved"] = reward > 0
        elif SOLVED == "first_terminal":
            info["returned_episode_solved"] = info["first_terminal_goal"]
        return obs, state, reward, done, info


W.LogWrapper = _LogWrapper

import eval_flow as E  # noqa: E402

import model as M  # noqa: E402
import train_expert as T  # noqa: E402

LEVELS = [
    "grasp_easy",
    "catapult",
    "cartpole_thrust",
    "hard_lunar_lander",
    "mjc_half_cheetah",
    "mjc_swimmer",
    "mjc_walker",
    "h17_unicycle",
    "chain_lander",
    "catcher_v3",
    "trampoline",
    "car_launch",
]  # official order
level_paths = [f"{src}/../worlds/l/{name}.json" for name in LEVELS]
sp = kenv_state.StaticEnvParams(**T.LARGE_ENV_PARAMS, frame_skip=T.FRAME_SKIP)
ep = kenv_state.EnvParams()
levels = T.load_levels(level_paths, sp, ep)
sp = sp.replace(screen_dim=T.SCREEN_DIM)
env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=sp)
log_dirs = sorted(
    [p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)
)


def load(step):
    sds = [pickle.load((log_dirs[step] / "policies" / f"worlds_l_{n}.pkl").open("rb")) for n in LEVELS]
    return jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *sds))


sd, wsd = load(-1), load(0)
print("strong", log_dirs[-1], "weak", log_dirs[0], "devices", jax.devices(), flush=True)
obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), ep)[0].shape[-1]
action_dim = env.action_space(ep).shape[0]
mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
pspec = jax.sharding.PartitionSpec("x")
sh = jax.sharding.NamedSharding(mesh, pspec)


@functools.partial(jax.jit, static_argnums=(0,), in_shardings=sh, out_shardings=sh)
@functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec)
@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
def _eval(config, rng, level, state_dict, weak_state_dict):
    pol = M.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config.model, rngs=nnx.Rngs(rng))
    g, s = nnx.split(pol)
    s.replace_by_pure_dict(state_dict)
    pol = nnx.merge(g, s)
    g, s = nnx.split(pol)
    s.replace_by_pure_dict(weak_state_dict)
    weak = nnx.merge(g, s)
    info, _ = E.eval(config, env, rng, level, pol, ep, sp, weak)
    return info


METHODS = {
    "naive": E.NaiveMethodConfig(),
    "rtc": E.RealtimeMethodConfig(),
    "rtc_hard": E.RealtimeMethodConfig(prefix_attention_schedule="zeros"),
    "bid": E.BIDMethodConfig(n_samples=32, bid_k=3),
}
core = [(m, d, max(d, 1)) for m in ("naive", "rtc", "rtc_hard") for d in range(5)]
horizon = [(m, 1, s) for m in ("naive", "rtc", "rtc_hard") for s in range(2, 8)]
bid = [("bid", d, max(d, 1)) for d in range(5)]
bid_h = [("bid", 1, s) for s in range(2, 8)]
combos = {
    "core": core,
    "horizon": horizon,
    "bid": bid,
    "bid_h": bid_h,
    "all": core + horizon + bid,
}[group]
done = set()
if pathlib.Path(out).exists():
    done = {(r["method"], r["d"], r["s"]) for r in map(json.loads, open(out)) if r}
rngs = jax.random.split(jax.random.key(0), len(LEVELS))  # = eval_flow.main seed 0
for m, d, s in combos:
    if (m, d, s) in done:
        continue
    c = E.EvalConfig(num_evals=2048, inference_delay=d, execute_horizon=s, method=METHODS[m], weak_step=0)
    t = time.perf_counter()
    r = jax.device_get(_eval(c, rngs, levels, sd, wsd))
    dt = time.perf_counter() - t
    with open(out, "a") as f:
        for i, name in enumerate(LEVELS):
            row = dict(
                level=name,
                method=m,
                d=d,
                s=s,
                n=2048,
                metric=SOLVED,
                physics_precision=PRECISION or "default",
                solved=float(r["returned_episode_solved"][i]),
                ret=float(r["returned_episode_returns"][i]),
                length=float(r["returned_episode_lengths"][i]),
                seconds=dt,
            )
            f.write(json.dumps(row) + "\n")
    print(m, d, s, f"{dt:.0f}s", "avg", float(jnp.mean(r["returned_episode_solved"])), flush=True)
