import glob
import os

import wandb


def _parse_tags(raw_tags):
    if not raw_tags:
        return None
    tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    return tags or None


def init_wandb_run(args, default_project, default_mode="online"):
    wandb_mode = getattr(args, "wandb_mode", os.environ.get("WANDB_MODE", default_mode))
    project = getattr(args, "wandb_project", default_project) or default_project
    entity = getattr(args, "wandb_entity", "") or None
    group = getattr(args, "wandb_group", "") or None
    tags = _parse_tags(getattr(args, "wandb_tags", ""))
    wandb_dir = getattr(args, "wandb_dir", os.environ.get("WANDB_DIR", "./wandb_result")) or "./wandb_result"

    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported wandb mode: {wandb_mode}")

    os.makedirs(wandb_dir, exist_ok=True)

    if wandb_mode == "online":
        try:
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=False)
            else:
                wandb.login(relogin=False)
        except Exception as exc:
            print(f"[wandb] login failed, fallback to offline mode: {exc}")
            wandb_mode = "offline"

    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        config=vars(args),
        name=getattr(args, "run_name", None),
        mode=wandb_mode,
        tags=tags,
        dir=wandb_dir,
        save_code=True,
    )

    if run is not None:
        wandb.define_metric("t")
        for metric_name in [
            "eval_reward",
            "train_reward",
            "episode_steps",
            "episodes",
            "actor_loss",
            "critic_loss",
            "ppo_ratios",
            "reward_pred_err",
            "reward_model_loss",
            "alpha",
            "alpha_loss",
        ]:
            wandb.define_metric(metric_name, step_metric="t")

    return run


def finalize_wandb_run(summary=None, file_paths=None, file_globs=None):
    if wandb.run is None:
        return

    try:
        summary = summary or {}
        for key, value in summary.items():
            if value is not None:
                wandb.run.summary[key] = value

        uploaded = set()
        for path in file_paths or []:
            if path and os.path.exists(path):
                norm_path = os.path.normpath(path)
                if norm_path not in uploaded:
                    wandb.save(norm_path, policy="now")
                    uploaded.add(norm_path)

        for pattern in file_globs or []:
            for path in glob.glob(pattern):
                norm_path = os.path.normpath(path)
                if norm_path not in uploaded and os.path.isfile(norm_path):
                    wandb.save(norm_path, policy="now")
                    uploaded.add(norm_path)

        wandb.finish()
    except Exception as exc:
        print(f"[wandb] finalize skipped after failure: {exc}", flush=True)
