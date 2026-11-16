"""Frozen L1 cohorts retain their original configs, registries and manifests."""

from dataclasses import asdict
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from .artifacts import file_hash, load_checkpoint
from .config import L1PoolConfig, canonical_hash, load_config
from .model import RecurrentActorCritic
from .provenance import source_metadata

CATALOG_SCHEMA = "re-nestrl-l1-catalog-v1"
ARTIFACT_FILES = ("config.json", "registry.json", "run.json", "complete.json",
                  "checkpoints/final.pt", "checkpoints/final.json", "checkpoints/latest.json")


def policy_contract(config):
    """Only environment, architecture and recurrent-input semantics affect inference."""
    return {"environment": asdict(config.environment), "hidden_dim": config.hidden_dim,
            "hidden_dims": list(config.hidden_dims), "episodes_per_round": config.episodes_per_round,
            "context_reset": config.context_reset, "previous_action_input": config.previous_action_input,
            "episode_boundary_input": config.episode_boundary_input,
            "finite_horizon_returns": config.finite_horizon_returns,
            "l1_ego_side": config.l1_ego_side, "ego_side": config.ego_side}


def relative_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError("Unsafe partner artifact path")
    result = Path(root) / Path(*path.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Partner artifact escapes run root")
    return result


def read_catalog(path):
    catalog = json.loads(Path(path).read_text())
    if catalog.get("schema") != CATALOG_SCHEMA or not catalog.get("models"):
        raise ValueError("Unsupported L1 catalog")
    identities = [m["id"] for m in catalog["models"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate L1 catalog identity")
    for row in catalog["models"]:
        if (type(row["seed"]) is not int or not re.fullmatch(r"l1-construction-[0-9]+", row["id"])
                or row["id"] != f"l1-construction-{row['seed']}" or row["stage"] != "l1"
                or row["split"] != "construction" or row["steps"] != 28_800_000 or row["updates"] != 400):
            raise ValueError("Catalog requires completed construction L1 policies")
        if set(row["files"]) != set(ARTIFACT_FILES) or any(not re.fullmatch(r"[a-f0-9]{64}", x) for x in row["files"].values()):
            raise ValueError("Catalog requires all original artifact SHA-256 values")
        qualification = row["qualification"]
        if set(qualification["successes"]) != {"LettuceOnionSalad", "TomatoCarrotSalad", "PotatoBroccoliSalad"}:
            raise ValueError("Catalog qualification must include all recipes")
        if qualification["episodes_per_recipe"] != 1000 or any(type(n) is not int or not 950 <= n <= 1000 for n in qualification["successes"].values()):
            raise ValueError("Catalog L1 failed the declared qualification threshold")
    return catalog


def select_partner_ids(identities, count, seed):
    """One independent uniform sample without replacement, with stable ordering."""
    import numpy as np
    if (len(set(identities)) != len(identities) or not identities
            or any(not re.fullmatch(r"l1-construction-[0-9]+", x) for x in identities)):
        raise ValueError("Selection candidates must be unique construction identities")
    if type(count) is not int or not 1 <= count <= len(identities) or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid selection count/seed")
    ordered = sorted(identities, key=lambda x: int(x.rsplit('-', 1)[1]))
    draw = np.random.Generator(np.random.PCG64(seed)).choice(len(ordered), size=count, replace=False)
    return tuple(ordered[i] for i in sorted(draw))


def load_catalog_pool(config, output_root, device, *, pool=None):
    """Validate each source configuration separately; never relabel a checkpoint."""
    from .pipeline import registry_spec
    spec = pool or config.l1_pool
    if not isinstance(spec, L1PoolConfig):
        raise ValueError("An explicit L1 catalog pool is required")
    spec.validate()
    catalog_path = relative_path(output_root, spec.catalog)
    if file_hash(catalog_path) != spec.catalog_sha256:
        raise ValueError("L1 catalog SHA-256 differs from the experiment contract")
    catalog = read_catalog(catalog_path)
    rows = {row["id"]: row for row in catalog["models"]}
    if not set(spec.partner_ids).issubset(rows):
        raise ValueError("Selected L1 is missing from the catalog")
    if not {rows[x]["seed"] for x in spec.partner_ids}.issubset(config.construction_seeds):
        raise ValueError("Only construction partners may be selected")
    current_source = source_metadata()
    models, provenance = [], []
    for identity in spec.partner_ids:
        row = rows[identity]
        if row["ego_side"] != config.l1_ego_side:
            raise ValueError("Catalog L1 role differs from L2 partner side")
        folder = relative_path(output_root, f"partners/{identity}")
        for name, digest in row["files"].items():
            if file_hash(relative_path(folder, name)) != digest:
                raise ValueError(f"Original partner artifact SHA-256 mismatch: {identity}/{name}")
        original = load_config(folder / "config.json")
        if original.sha256 != row["config_sha256"] or row["seed"] not in original.construction_seeds:
            raise ValueError("Original L1 configuration or partition differs")
        if canonical_hash(policy_contract(original)) != canonical_hash(policy_contract(config)):
            raise ValueError("L1 policy/environment/history contract differs from L2")
        registry = json.loads((folder / "registry.json").read_text())
        if registry != registry_spec(original):
            raise ValueError("Original L1 registry differs from its own configuration")
        state, manifest = load_checkpoint(folder / "checkpoints/final.pt", device, original.sha256)
        expected = {"stage": "l1", "split": "construction", "seed": row["seed"],
                    "ego_side": config.l1_ego_side, "run_id": identity,
                    "registry_sha256": canonical_hash(registry), "steps": row["steps"]}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("L1 manifest role, identity, budget or original registry differs")
        if (manifest["environment_source_sha256"] != current_source["environment_source_sha256"]
                or manifest["implementation_sources"].get("src/nestrl_research/model.py")
                != current_source["implementation_sources"]["src/nestrl_research/model.py"]):
            raise ValueError("L1 environment or policy implementation differs")
        run = json.loads((folder / "run.json").read_text())
        for key in ("stage", "split", "seed", "ego_side", "run_id", "config_sha256", "registry_sha256",
                    "training_partners", "environment_source_sha256", "implementation_source_sha256"):
            if run.get(key) != manifest.get(key):
                raise ValueError(f"Original L1 run/manifest disagree: {key}")
        complete = json.loads((folder / "complete.json").read_text())
        if complete["sha256"] != manifest["sha256"] or complete["steps"] != row["steps"]:
            raise ValueError("L1 completion marker differs")
        if original.total_steps != row["steps"] or original.total_steps // original.batch_size != row["updates"]:
            raise ValueError("Original L1 budget differs")
        model = RecurrentActorCritic(obs_dim=105, action_dim=6, hidden_dim=original.hidden_dim,
                                    hidden_dims=tuple(original.hidden_dims)).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        models.append(model)
        provenance.append({"id": identity, "split": "construction", "seed": row["seed"],
                           "checkpoint": f"partners/{identity}/checkpoints/final.pt",
                           "checkpoint_sha256": manifest["sha256"],
                           "manifest_sha256": row["files"]["checkpoints/final.json"],
                           "source_config_sha256": original.sha256,
                           "source_registry_sha256": canonical_hash(registry),
                           "catalog_sha256": spec.catalog_sha256, "frozen": True})
    return models, provenance


def stage_catalog(config, catalog_path, archives_dir, output_root):
    """Copy original artifact bytes, validating all destinations before writing."""
    if config.l1_pool is None or file_hash(Path(catalog_path)) != config.l1_pool.catalog_sha256:
        raise ValueError("Preparation catalog differs from config")
    catalog = read_catalog(catalog_path)
    from .pipeline import registry_spec
    existing_registry = Path(output_root) / "registry.json"
    if existing_registry.exists() and json.loads(existing_registry.read_text()) != registry_spec(config):
        raise ValueError("Existing target registry belongs to another experiment")
    rows = {row["id"]: row for row in catalog["models"]}
    if not set(config.l1_pool.partner_ids).issubset(rows):
        raise ValueError("Selected L1 is missing from catalog")
    planned = {config.l1_pool.catalog: Path(catalog_path).read_bytes()}
    # Stage all qualified candidates once so a common construction diagnostic
    # can inspect trained and untrained partners without changing their splits.
    by_archive = {}
    for row in catalog["models"]:
        by_archive.setdefault(row["archive"]["filename"], []).append(row)
    for filename, members in by_archive.items():
        archive = relative_path(archives_dir, filename)
        for row in members:
            if file_hash(archive) != row["archive"]["sha256"]:
                raise ValueError("L1 archive SHA-256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            seen = set()
            for item in bundle.infolist():
                if item.filename in seen or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Duplicate or symbolic-link archive member")
                relative_path(output_root, item.filename)
                seen.add(item.filename)
            for row in members:
                for name in ARTIFACT_FILES:
                    member = row["archive"]["members"][name]
                    payload = bundle.read(member)
                    import hashlib
                    if hashlib.sha256(payload).hexdigest() != row["files"][name]:
                        raise ValueError("Original L1 archive member SHA-256 mismatch")
                    planned[f"partners/{row['id']}/{name}"] = payload
    for name, contents in planned.items():
        path = relative_path(output_root, name)
        if path.exists() and path.read_bytes() != contents:
            raise ValueError(f"Existing partner artifact differs; will not overwrite: {name}")
    for name, contents in planned.items():
        path = relative_path(output_root, name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(contents)
    models, provenance = load_catalog_pool(config, output_root, "cpu")
    from .pipeline import ensure_registry
    ensure_registry(config, Path(output_root))
    return models, provenance
