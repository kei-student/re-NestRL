#!/usr/bin/env python3
"""Draw an L1 partner set separately for one L2 and freeze its run configuration."""
import argparse
import json
from pathlib import Path
import secrets

from nestrl_research.artifacts import file_hash, safe_name
from nestrl_research.config import ROOT, load_config
from nestrl_research.partner_pool import read_catalog, relative_path, select_partner_ids


def create(study_path, condition, learner_seed, output_root, *, selection_seed=None):
    study_path, output_root = Path(study_path), Path(output_root)
    study = json.loads(study_path.read_text())
    if study.get('schema') != 're-nestrl-l2-per-agent-pools-v1':
        raise ValueError('Use the per-agent partner-selection study')
    if type(learner_seed) is not int or not 0 <= learner_seed < 2**32:
        raise ValueError('Learner seed must be an integer in [0,2**32)')
    if output_root.exists():
        raise FileExistsError('L2 instance already exists; reuse its saved config without drawing again')
    identity = safe_name(output_root.name)
    conditions = {c['id']: c for c in study['conditions']}
    if condition not in conditions:
        raise ValueError('Unknown L2 partner-count condition')
    spec = conditions[condition]
    base_path = relative_path(ROOT,spec['base_config'])
    catalog_path = relative_path(ROOT,study['catalog'])
    base = load_config(base_path)
    if base.sha256 != spec['base_config_sha256'] or file_hash(catalog_path) != study['catalog_sha256']:
        raise ValueError('Base config or catalog changed')
    catalog = read_catalog(catalog_path)
    candidates = [m['id'] for m in catalog['models']]
    if sorted(m['seed'] for m in catalog['models']) != study['candidate_seeds']:
        raise ValueError('Candidate cohort differs from study')
    count = spec['partner_count']
    if count != int(condition[1:]) or count not in (3,5,15):
        raise ValueError('Partner count differs from the requested condition')
    if count == len(candidates):
        if selection_seed is not None:
            raise ValueError('All-partner condition does not require a selection seed')
        chosen = tuple(candidates)
        seed = None
        source = 'all qualified candidates'
    else:
        # Do not derive this from learner_seed or reuse a condition-level seed.
        # A supplied seed is only for explicit reproduction of a recorded draw.
        seed = secrets.randbits(32) if selection_seed is None else selection_seed
        chosen = select_partner_ids(candidates, count, seed)
        source = 'fresh OS entropy' if selection_seed is None else 'explicit replay seed'
    value = base.to_dict()
    value['profile'] = f'l2-env3-{condition}-{identity}'
    value['learner_seed'] = learner_seed
    value['l1_pool'] = {'catalog':'partners/catalog.json','catalog_sha256':study['catalog_sha256'],
                        'partner_ids':list(chosen)}
    # Validate the complete config before creating an instance directory.
    from nestrl_research.config import EnvironmentConfig, ExperimentConfig, L1PoolConfig, PPOConfig
    resolved = ExperimentConfig(**{**value, 'environment':EnvironmentConfig(**value['environment']),
                                  'ppo':PPOConfig(**value['ppo']), 'l1_pool':L1PoolConfig(**value['l1_pool'])})
    resolved.validate()
    if resolved.total_steps // resolved.batch_size != 400 or resolved.total_steps != 28_800_000:
        raise ValueError('Per-agent selection must preserve the 400-update budget')
    record = {'schema':'re-nestrl-l2-selection-v1','instance_id':identity,'condition':condition,
              'learner_seed':learner_seed,'partner_count':count,'selection_seed':seed,
              'selection_seed_source':source,'selection_rng':'numpy.PCG64' if seed is not None else None,
              'partner_seeds':[int(x.rsplit('-',1)[1]) for x in chosen],
              'partner_ids':list(chosen),'catalog_sha256':study['catalog_sha256'],
              'study_file_sha256':file_hash(study_path),'base_config_sha256':base.sha256,
              'config_file':'config.json','config_sha256':resolved.sha256,
              'membership_lifetime':'fixed for this L2; reuse on resume','training_started':False}
    output_root.mkdir(parents=True,exist_ok=False)
    with (output_root/'config.json').open('x') as f:
        f.write(json.dumps(value,indent=2)+'\n')
    record['config_file_sha256'] = file_hash(output_root/'config.json')
    with (output_root/'selection.json').open('x') as f:
        f.write(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'configs/l2-env3-per-agent-20261116.json')
    parser.add_argument('--condition',choices=('k03','k05','k15'),required=True)
    parser.add_argument('--learner-seed',type=int,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--selection-seed',type=int,
                        help='Reproduce a recorded draw; omit for a fresh independent draw per L2')
    args = parser.parse_args()
    print(json.dumps(create(args.study,args.condition,args.learner_seed,args.output_root,
                            selection_seed=args.selection_seed),indent=2))


if __name__ == '__main__':
    main()
