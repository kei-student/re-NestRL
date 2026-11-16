#!/usr/bin/env python3
"""Print a reproducible independent construction-pool draw; never start training."""
import argparse
import json
from pathlib import Path

from nestrl_research.artifacts import file_hash
from nestrl_research.partner_pool import read_catalog, select_partner_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--selection-seed', type=int, required=True)
    args = parser.parse_args()
    catalog = read_catalog(args.catalog)
    selected = select_partner_ids([m['id'] for m in catalog['models']], args.count, args.selection_seed)
    print(json.dumps({'catalog_sha256':file_hash(args.catalog), 'rng':'numpy.PCG64',
                      'selection_seed':args.selection_seed, 'partner_ids':selected},indent=2))


if __name__ == '__main__':
    main()
