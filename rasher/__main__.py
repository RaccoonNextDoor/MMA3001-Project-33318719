"""Command-line entry point for dataset preparation and CPU baselines."""
import argparse
import json
from .data import prepare
from .baseline import run_baselines


def main():
    """Parse commands and report completion; invalid inputs raise clear errors."""
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--zip',required=True);a.add_argument('--out',default='data/prepared')
    a=sub.add_parser('baseline');a.add_argument('--data',default='data/prepared');a.add_argument('--out',default='results/baseline')
    args=p.parse_args()
    result=prepare(args.zip,args.out) if args.command=='prepare' else run_baselines(args.data,args.out)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
