from __future__ import annotations
import argparse
from pathlib import Path
from mutsig.pipeline import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='TCGA-LUAD')
    ap.add_argument('--n-cases', type=int, default=50)
    ap.add_argument('--k-min', type=int, default=2)
    ap.add_argument('--k-max', type=int, default=7)
    args = ap.parse_args()
    out_root = Path(__file__).resolve().parent.parent
    summary = run(project_id=args.project, n_cases=args.n_cases, k_values=list(range(args.k_min, args.k_max + 1)), out_root=out_root)
    print('\n=== Summary ===')
    for k, entry in summary.items():
        if k == 'loss_curve_fig':
            continue
        print(f"K={k}: loss={entry['loss']:.2f}, figure={entry['fig']}")
        print(entry['assignment'].to_string(index=False))
        print()
if __name__ == '__main__':
    main()
