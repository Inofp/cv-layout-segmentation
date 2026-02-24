import argparse
from src.cvlayout.synth import SynthConfig, generate_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--w", type=int, default=768)
    p.add_argument("--h", type=int, default=1086)
    p.add_argument("--classes", type=str, default="header,experience,education,skills,contacts")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    labels = [s.strip() for s in args.classes.split(",") if s.strip()]
    cfg = SynthConfig(w=args.w, h=args.h)
    spec = generate_dataset(args.out, args.n, labels, cfg, val_ratio=args.val_ratio, seed=args.seed)
    print(spec)

if __name__ == "__main__":
    main()