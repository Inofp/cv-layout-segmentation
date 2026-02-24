import argparse
from src.cvlayout.train import TrainConfig, train

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=str, default="runs/exp1")
    p.add_argument("--base", type=int, default=32)
    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        base=args.base,
    )
    hist = train(cfg)
    print(hist["epochs"][-1] if hist["epochs"] else {})

if __name__ == "__main__":
    main()