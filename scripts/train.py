import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='data')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--out', type=str, default='runs/exp1')
    args = p.parse_args()
    print(vars(args))

if __name__ == '__main__':
    main()
