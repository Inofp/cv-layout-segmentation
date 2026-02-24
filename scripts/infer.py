import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--out', type=str, default='out')
    args = p.parse_args()
    print(vars(args))

if __name__ == '__main__':
    main()
