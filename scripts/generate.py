import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='data')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--w', type=int, default=768)
    p.add_argument('--h', type=int, default=1086)
    p.add_argument('--classes', type=str, default='header,experience,education,skills,contacts')
    args = p.parse_args()
    print(vars(args))

if __name__ == '__main__':
    main()
