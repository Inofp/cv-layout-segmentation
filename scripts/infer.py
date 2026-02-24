import argparse
from src.cvlayout.infer import infer_image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--min_area", type=int, default=900)
    args = p.parse_args()

    res = infer_image(args.model, args.image, args.out, min_area=args.min_area)
    print(res)

if __name__ == "__main__":
    main()