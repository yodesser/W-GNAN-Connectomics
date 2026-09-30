import argparse
import torch
from models import wgnan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["nodeonly", "topology", "linear", "learned"])
    parser.add_argument("--repeats", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Running real-data CV for variant: {args.variant} on {args.device} (repeats {args.repeats})")
    wgnan.run_cv(args.variant, args.repeats, device=args.device)

if __name__ == "__main__":
    main()
