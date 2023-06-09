from .solver import CaptchaSolver
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Captcha decoder")

    parser.add_argument("-url", dest="url", required=True)
    parser.add_argument("-model", dest="model_fname", required=True)
    parser.add_argument("-height", dest="img_height", default=40, type=int)
    parser.add_argument("-width", dest="img_width", default=300, type=int)
    parser.add_argument("-max-length", dest="max_length", default=10, type=int)

    args = parser.parse_args()
    return {
        "url": args.url,
        "minimum_accuracy": args.minimum_accuracy,
        "repeat_count": args.repeat_count,
    }


if __name__ == '__main__':
    ans = CaptchaSolver().solve(**parse_args())
    print(" ".join(map(str, ans)))
