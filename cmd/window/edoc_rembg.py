import argparse
import os
import cv2
from PIL import Image


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


def remove_background_signature(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = Image.fromarray(thresh).convert("L")
    img = Image.fromarray(image).convert("RGB")
    cutout = naive_cutout(img, mask)
    cutout.save(output_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        type=str
    )

    ap.add_argument(
        "-o",
        "--output",
        type=str
    )
    args = ap.parse_args()
    remove_background_signature(args.input, args.output)


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()
