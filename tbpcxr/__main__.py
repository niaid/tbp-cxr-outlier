import argparse
import pandas as pd
from tbpcxr.model import Model
from tbpcxr.utilities import read_dcm
from pathlib import Path
import SimpleITK as sitk

import logging

logger = logging.getLogger(__name__)


def read_image(fname: Path) -> sitk.Image:
    reader = sitk.ImageFileReader()
    imageio = reader.GetImageIOFromFileName(fname)

    logger.debug(f"Reading {fname}...")

    if imageio == "GDCMImageIO":
        return read_dcm(fname)
    else:
        reader.SetFileName(fname)
        reader.SetOutputPixelType(sitk.sitkFloat32)
        image = reader.Execute()

        return image


def main():
    parser = argparse.ArgumentParser(prog="tbpcxr", description="Run classifer on CXR images to detect outliers.")
    parser.add_argument("--model", "-m", type=str, default=None, help="The name of the model to use.")
    parser.add_argument("-o", "--output", type=str, help="Output CSV file to save results.")

    parser.add_argument(
        "--decision", default=False, action=argparse.BooleanOptionalAction, help="Output decision values."
    )
    parser.add_argument("--verbose", "-v", default=0, action="count", help="Increase verbosity.")
    parser.parse_args(["--no-decision"])

    parser.add_argument("input_files", type=Path, nargs="+", help="List of input image files.")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.model is None:
        outlier_model = Model.load_outlier_pcamodel()
    else:
        outlier_model = Model.load_model(args.model)

    arr = outlier_model.to_observations(map(read_image, args.input_files))

    results = outlier_model.outlier_predictor(arr, decision=args.decision)

    if not args.decision:
        results = [result == -1 for result in results]

    if args.output:
        df = pd.DataFrame({"file": args.input_files, "outlier": results})
        logger.debug(f"Writing {args.output}...")
        df.to_csv(args.output, index=False)
    else:
        for fn, result in zip(args.input_files, results):
            print(f"{fn}: {'Outlier' if result else 'Normal'}")


if __name__ == "__main__":
    main()
