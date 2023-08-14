import argparse
import util
import mesh_creation
def main():
    parser = argparse.ArgumentParser(description="Choose a flag")
    parser.add_argument("-gen_bbox", type=str, help="Generate individual instances from Picture (path to image)")
    parser.add_argument("-gen_obj", type=str, help="Generate obj file from object")

    args = parser.parse_args()

    if args.gen_bbox is not None:
        util.extract_roi(args.gen_bbox)
    elif args.gen_obj is not None:
        mesh_creation.create_3D(args.gen_obj)
    else:
        print("No function specified.")

if __name__ == "__main__":
    main()
