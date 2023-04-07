import argparse
from bz2 import compress
import os
import tarfile
import pandas as pd
import zipfile
from tqdm.auto import tqdm

margin = 8
annotations_root = "train_val"
data_path = "/data/ggoletto/EpicKitchenDA/rgb_flow/"
output_path = "utils/output/"


def main():
    # Read all annotations
    annotations = pd.concat([
        pd.read_pickle(os.path.join(annotations_root, annotations))
        for annotations in os.listdir(annotations_root)
    ])

    os.makedirs(output_path, exist_ok=True)

    print(len(annotations.video_id.unique()))

    for video_id in annotations.video_id.unique():
        seen = []

        video_annotations = annotations[annotations.video_id == video_id]

        print(f"Processing {video_id} [length={len(video_annotations)}]...")

        try:
            zfile = os.path.join(output_path, f"{video_id}.tar.gz")

            if os.path.exists(zfile):
                continue
            
            with tarfile.open(zfile, mode='w:gz') as archive:
                for _, row in tqdm(video_annotations.iterrows()):
                    start_frame, stop_frame = row.start_frame, row.stop_frame

                    for frame in range(start_frame - margin, stop_frame + margin + 1):
                        if frame in seen:
                            continue

                        seen.append(frame)
                        fn = f"img_{frame:010d}.jpg"
                        if not os.path.exists(os.path.join(data_path, video_id, fn)):
                            print(f"Frame {fn} is missing.")
                            continue

                        archive.add(os.path.join(data_path, video_id, fn), arcname=fn)

        except zipfile.BadZipFile as error:
            print(error)


if __name__ == "__main__":
    main()
