import ast
import os
import sys
import warnings

import pandas as pd
from pandas.api.types import CategoricalDtype

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import json


def load(filepath):
    # From https://github.com/mdeff/fma/blob/rc1/utils.py / MIT License

    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            CategoricalDtype(categories=SUBSETS, ordered=True)
        )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


def get_id_from_path(path):
    base_name = os.path.basename(path)

    return base_name.replace(".mp3", "").replace(".npy", "")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path")
    args = parser.parse_args()

    base_path = Path(args.metadata_path)

    in_path = base_path / "tracks.csv"
    genres_path = base_path / "genres.csv"

    out_path = base_path / "tracks_genre.json"
    mapping_path = base_path / "mapping.json"

    df = load(in_path)

    df2 = pd.read_csv(genres_path)

    id_to_title = {k: v for k, v in zip(df2.genre_id.tolist(), df2.title.tolist())}

    df.reset_index(inplace=True)

    print(df.head())
    print(df.columns.values)
    print(set(df[("set", "subset")].tolist()))

    df = df[df[("set", "subset")].isin(["small"])]

    print(set(df[("track", "genre_top")].tolist()))

    print(
        df[
            [
                ("track_id", ""),
                ("track", "genre_top"),
                ("track", "genres"),
                ("set", "subset"),
            ]
        ]
    )

    data = {
        k: v
        for k, v in zip(
            df[("track_id", "")].tolist(), df[("track", "genre_top")].tolist()
        )
    }

    json.dump(data, open(out_path, "w"), indent=4)

    mapping = {k: i for i, k in enumerate(set(df[("track", "genre_top")].tolist()))}

    json.dump(mapping, open(mapping_path, "w"), indent=4)
