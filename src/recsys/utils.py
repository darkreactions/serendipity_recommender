import pandas as pd
from scipy.spatial.distance import cdist


def find_similar_recs(hist_df: pd.DataFrame, cand_df: pd.DataFrame,
                      recs: list) -> list:
    """TODO DOCUMENTATION"""

    similar_recs = []

    cols_to_exclude = [
        'name',
        '_rxn_organic-inchikey',
        '_raw_modelname',
        '_out_crystalscore'
    ]

    hist_data = hist_df.drop(cols_to_exclude, axis=1)

    hist_cols = hist_data.columns

    for rec in recs:
        reaction_df = cand_df[cand_df['name'] == rec]

        if not reaction_df:
            print("REACTION NOT FOUND, CHECK NAME'S FORMAT")

        react_data = reaction_df[hist_cols]

        euclid_dists = cdist(hist_data, react_data, 'euclid')

        min_dist, min_idx = min((dist, idx) for (idx, dist) in enumerate(
            euclid_dists))

        similar_rec_name = hist_df.iloc[min_idx]['name']

        similar_recs.append(similar_rec_name)

    return similar_recs
