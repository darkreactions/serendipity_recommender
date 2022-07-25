from multiprocessing import Pool, cpu_count
import os
import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Catalog:
    """Recommendation catalog with historical data and candidates

    PROCEDURE:

    (__init__)
    0. Initialization with the amine whose reactions are used for
        recommendations.

    (load_grid)
    1. Load in a pre-cleaned, formatted full grid.
    2. Partition the full grid into historical and candidate reactions.
    3. For both partitions: non-numerical columns are dropped. Each partition is
        split into reaction id (name column) and numerical values. The
        numerical values are then normalized using the same MinMaxScaler,
        which is then stored into a dictionary with the corresponding reaction
        id.
    4. For candidate reactions, the predicted success probability is stored
        separately.

    (get_combinations)
    5. Get all pair-wise combinations of candidates X historical and
        candidates X candidates without duplicates.

    (find_all_sry)
    6. If the combinations are not generated, get combinations.
    7. With the generated combinations set up, a multiprocessing pool and a
        shared dictionary is set up to calculate the serendipity value in
        parallel. Each time, the pool takes in one objective function,
        which can be built and toggled in objectives.py.

    (store_all_sry)
    8. After one type of serendipity finishes calculation, the result will be
        store as a dictionary or updated with the existing dictionary.

    (save_catalog)
    9. If desired, one can call save_catalog function to store all the data
        for ad-hoc analysis. The catalog will be stored in an amine name
        specific folder under /data.

    (load_catalog)
    10. If at any point the catalog is saved under the specified folder path,
        one can retrieve it by building a new catalog object first, then call
        load_catalog to read in the previously store data.

    #TODO add scalers to list
    Attributes:
        amine (str):            The key string of the amine for recommendation.
        sry_types (list):       List of all calculated serendipity measurement
                                    forms.
        historical (dict):      Standardized historical reaction data.
                                    * Format: {reaction name --> reaction data}
        hist_names (list):      List of all historical reaction names.
        candidates (dict):      Standardized candidate reaction data.
                                    * Format: {reaction name --> reaction data}
        cand_names (list):      List of all candidate reaction names.
        succ_probs (dict):      Predicted success probability of the candidate
                                    reactions.
                                    * Format: {reaction name --> probability}
        combinations (list):    Reaction combinations used for pairwise
                                    serendipity calculation.
                                    * Format: [[pair name, reaction 1 data,
                                    reaction 2 data]]
        all_sry (dict):         The serendipity measurements of all reaction
                                    combinations.
                                    * Format: {combo name: [serendipity values]}

    Examples:
    Create a new catalog
    >>> catalog = Catalog('JMXLWMIFDJCGBV-UHFFFAOYSA-N')

    Load in the grid and parse the data for the specified amine
    >>> catalog.load_grid()

    Find the serendipity (e.g. cosine similarity) of all pairwise combinations
    >>> catalog.find_all_sry(obj=cosine_similarity)

    Save the constructed catalog
    >>> catalog.save_catalog()

    If there is a previously saved catalog, load by doing
    >>> catalog = Catalog('JMXLWMIFDJCGBV-UHFFFAOYSA-N')
    >>> catalog.load_catalog()

    """

    def __init__(self, amine: str = '') -> None:
        """Initialization of the recommendation catalog

        Args:
            amine (str):    The key string of the amine for recommendation.

        """
        self.amine = amine
        self.all_sry = None
        self.sry_types = []
        self.combinations = []

    def load_grid(self, grid_path: str = None) -> None:
        """Load and parse the reaction grid from /data/grid.csv"""

        # Read in the grid as DataFrame
        if not grid_path:
            grid_path = os.getcwd() + f'/data/grid_{self.amine}.csv'

        full_grid = pd.read_csv(grid_path)

        # Identify descriptive columns and drop them before processing
        descr_cols = ['_raw_modelname', '_out_crystalscore']
        full_grid.drop(columns=descr_cols, inplace=True)

        # Parse the grid in to historical and candidate reactions
        hist = full_grid[full_grid['status'] == 'hist']
        if self.amine != '':
            cand = full_grid[
                (full_grid['status'] == 'cand') &
                (full_grid['_rxn_organic-inchikey'] == self.amine)
                ]
        else:
            cand = full_grid[full_grid['status'] == 'cand']

        # Clear up memory
        del full_grid

        # PROCESS HISTORICAL DATA
        # Drop the amine name, status, and success probability column
        hist.drop(
            columns=['_rxn_organic-inchikey', 'status', 'succ_prob'],
            inplace=True
        )

        # Convert the DataFrame to a dictionary
        hist_data_raw = hist.set_index('name').to_dict('split')
        del hist
        del hist_data_raw['columns']

        # Unpack reaction names and reaction data
        hist_names, hist_data = hist_data_raw['index'], hist_data_raw['data']

        # TODO fix the dimension problem, this part is dumb
        scaler = MinMaxScaler()
        scaler.fit(hist_data)
        hist_data = scaler.transform(hist_data)
        hist_data = scaler.inverse_transform(hist_data)

        # Store the reactions in a dictionary attribute of the catalog
        self.historical = {
            hist_names[i]: hist_data[i] for i in range(len(hist_names))
        }

        # PROCESS CANDIDATES
        # Keep the success probability predicted by PLATIPUS
        dict_of_cands = cand.to_dict('index')
        self.succ_probs = {
            dict_of_cands[i]['name']: dict_of_cands[i]['succ_prob']
            for i in dict_of_cands.keys()
        }

        # Drop the amine name, status, and success probability column
        cand.drop(
            columns=['_rxn_organic-inchikey', 'status', 'succ_prob'],
            inplace=True
        )

        # Convert the DataFrame to a dictionary
        cand_data_raw = cand.set_index('name').to_dict('split')
        del cand
        del cand_data_raw['columns']

        # Unpack reaction names and reaction data
        cand_names, cand_data = cand_data_raw['index'], cand_data_raw['data']

        # TODO fix the dimension problem, this part is dumb
        scaler = MinMaxScaler()
        scaler.fit(cand_data)
        cand_data = scaler.transform(cand_data)
        cand_data = scaler.inverse_transform(cand_data)

        # Store the reactions in a dictionary attribute of the catalog
        self.candidates = {
            cand_names[i]: cand_data[i] for i in range(len(cand_names))
        }

    def get_combinations(self) -> None:
        """Generate pair-wise combinations of reactions and split into
            combinations.
        """

        # Extract historical and candidate reaction names
        self.hist_names = list(self.historical.keys())
        self.cand_names = list(self.candidates.keys())
        #self.cand_names = [str(i) for i in self.cand_names]

        # Cross product all candidates and all historical data
        all_combinations = [
            ('_x_'.join([i, j]), self.candidates[i], self.historical[j])
            for i in self.cand_names
            for j in self.hist_names
        ]

        # Cross product all candidates with themselves w/o duplicates
        all_combinations += [
                ('_x_'.join([i, j]), self.candidates[i], self.candidates[j])
                for idx, i in enumerate(self.cand_names)
                for j in self.cand_names[idx + 1:]
        ]

        # Find the max batch size of each batch
        self.combinations = all_combinations

    def find_all_sry(self, obj) -> None:
        """Calculate serendipity measurements in parallel.

        Attributes:
            obj (function):     The objective function to calculate serendipity.
        """

        # Get all pair-wise combinations
        if not self.combinations:
            self.get_combinations()

        # Multiprocess the serendipity calculation process
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(obj, self.combinations)

        # Separate serendipity w.r.t historical and candidate reactions
        pair_names = [result[0] for result in results]
        sry_vals = [[result[1]] for result in results]

        num_hist_sry = len(self.hist_names) * len(self.cand_names)
        hist_srys, cand_srys = sry_vals[:num_hist_sry], sry_vals[num_hist_sry:]

        # Scale each portion using a MinMaxScaler
        self.hist_cand_scaler = MinMaxScaler()
        self.hist_cand_scaler.fit(hist_srys)
        sry_vals[:num_hist_sry] = [
            i[0] for i in self.hist_cand_scaler.transform(hist_srys)
        ]

        self.cand_cand_scaler = MinMaxScaler()
        self.cand_cand_scaler.fit(cand_srys)
        sry_vals[num_hist_sry:] = [
            i[0] for i in self.cand_cand_scaler.transform(cand_srys)
        ]

        all_sry = {pair_names[i]: sry_vals[i] for i in range(len(results))}

        # Store the calculated serendipity to current catalog
        self.store_all_sry(all_sry, obj.__name__)

    def store_all_sry(self, all_sry: dict, sry_type: str) -> None:
        """Store the calculated serendipity to current catalog

        Attributes:
            all_sry (dict):     Calculated serendipity measurements.
                                    * Format: {combination name: serendipity}
            sry_type (str):    The type of serendipity calculated.
        """

        if not self.all_sry:
            # Store the calculated serendipity directly
            self.all_sry = {
                name: {sry_type: sry} for name, sry in all_sry.items()
            }

        else:
            # Update to the existing dictionary
            for name, sry in all_sry.items():
                self.all_sry[name][sry_type] = sry

        # Log serendipity measurement name for lookup later
        self.sry_types.append(sry_type)

    def save_catalog(self) -> None:
        """Save the catalog locally through pickle."""
        folder_path = os.getcwd() + f'/data/{self.amine}'

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        catalog_path = folder_path + f'/{self.amine}.pkl'

        with open(catalog_path, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=-1)

    def load_catalog(self) -> bool:
        """Load existing catalog from local directory."""
        catalog_path = os.getcwd() + f'/data/{self.amine}/{self.amine}.pkl'

        if os.path.exists(catalog_path):
            print("FOUND EXISTING CATALOG")

            with open(catalog_path, 'rb') as f:
                attr_dict = pickle.load(f)

            self.__dict__.update(attr_dict)

            return True

        else:
            print('THE SPECIFIED CATALOG DOES NOT EXIST')
            return False
