import config
import glob


class DefaultDatasetLoader:
    vehicles = []
    non_vehicles = []
    def __init__(self, debug = False):
        self.vehicles.extend(glob.glob(config.Dataset.DEFAULT_VEHICLES, recursive=True))
        self.non_vehicles.extend(glob.glob(config.Dataset.DEFAULT_NON_VEHICLES, recursive=True))
        if debug:
            print("Number of vehicle images in DefaultDataset:", len(self.vehicles))
            print("Number of non-vehicle images in DefaultDataset:", len(self.non_vehicles))




