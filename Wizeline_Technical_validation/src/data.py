import yaml
from ucimlrepo import fetch_ucirepo
import sys

def get_data(params):

    regensburg_pediatric_appendicitis = fetch_ucirepo(id=params["base"]["repo_id"])

    # data (as pandas dataframes)
    Features = regensburg_pediatric_appendicitis.data.features
    Targets = regensburg_pediatric_appendicitis.data.targets

    Features.to_csv(params["data_load"]["features_data"], index=False)
    Targets.to_csv(params["data_load"]["targets_data"], index=False)

if __name__ == '__main__':

    data_path = sys.argv[1]

    with open(data_path, "r") as f:
        params = yaml.safe_load(f)

    get_data(params)