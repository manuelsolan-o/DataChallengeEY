import pandas as pd

import chardet
import csv

def create_meta_data(url):
    
    with open(url, "rb") as f:
        result = chardet.detect(f.read(100000)) # 100000 bytes
        encoding = result["encoding"]
    
    with open(url, "r", encoding=encoding) as f:
        sample = f.read(50000)  # Read first 50000 chars
        dialect = csv.Sniffer().sniff(sample)  # Detect delimiter
        sep = dialect.delimiter
    
    return [url, sep, encoding]

# ----------------------------------------------------------

def read_school(cycle, school_dict):
    url = school_dict[cycle][0]
    sep = school_dict[cycle][1]
    encoding = school_dict[cycle][2]
    return pd.read_csv(url, sep=sep, encoding=encoding, low_memory=False) # low_memory=False for big files

def read_school_cols(cycle, cols_dict):
    url = cols_dict[cycle][0]
    return pd.read_excel(url)

def curate_data_set(data_set, col_names):

    # Get only the first two columns are used and create a copy to avoid SettingWithCopyWarning
    col_names = col_names.iloc[:, 0:2].copy()  

    # Drop rows with missing values and reset index
    col_names = col_names.dropna().reset_index(drop=True)  

    # Create a dictionary to map old column names to new ones
    rename_dict = dict(zip(col_names['Field'], col_names['Rename']))

    # Identify which columns from col_names are actually present in the dataset (Sanity Check)
    available_cols = [col for col in col_names['Field'] if col in data_set.columns]

    # Filter the dataset to include only the available columns and create a copy
    data_set = data_set.loc[:, available_cols].copy()  

    # Update the rename dictionary to include only the present columns
    rename_dict = {k: v for k, v in rename_dict.items() if k in available_cols}

    # Rename the columns based on the dictionary
    data_set = data_set.rename(columns=rename_dict)

    return data_set


# My analysis should focus on the following academic cycles:

cycles = ["2019-2020","2020-2021","2021-2022","2022-2023"]

# Create Meta Data

# High School ----------------------------------------------------------------

high_school_url = {"2019-2020":["High School/Datasets/F9117G_1920.csv"],
                 "2020-2021":["High School/Datasets/F9117G_2021.csv"],
                 "2021-2022":["High School/Datasets/F9117G_2122.csv"],
                 "2022-2023":["High School/Datasets/F9117G_2223.csv"]}

high_school_meta = {}

for cycle in cycles:
    high_school_meta[cycle] = create_meta_data(high_school_url[cycle][0])
    
high_school_cols = {"2019-2020":["High School/Column names/F9117G_1920.xlsx"],
                   "2020-2021":["High School/Column names/F9117G_2021.xlsx"],
                   "2021-2022":["High School/Column names/F9117G_2122.xlsx"],
                   "2022-2023":["High School/Column names/F9117G_2223.xlsx"]}

under_grad_url = {"2019-2020":["Undergraduate School 1/Datasets/KI9119A_1920.csv"],
                "2020-2021":["Undergraduate School 1/Datasets/KI9119A_2021.csv"],
                "2021-2022":["Undergraduate School 2/Datasets/KI9119A_2122.csv"],
                "2022-2023":["Undergraduate School 2/Datasets/KI9119A_2223.csv"]
                  }

# Under Graduate School ----------------------------------------------------------------

under_grad_meta = {}

for cycle in cycles:
    under_grad_meta[cycle] = create_meta_data(under_grad_url[cycle][0])
    
under_grad_cols = {"2019-2020":["Undergraduate School 1/Column names/KI9119A_1920.xlsx"],
                 "2020-2021":["Undergraduate School 1/Column names/KI9119A_2021.xlsx"],
                 "2021-2022":["Undergraduate School 2/Column names/KI9119A_2122.xlsx"],
                 "2022-2023":["Undergraduate School 2/Column names/KI9119A_2223.xlsx"]}

# Graduate School ----------------------------------------------------------------

grad_url = {"2019-2020":["Graduate School/Datasets/KI9119B_1920.csv"],
              "2020-2021":["Graduate School/Datasets/KI9119B_2021.csv"],
              "2021-2022":["Graduate School/Datasets/KI9119B_2122.csv"],
              "2022-2023":["Graduate School/Datasets/KI9119B_2223.csv"]
             }

grad_meta = {}

for cycle in cycles:
    grad_meta[cycle] = create_meta_data(grad_url[cycle][0])
    
grad_cols = {"2019-2020":["Graduate School/Column names/KI9119B_1920.xlsx"],
                  "2020-2021":["Graduate School/Column names/KI9119B_2021.xlsx"],
                  "2021-2022":["Graduate School/Column names/KI9119B_2122.xlsx"],
                  "2022-2023":["Graduate School/Column names/KI9119B_2223.xlsx"]
                  }

# -------------------------------------------------------------------------------

cycle = "2019-2020"

raw_high_school_1920 = read_school(cycle, high_school_meta)
high_school_cols_1920 = read_school_cols(cycle, high_school_cols)
high_school_1920 = curate_data_set(raw_high_school_1920, high_school_cols_1920)

raw_under_grad_1920 = read_school(cycle, under_grad_meta)
under_grad_cols_1920 = read_school_cols(cycle, under_grad_cols)
under_grad_1920 = curate_data_set(raw_under_grad_1920, under_grad_cols_1920)

raw_grad_1920 = read_school(cycle, grad_meta)
grad_cols_1920 = read_school_cols(cycle, grad_cols)
grad_1920 = curate_data_set(raw_grad_1920, grad_cols_1920)

# -------------------------------------------------------------------------------

cycle = "2020-2021"

raw_high_school_2021 = read_school(cycle, high_school_meta)
high_school_cols_2021 = read_school_cols(cycle, high_school_cols)
high_school_2021 = curate_data_set(raw_high_school_2021, high_school_cols_2021)

raw_under_grad_2021 = read_school(cycle, under_grad_meta)
under_grad_cols_2021 = read_school_cols(cycle, under_grad_cols)
under_grad_2021 = curate_data_set(raw_under_grad_2021, under_grad_cols_2021)

raw_grad_2021 = read_school(cycle, grad_meta)
grad_cols_2021 = read_school_cols(cycle, grad_cols)
grad_2021 = curate_data_set(raw_grad_2021, grad_cols_2021)

# -------------------------------------------------------------------------------

cycle = "2021-2022"

raw_high_school_2122 = read_school(cycle, high_school_meta)
high_school_cols_2122 = read_school_cols(cycle, high_school_cols)
high_school_2122 = curate_data_set(raw_high_school_2122, high_school_cols_2122)

raw_under_grad_2122 = read_school(cycle, under_grad_meta)
under_grad_cols_2122 = read_school_cols(cycle, under_grad_cols)
under_grad_2122 = curate_data_set(raw_under_grad_2122, under_grad_cols_2122)

raw_grad_2122 = read_school(cycle, grad_meta)
grad_cols_2122 = read_school_cols(cycle, grad_cols)
grad_2122 = curate_data_set(raw_grad_2122, grad_cols_2122)

# -------------------------------------------------------------------------------

cycle = "2022-2023"

raw_high_school_2223 = read_school(cycle, high_school_meta)
high_school_cols_2223 = read_school_cols(cycle, high_school_cols)
high_school_2223 = curate_data_set(raw_high_school_2223, high_school_cols_2223)

raw_under_grad_2223 = read_school(cycle, under_grad_meta)
under_grad_cols_2223 = read_school_cols(cycle, under_grad_cols)
under_grad_2223 = curate_data_set(raw_under_grad_2223, under_grad_cols_2223)

raw_grad_2223 = read_school(cycle, grad_meta)
grad_cols_2223 = read_school_cols(cycle, grad_cols)
grad_2223 = curate_data_set(raw_grad_2223, grad_cols_2223)