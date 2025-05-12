import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""
    return pd.read_csv(path)


def clean_world_gdp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for 'World' observation and Real GDP per Capita indicator.
    Returns DataFrame with ['Year', 'GDP_per_capita'] sorted by Year.
    """
    world = df[
        (df['Observation'] == 'World') &
        (df['Unit'] == 'Real GDP Per Capita in USD, Base Year = 2017')
    ].copy()
    world = world[['Year', 'Value']].rename(columns={'Value': 'GDP_per_capita'})
    world = world.dropna().sort_values('Year').reset_index(drop=True)
    return world


def save_processed_data(df: pd.DataFrame, path: str):
    """Save processed DataFrame to CSV."""
    df.to_csv(path, index=False)
