import requests
import pandas as pd
import zipfile
import io

class JKPFactors(object):
    def __init__(self,
                 region=None,
                 theme=None,
                 frequency=None,
                 weighting=None,
                 ):

        self._post_jkp(region, theme, frequency, weighting)

    def get_url(self, region, theme_factor, frequency, weighting):
        base_url = 'https://jkpfactors.com/data/'

        if region is not None:
            base_url = base_url + '[{}]_'.format(region)
        if theme_factor is not None:
            base_url = base_url + '[{}]_'.format(theme_factor)
        if frequency is not None:
            base_url = base_url + '[{}]_'.format(frequency)
        if weighting is not None:
            base_url = base_url + '[{}]'.format(weighting)
        return base_url + '.zip'

    def _post_jkp(self,
                  region=None,
                  theme=None,
                  frequency=None,
                  weighting=None):

        url = self.get_url(region, theme, frequency, weighting)
        response = requests.get(url)

        # Check if request was successful (HTTP Status Code 200)
        if response.status_code == 200:
            # Use io.BytesIO to create a byte stream of the content
            zip_data = io.BytesIO(response.content)

            # Open the ZIP file
            with zipfile.ZipFile(zip_data) as z:
                # Extract Excel file from the ZIP
                # Assuming there's only one Excel file in the ZIP
                filename = [name for name in z.namelist() if name.endswith('.csv')][0]
                self._df = pd.read_csv(z.open(filename))
        else:
            raise ValueError(f"Failed to retrieve content: {response.status_code}")


if __name__ == "__main__":

    session = JKPFactors(region='world', theme='all_themes', frequency='daily', weighting='vw_cap')
