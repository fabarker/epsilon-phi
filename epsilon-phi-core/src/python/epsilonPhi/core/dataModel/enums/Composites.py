

class CompositeRiskFreeRates(object):

    composite_rfr_regions =  ['AC World',
                              'World',
                              'Pacific ex-Japan',
                              'Pacific',
                              'Europe ex-UK',
                              'Europe',
                              'EMU',
                              'EM Latin America',
                              'EM Europe and Middle East',
                              'EM Europe',
                              'EM Asia',
                              'Emerging Markets',
                              'EAFE',
                              'AC Europe and Middle East',
                              'AC Asia Pacific',
                              'AC Americas']

    def __init__(self):
        pass

class CompositeFXRates(object):


    composite_currencies = ['AAP',
                            'ACA',
                            'ACW',
                            'AEM',
                            'EEP',
                            'EFE',
                            'ELA',
                            'EMA',
                            'EME',
                            'EMM',
                            'EMU',
                            'ERP',
                            'EXU',
                            'PAC',
                            'PXJ',
                            'WLD']

    def __init__(self):
        pass

class CompositeRates:

    composite_rfr_regions =  [('AC World', 'ACW'),
                              ('World', 'WLD'),
                              ('Pacific ex-Japan', 'PXJ'),
                              ('Pacific', 'PAC'),
                              ('Europe ex-UK', 'EXU'),
                              ('Europe', 'ERP'),
                              ('EMU','EMU'),
                              ('EM Latin America', 'ELA'),
                              ('EM Europe and Middle East', 'EME'),
                              ('EM Europe', 'EEP'),
                              ('EM Asia', 'EMA'),
                              ('Emerging Markets', 'EMM'),
                              ('EAFE', 'EFE'),
                              ('AC Europe and Middle East', 'AEM'),
                              ('AC Asia Pacific', 'AAP'),
                              ('AC Americas', 'ACA')]

    @staticmethod
    def get_currency_from_region(region):
        """Return the currency code corresponding to a region name."""
        for r, c in CompositeRates.composite_rfr_regions:
            if r == region:
                return c
        return None

    @staticmethod
    def get_region_from_currency(currency):
        """Return the region name corresponding to a currency code."""
        for r, c in CompositeRates.composite_rfr_regions:
            if c == currency:
                return r
        return None

    @staticmethod
    def get_regions():
        """Return a list of all region names."""
        return [r for r, _ in CompositeRates.composite_rfr_regions]

    @staticmethod
    def get_currencies():
        """Return a list of all currency codes."""
        return [c for _, c in CompositeRates.composite_rfr_regions]

if __name__ == "__main__":

    currencies = CompositeRates.get_currencies()




