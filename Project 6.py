# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import statsmodels.api as sm


class FX:

    # Constructor
    def __init__(self):
        end_date = '2017-10-01'
        start_date = '2015-05-01'

    # Data Download
        self.__jpn_cpi_data = pdr.DataReader('JPNCPIALLMINMEI', 'fred',
                                             start_date, end_date)
        self.__jpn_cpi_data = self.__jpn_cpi_data['JPNCPIALLMINMEI']

        self.__usa_cpi_data = pdr.DataReader('USACPIALLMINMEI', 'fred',
                                             start_date, end_date)
        self.__usa_cpi_data = self.__usa_cpi_data['USACPIALLMINMEI']

        self.__usa_ir_10 = pdr.DataReader('GS10', 'fred', start_date, end_date)
        self.__usa_ir_10 = self.__usa_ir_10['GS10']

        self.__fx_rate = pdr.DataReader('EXJPUS', 'fred', start_date,
                                        end_date)
        self.__fx_rate = self.__fx_rate['EXJPUS']

    # Getters

    def get_jpn_cpi_data(self):
        return self.__jpn_cpi_data

    def get_usa_cpi_data(self):
        return self.__usa_cpi_data

    def get_usa_interest_rate_10(self):
        return self.__usa_ir_10

    def get_fx_rate(self):
        return self.__fx_rate

    # Regressions

    def __implement_regression_1(self):
        ratio = np.log(self.__usa_cpi_data / self.__jpn_cpi_data)
        explanatory_variable = sm.add_constant(ratio)
        model = sm.OLS(np.log(self.__fx_rate), explanatory_variable)
        results = model.fit()

        print(results.summary())

        # Visualization
        plt.plot(self.__fx_rate, label='Actual')
        plt.plot(np.exp(results.fittedvalues), color='red',
                 label='Equilibrium')
        plt.title("USD/JPY exchange rates")
        plt.legend()
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __implement_regression_2(self):
        ratio = np.log(self.__usa_cpi_data / self.__jpn_cpi_data)
        explanatory_variable = sm.add_constant(ratio)
        X = pd.DataFrame(explanatory_variable)
        X['USACPIALLMINMEI'] = self.__usa_ir_10
        model = sm.OLS(np.log(self.__fx_rate), X)
        results = model.fit()

        print(results.summary())

        # Visualization
        plt.plot(self.__fx_rate, label='Actual')
        plt.plot(np.exp(results.fittedvalues), color='red',
                 label='Equilibrium')
        plt.title("USD/JPY exchange rates")
        plt.legend()
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()

    def main(self):
        self.__implement_regression_1()
        self.__implement_regression_2()


fx = FX()
fx.main()
