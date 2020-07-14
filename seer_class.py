'''
A support class used to quickly test multiple options with Prophet.
The fundamental method is test_model.
'''

import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py


class Seer:
    
    def merge_forecast(self, actual, forecast):
        output = pd.merge(left = actual, 
                          right = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],  #forecast is the result of model.predict()
                          how = 'right',
                          on = 'ds')


        output['mae'] = abs(output['yhat'] - output['y'])
        output['mape'] = output['mae'] / output['y']

        return output


    #show interactive Graph
    def show_graph(self, model, forecast):
        '''
        model = Prophet object
        forecast = Dataframe. Result of the model.predict(future)

        '''   
        fig = plot_plotly(model, forecast)
        py.iplot(fig)

    
    def test_model(self, model, input_df, present_and_future = None, viz = True):
        '''
        INPUT:    
        - model: A Prophet object (as a model)
        - input_df: An input dataframe that will fit model, hence requires the Prophet structure [['ds','y']]:    
        - present_and_future (Optional): A full-time dataframe of present annd future values. 
          It controls the extension of the and supports additional regressors. 

        ---------------------------------
        OUTPUT:
        - A pyplot view of the model plot, plus the seasonal components
        - A dataframe (fc) of the given model
        '''

        # 2. Fit phase. Fit your previously set prophet object with the input dataframe
        model_fit = model.fit(input_df)


        # 3. Predict phase. 
        # Create a forecast: if the future parameter is None, prediction will be run only on the existing data
        if present_and_future is None:
            future = input_df
        else:
            future = present_and_future

        fc_predict = model.predict(future)
        fc = self.merge_forecast(actual = input_df, forecast = fc_predict)


        #Bring together forecast (yhat,yhat_lower,yhat_upper) and the full dataframe with extended regressors, and perform cleaning
        fc = pd.merge(left = future, right = fc, how = 'left', on = 'ds', suffixes=('', '_DROP'))#actual = fc, forecast = future)
        to_drop = fc.columns[fc.columns.str.endswith('_DROP')]
        fc = fc.drop(columns = to_drop, axis = 1)


        #Immediately print out error metrics for quick check
        print('The Mean Absolute Percentage Error of our forecasts is {}'.format(round(fc['mape'].mean(), 4)))

        # 4. Plot phase.Plot results for visual representation
        if viz == True:
            self.show_graph(model, fc)
            model.plot_components(fc_predict)
        else:
            pass
        
        return fc