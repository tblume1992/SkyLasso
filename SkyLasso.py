"""
Created on Sun Jun  7 20:54:47 2020

@author: Tyler Blume
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm

class SkyLasso:
    
    def __init__(self,y, 
                 exogenous = None, 
                 trend = True, 
                 changepoint = True, 
                 poly = 3,
                 scale_y = True,
                 scale_X = False,
                 trend_lambda = .01,
                 changepoint_lambda = .005,
                 exogenous_lambda = .001,
                 seasonal_lambda = .001,
                 linear_changepoint_lambda = .001,
                 seasonal_period = 12,
                 components = 10,
                 approximate_changepoints = True,
                 linear_changepoint = True,
                 n_changepoints = 25,
                 intercept = True,
                 pre = 1,
                 post = 1,
                 bagged_samples = 0,
                 global_cost = 'maicc'):
        self.y = y
        self.exogenous = exogenous
        self.trend = trend
        self.changepoint = changepoint
        self.poly = poly
        self.seasonal_period = seasonal_period
        self.seasonal_lambda = seasonal_lambda
        self.changepoint_lambda = changepoint_lambda
        self.exogenous_lambda = exogenous_lambda
        self.trend_lambda = trend_lambda
        self.scale_y = scale_y
        self.scale_X = scale_X
        self.components = components
        self.approximate_changepoints = approximate_changepoints
        self.linear_changepoint = linear_changepoint
        self.linear_changepoint_lambda = linear_changepoint_lambda
        self.pre = pre
        self.post = post
        self.intercept = intercept
        self.n_changepoints = n_changepoints
        self.bagged_samples = bagged_samples
        self.global_cost = 'maicc'
        
        
        return
        
    def get_fourier_series(self, t, p=12, n=10):
        x = 2 * np.pi * np.arange(1, n + 1) / p
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        
        return fourier_series
    
    def get_harmonics(self):
        harmonics = self.get_fourier_series(np.arange(len(self.y)), 
                                self.seasonal_period, 
                                n = self.components)
        
        return harmonics

    def get_future_harmonics(self):
        harmonics = self.get_fourier_series(np.arange(len(self.y) + self.n_steps), 
                                self.seasonal_period, 
                                n = self.components)
        
        return harmonics
    
    def get_changepoint(self):
        changepoints = np.zeros(shape=(len(self.y),int(len(self.y))))
        for i in range(int(len(self.y) - 1)):
            changepoints[:i + 1, i] = 1
        
        return pd.DataFrame(changepoints)
    
    def get_linear_changepoint(self, future_steps = 0):
        n_changepoints = self.n_changepoints
        array_splits = np.array_split(np.array(self.y),n_changepoints)
        y = np.array(self.y)
        initial_point = y[0]
        final_point = y[-1]
        changepoints = np.zeros(shape=(len(y) + future_steps,n_changepoints))
        for i in range(n_changepoints):
            moving_point = array_splits[i][-1]
            if i == 0:
                len_splits = len(array_splits[i])
            else:
                len_splits = len(np.concatenate(array_splits[:i+1]).ravel())
            slope = (moving_point - initial_point)/(len_splits)
            slope = slope*self.pre
            if i != n_changepoints - 1:        
                reverse_slope = (final_point - moving_point)/(len(y) - len_splits)
                reverse_slope = reverse_slope*self.post
            
            changepoints[0:len_splits, i] = slope * (1+np.array(list(range(len_splits))))
            changepoints[len_splits:, i] = changepoints[len_splits-1, i] + reverse_slope * (1+np.array(list(range((len(y) + future_steps - len_splits)))))
        if future_steps:
            changepoints[len(self.y):, :] = np.array([np.mean(changepoints[len(self.y):, :], axis = 1)]*np.shape(changepoints)[1]).transpose()

        return changepoints 
        
    
    def get_approximate_changepoints(self):
        from sklearn import tree
        changepoints = np.zeros(shape=(len(self.y),int(len(self.y))))
        for i in range(int(len(self.y) - 1)):
            changepoints[:i + 1, i] = 1
        changepoints = pd.DataFrame(changepoints)
        clf = tree.DecisionTreeRegressor(criterion = 'mae', max_depth = 10)
        clf = clf.fit(changepoints, self.y)
        imp = pd.Series(clf.feature_importances_)
        imp_idx = imp[imp > .01]
        model_dataset = changepoints.iloc[:, imp_idx.index]
        
        return model_dataset
    
    def get_trend(self):
        n = len(self.y)
        linear_trend = np.arange(len(self.y))
        trends = np.asarray(linear_trend).reshape(n, 1)
        trends = np.append(trends, np.asarray(linear_trend**2).reshape(n, 1), axis = 1)

        return trends

    def get_future_trend(self):
        n = len(self.y) + self.n_steps
        linear_trend = np.arange(n)
        trends = np.asarray(linear_trend).reshape(n, 1)
        trends = np.append(trends, np.asarray(linear_trend**2).reshape(n, 1), axis = 1)

        return trends
    
    def fit(self):
        if self.seasonal_period:
            X = self.get_harmonics()
            regularization_array = np.ones(np.shape(X)[1])*self.seasonal_lambda
            component_dict = {'harmonics': [0, np.shape(X)[1] - 1]}
        elif self.trend:
            X = self.get_trend()
            component_dict = {'trends': [0, np.shape(X)[1] - 1]}
            regularization_array = np.ones(np.shape(X)[1])*self.trend_lambda           
        if self.trend and self.seasonal_period:
            trends = self.get_trend()
            component_dict['trends'] = [np.shape(X)[1], np.shape(trends)[1]]
            X = np.append(X, trends, axis = 1)
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(trends)[1])*self.trend_lambda)
            
        if self.changepoint:
            if self.approximate_changepoints:
                changepoints = self.get_approximate_changepoints()
            else:   
                changepoints = self.get_changepoint()
            self.changepoints = changepoints
            try:
                component_dict['changepoint'] = [np.shape(X)[1], np.shape(changepoints)[1]]
                X = np.append(X, changepoints, axis = 1)
                regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(changepoints)[1])*self.changepoint_lambda)
            except:
                X = self.changepoints
                component_dict = {'changepoint': [0, np.shape(X)[1] - 1]}
                regularization_array = np.ones(np.shape(changepoints)[1])*self.changepoint_lambda
                X = np.array(X)

        
        if self.scale_X:
            self.X_scaler = StandardScaler()
            self.X_scaler = self.X_scaler.fit(X)
            X = self.X_scaler.transform(X)
        if self.scale_y:
            self.scaler = StandardScaler()
            self.scaler.fit(np.asarray(self.y).reshape(-1, 1))   
            self.og_y = self.y.copy()
            self.y = self.scaler.transform(np.asarray(self.y).reshape(-1, 1))

       
        if self.exogenous is not None:
            component_dict['exogenous'] = [np.shape(X)[1], np.shape(self.exogenous)[1]]
            X = np.append(X, self.exogenous, axis = 1)
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(self.exogenous)[1])*self.exogenous_lambda)
            
        if self.linear_changepoint:
            self.linear_changepoints = self.get_linear_changepoint()
            component_dict['linear_changepoint'] = [np.shape(X)[1], np.shape(self.linear_changepoints)[1]]
            X = np.append(X, self.linear_changepoints, axis = 1)
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(self.linear_changepoints)[1])*self.linear_changepoint_lambda)            
        if self.intercept:
            intercept_term = np.asarray(np.ones(len(self.y))).reshape(-1, 1)
            component_dict['intercept'] = [np.shape(X)[1], np.shape(intercept_term)[1]]
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(intercept_term)[1])*self.trend_lambda)
            X = np.append(X, intercept_term, axis = 1)
        if self.bagged_samples:
            params = []
            weights = []
            for i in tqdm(range(self.bagged_samples)):
                sample = sklearn.utils.resample(self.y, X)
                lasso = sm.GLM(exog = sample[1], endog = sample[0])
                lasso = lasso.fit_regularized(alpha = regularization_array)
                sampled_params = lasso.params
                c = len([i for i in sampled_params if i != 0])
                params.append(sampled_params)
                try:
                    weights.append(-self.calc_cost(lasso.predict(X), c))
                except: 
                    weights.append(np.nan)
            average_params = np.average(params, weights = weights,axis = 0)
            lasso = sm.GLM(exog = X, endog = self.y)
            lasso = lasso.fit_regularized(alpha = regularization_array)
            lasso.params = average_params
        else:
            lasso = sm.GLM(exog = X, endog = self.y)
            lasso = lasso.fit_regularized(alpha = regularization_array)
        self.lasso = lasso
        self.component_dict = component_dict
        self.X = X
        
        return 
    
    def calc_cost(self, prediction, c):
        n = len(self.y)
        if self.global_cost == 'maic':
          cost = 2*(c) + n*np.log(np.sum((self.y - prediction )**2)/n)
        if self.global_cost == 'maicc':
          cost = (2*c**2 + 2*c)/(n-c-1) + 2*(c) + \
                  n*np.log(np.sum((self.y - prediction )**2)/n)    
        elif self.global_cost == 'mbic':
          cost = n*np.log(np.sum((self.y - prediction )**2)/n) + \
                (c) * np.log(n)
        return cost  
 
    def predict(self, X = None):
        if X is None:
            X = self.X    
        predicted = self.lasso.predict(X)
        if self.scale_y == True:
            predicted = self.scaler.inverse_transform(predicted.reshape(-1, 1))
            
        return predicted
    
    def make_future_dataframe(self, n_steps, exogenous = None):
        self.n_steps = n_steps
        if self.seasonal_period:
            X = self.get_future_harmonics()
        else:
            X = self.get_future_trend()
        X = X[-n_steps:, :]
        if self.trend and self.seasonal_period:
            trends = self.get_future_trend()
            trends = trends[-n_steps:, :]
            X = np.append(X, trends, axis = 1)
            
        if self.changepoint:
            future_changepoints = self.changepoints.iloc[-1:, :]
            future_changepoints = pd.concat([future_changepoints]*n_steps, ignore_index=True)
            X = np.append(X, future_changepoints, axis = 1)
 
        if self.scale_X:
            X = self.X_scaler.transform(X)            

        if self.exogenous is not None:
            X = np.append(X, exogenous, axis = 1)


        if self.linear_changepoint:
            future_linear_changepoints = self.get_linear_changepoint(n_steps)
            future_linear_changepoints = future_linear_changepoints[-n_steps:, :]
            X = np.append(X, future_linear_changepoints, axis = 1)
        if self.intercept:
            intercept_term = np.asarray(np.ones(n_steps)).reshape(n_steps, 1)
            X = np.append(X, intercept_term, axis = 1)
            
        return X
            
    def plot_components(self):
        fig, ax = plt.subplots(len(self.component_dict.keys()), figsize = (16,16))
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.X.copy()
            mask = np.repeat(True, np.shape(self.X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.predict(Test)
            if self.scale_y and component != 'intercept':
                component_prediction = component_prediction - np.mean(self.og_y)
            ax[i].plot(component_prediction)
            ax[i].set_title(component)
        plt.show()

    def get_components(self):
        components = {}
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.X.copy()
            mask = np.repeat(True, np.shape(self.X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.predict(Test)
            if self.scale_y and component != 'intercept':
                component_prediction = component_prediction - np.mean(self.og_y)
            components[component] = component_prediction
        
        return components
    
    def summary(self):
        return self.lasso.summary()
