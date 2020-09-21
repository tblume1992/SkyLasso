# SkyLasso
```python
    import quandl
    import matplotlib.pyplot as plt
    import fbprophet
    import SkyLasso as sl
    data = quandl.get("BITSTAMP/USD")
    
    y = data['Low']
    y = y[-730:]
    
    
    df = pd.DataFrame(y)
    df['ds'] = y.index
    #adjust to make ready for Prophet
    df.columns = ['y', 'ds']
    model = fbprophet.Prophet()
    model.fit(df)
    future_df = model.make_future_dataframe(150)
    prophet_forecast = model.predict(future_df)    
    
    skylasso_model = sl.SkyLasso(y, 
                                    changepoint = False,
                                    trend = False,
                                    linear_changepoint = True,
                                    seasonal_period = 365,
                                    scale_y = True,
                                    scale_X = False,
                                    intercept = True,
                                    )

        
    skylasso_model.fit()
    predicted = skylasso_model.predict()
    future_df = skylasso_model.make_future_dataframe(150)
    forecast = skylasso_model.predict(future_df)
    plt.plot(np.append(predicted, forecast), label = 'Lasso')
    plt.plot(prophet_forecast['yhat'], label = 'Prophet')
    plt.plot(y.values, color = 'black', label = 'Actual')
    plt.legend()
    plt.show()
    skylasso_model.plot_components()
    skylasso_model_components = skylasso_model.get_components()    
    plt.plot(skylasso_model_components['linear_changepoint'] + skylasso_model_components['intercept'], label = 'Lasso')    
    plt.plot(prophet_forecast['trend'], label = 'Prophet')
    plt.legend()
    plt.show()
```
![alt text](https://github.com/tblume1992/SkyLasso/blob/master/sl_1.png?raw=true)
![alt text](https://github.com/tblume1992/SkyLasso/blob/master/sl_2.png?raw=true)
