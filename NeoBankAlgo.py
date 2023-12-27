import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

nRowsRead = 400
nColRead = 400
DBpath = "~/Desktop/NeoBank/"


def Station1_loadData():

    client = pd.ExcelFile(DBpath + "Client_Cash_Accounts.xlsx")
    corn = pd.read_excel(DBpath + "CORN_PriceHistory.xlsx", header=15) . dropna(axis=1)
    wheat = pd.read_excel(DBpath + "WHEAT_PriceHistory.xlsx", header=15) . dropna(axis=1)
    news = pd.read_json(DBpath + "Commodity_News.json")
    weather = pd.read_csv(DBpath + "Weather.csv", delimiter=',', nrows = nRowsRead, encoding= 'unicode_escape')
    
    # COMBINE CLIENT CASH FLOWS TO A CENRAL DATABASE
    
    cd = {}
    for sheet_name in client.sheet_names:
        cd[sheet_name] = client.parse(sheet_name)   
    cd["1x"]["Client"] = 1
    cd["2x"]["Client"] = 2
    cd["2x"].rename(columns = {'Row Labels' : 'Date'},inplace=True)
    colsx = list(cd["2x"].columns.values)
    colsx= ['Client','Date', 'Daily Flows', 'Cash Balance']
    cd["2x"] = cd["2x"][colsx]
    cd["3x"]["Client"] = 3
    cd["3x"] = cd["3x"][colsx]
    cd["4x"]["Client"] = 4
    cd["4x"].rename(columns = {'Unnamed: 0' : 'Date', 
                               'Unnamed: 1' : 'Daily Flows', 
                               'Unnamed: 2' : 'Cash Balance'}, inplace=True)
    cd["4x"]=cd["4x"][colsx]
    cd["4x"]=cd["4x"][1:-1]
    cd["5x"]["Client"] = 5
    cd["5x"] = cd["5x"][colsx]
    cflowx = pd.concat([cd["1x"],
                       cd["2x"],
                       cd["3x"],
                       cd["4x"],
                       cd["5x"]])
    cflow = pd.concat([cd["1"],
                       cd["2"],
                       cd["3"],
                       cd["4"],
                       cd["5"]])
    
    #CLEAN WEATHER DATA FOR STATION 2
    
    weather.columns = ['Date','MinTemp','MaxTemp','Rainfall', 'WindDir','MaxWindSpeed','MaxWindSpeedTime','9amTemp',
                       '9amHum','9amWindD','9amWindS','3pmTemp','3pmHum','3pmWindD','3pmWindS']
    for n in weather.index:
        if weather['9amWindS'][n] == 'Calm':
            weather['9amWindS'][n] = 0
            
        elif weather['3pmWindS'][n] == 'Calm':
            weather['3pmWindS'][n] = 0
    time_index = pd.date_range('2019-01-07', periods=len(weather),  freq='d')
    weather["Date"] = time_index
    weather = weather.dropna()
    weather['9amWindS'] = weather['9amWindS'].astype(int)
    weather['3pmWindS'] = weather['3pmWindS'].astype(int)
    
    return cflow, cflowx, weather, corn, wheat, news


def Station2_RelevanttFeats():
    
    [cflow, cflowx, weather, corn, wheat, news] = Station1_loadData()

    # CREATE ADDITIONAL INPUTS FOR OUR RNN MODEL 

    weather['AvgTemp'] = (weather['MinTemp'] + weather['MaxTemp'])/2
    weather['AvgHum'] = (weather['9amHum'] + weather['3pmHum'])/2
    weather['AvgDayTemp'] = (weather['9amTemp'] + weather['3pmTemp'])/2
    weather['AvgDayWind'] = (weather['9amWindS'] + weather['3pmWindS'])/2
    
    
    
    df = pd.merge(corn,wheat,suffixes=["_corn","_wheat"],on="Date")
    df = pd.merge(df, weather, on="Date")
    
    return df


def Station3(df,features_considered):
    
    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
    
        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)
    
    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size
    
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])
    
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
    
        return np.array(data), np.array(labels)
    
    def create_time_steps(length):
        return list(range(-length, 0))
    
    def baseline(history):
        return np.mean(history)
    
    def show_plot(plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0
    
        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                         label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future+5)*2])
        plt.xlabel('Time-Step')
        return plt.show()
    
    def plot_train_history(history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()
        plt.show()
    
    def multi_step_plot(history, true_future, prediction):
        plt.figure(figsize=(12, 6))
        num_in = create_time_steps(len(history))
        num_out = len(true_future)
    
        plt.plot(num_in, np.array(history[:, 1]), label='History')
        plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
                 label='True Future')
        if prediction.any():
            plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                     label='Predicted Future')
        plt.legend(loc='upper left')
        plt.show()
    
    # DATA INPUTS FOR ADDITIONAL FEATURES

    tf.random.set_seed(88)
    BATCH_SIZE = 50
    BUFFER_SIZE = 1000
    EVALUATION_INTERVAL = 300
    EPOCHS = 5
    TRAIN_SPLIT = 80
    past_history = 30
    future_target = 5
    STEP = 1
    
    
    features = df[features_considered]
    features.index = df['Date']
    print(features.head())
    features.plot(subplots=True)
    plt.show()
    
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    

    
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
    
    print ('Single window of past history : {}'.format(x_train_single[0].shape))
    
    train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
    
    # BUILD SINGLE POINT FORWARD LSTM RNN MODEL
    
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))
    
    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    
    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)
    
    plot_train_history(single_step_history,'Single Step Training and validation loss')
    
    for x, y in val_data_single.take(3):
        print(val_data_single)
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          single_step_model.predict(x)[0]], 5,
                         'Single Step Prediction')
        plot
    
    # MUTLISTEP PREDICTION MODEL - 5 OBSERVATIONS FORWARD
    
    future_target = 5
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)
    
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    
    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]))
    
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(5))
    
    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=50)
    
    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
    
    # RETURN RESULTS

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    

def Station3_Sentiment(news):

    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = news['Headline'].apply(vader.polarity_scores).tolist()
    
    # Convert the 'scores' list of dicts into a DataFrame
    scores_news = pd.DataFrame(scores)
    
    print(scores_news.head(20))
    
    # Join the DataFrames of the news and the list of dicts
    news = news.join(scores_news, rsuffix='_right')
    news.to_json(DBpath + 'vader_scores.json')
    print(news.head(10))
    
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = news.groupby(['Date']).mean()
    
    print(mean_scores)
    
    # Boxplot of time-series of sentiment score for each date
    mean_scores.plot(kind = 'box')
    plt.grid()
    plt.title('Box plot of sentiment score overtime')
    plt.show()
    
    # get the lastest month sentiment score as adjustment
    lastest_month_score = mean_scores.tail(20).mean()
    lastest_month_score.fillna(0,inplace=True)
    lastest_month_score.plot(kind='bar')
    plt.title('Past Month Sentiment Score')
    plt.show()
    
def Station4():
    
    [cflow, cflowx, weather, corn, wheat, news] = Station1_loadData()
    df = Station2_RelevanttFeats()   

    wheat_feats = ['Last_wheat','AvgTemp','AvgHum','Rainfall','CVol_wheat','Open Interest_wheat']
    corn_feats = ['Last_corn','AvgTemp','AvgHum','Rainfall','CVol_corn','Open Interest_corn']
    
    # RNN MODEL IMPLEMENTATION
        
    Station3(df,wheat_feats)
    Station3(df,corn_feats)
    
    # NPL MODEL IMPLEMENTATION
    
    Station3_Sentiment(news)
    
    # BALANCE HISORY PLOT
    
    cflow_display = (cflow[['Transaction Date','Description','Flow','Balance']])
    cflow_display.rename(columns = {'Flow' : 'Amount'}, inplace=True)
    cflow_1 = cflow.loc[cflow["Client"] == 1, cflow.columns]
    plt.plot(cflow_1["Transaction Date"].head(90), cflow_1["Balance"].head(90))
    
    # PAST 3-MONTH COMMODITY PRICE
    
    features_considered = ['Settlement Price']
    features = wheat[features_considered].head(90)
    features.index = wheat['Date'].head(90)
    print(features.head())
    features.plot(subplots=True, title = "3-month Wheat Settlement Price")
    plt.show()
    
    features_considered = ['Settlement Price']
    features = corn[features_considered].head(90)
    features.index = corn['Date'].head(90)
    print(features.head())
    features.plot(subplots=True, title = "3-month Corn Settlement Price")
    plt.show()

Station4()





    

    
