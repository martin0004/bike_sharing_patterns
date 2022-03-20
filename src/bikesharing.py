################################################################################
#                                                                              #
# IMPORTS                                                                      #
#                                                                              #
################################################################################

# Standard Libraries

import datetime
from typing import Dict, List, Tuple

# Third Party Libraries

import numpy as np
import pandas as pd
import plotly.graph_objects as go


################################################################################
#                                                                              #
# GLOBAL VARIABLES                                                             #
#                                                                              #
################################################################################

# Means / standard deviations for scaling numerical features
MEANS = {"temp": 0.513, "hum": 0.626, "windspeed": 0.191, "cnt": 186.334}
STDS = {"temp": 0.195, "hum": 0.195, "windspeed": 0.122, "cnt": 179.023}


################################################################################
#                                                                              #
# DATALOADER                                                                   #
#                                                                              #
################################################################################


class DataLoader():
    
    def __init__(self) -> None:
        
        pass
    
    def load(self, file_path: str) -> pd.DataFrame:
        
        df_data = pd.read_csv(file_path)        
        
        return df_data


################################################################################
#                                                                              #
# DATA PRE-PROCESSOR                                                           #
#                                                                              #
################################################################################

class DataPreprocessor():
    
    def __init__(self, means: Dict, stds: Dict):
      
        # Features
        self.features = ["season", "yr", "mnth", "hr", "holiday", "weathersit",
                         "weekday", "temp", "hum", "windspeed"]
        
        # Features to be one-hot encoded
        self.features_one_hot_encode = ["season", "weathersit", "mnth", "hr", "weekday"]
        
        # Features to be scaled
        self.features_scale = ["temp", "hum", "windspeed"]
        
        # Means and standard deviations of features to scale
        # (values derived from training dataset)
        self.means = means
        self.stds = stds
        
        # Features ordered
        #
        # This is the order in which features must be after pre-processing.
        # It ensures a feature always has the same index when going tru
        # the neural network. Any missing feature column with be filled with 0.
        
        self.features_ordered = ['holiday', 'hr_0', 'hr_1', 'hr_10', 'hr_11', 'hr_12',
                                 'hr_13', 'hr_14', 'hr_15', 'hr_16', 'hr_17', 'hr_18',
                                 'hr_19', 'hr_2', 'hr_20', 'hr_21', 'hr_22', 'hr_23',
                                 'hr_3', 'hr_4', 'hr_5', 'hr_6', 'hr_7', 'hr_8', 'hr_9',
                                 'hum', 'mnth_1', 'mnth_10', 'mnth_11', 'mnth_12', 'mnth_2',
                                 'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6', 'mnth_7', 'mnth_8',
                                 'mnth_9', 'season_1', 'season_2', 'season_3', 'season_4',
                                 'temp', 'weathersit_1', 'weathersit_2', 'weathersit_3',
                                 'weathersit_4', 'weekday_0', 'weekday_1', 'weekday_2',
                                 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'windspeed', 'yr']
        
        # Targets
        self.targets = ["cnt"]
    
    def one_hot_encode(self, df_in: pd.DataFrame, fts: List[str]) -> pd.DataFrame:
        
        df_out = df_in.copy()
        df_out = pd.get_dummies(df_out, columns=fts)
        
        return df_out
    
    def scale(self, df_in: pd.DataFrame, fts: List[str]) -> pd.DataFrame:
        
        df_out = df_in.copy()
        
        for ft in fts:
            df_out[ft] = (df_out[ft] - self.means[ft]) / self.stds[ft]
        
        return df_out
    
    def order(self, df_in: pd.DataFrame, fts: List[str]) -> pd.DataFrame:
        
        m = df_in.shape[0]
        n = len(fts)       
        
        data = np.zeros((m,n))
        
        df_out = pd.DataFrame(columns=fts, data=data)
        
        for ft in fts:            
            if ft in df_in.columns:                
                df_out[ft] = df_in[ft].copy()
        
        return df_out
    
    def run(self, df_in: pd.DataFrame, targets = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        df_features = df_in.copy()
        df_features = df_features[self.features]
        df_features = self.one_hot_encode(df_features, self.features_one_hot_encode)
        df_features = self.scale(df_features, self.features_scale)
        df_features = self.order(df_features, self.features_ordered)
        
        if targets:
            
            df_targets = df_in.copy()
            df_targets = df_targets[self.targets]
            df_targets = self.scale(df_targets, self.targets)
            
            return df_features, df_targets
        
        else:
            
            return df_features

        return df_out


################################################################################
#                                                                              #
# NEURAL NETWORK                                                               #
#                                                                              #
################################################################################


class NeuralNetwork(object):
    """ Class for building a simple fully-connected neural network.
    
        Note
        ---------
        Code below based on original code provided by Udacity.
        Original code can be found here:
        https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing
    
    """
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

    def activation_function(self, x):
        
        return 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        
        # Hidden layer
        hidden_inputs = X @ self.weights_input_to_hidden
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        final_inputs = hidden_outputs @ self.weights_hidden_to_output
        final_outputs = final_inputs
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        
        # Reshape inputs to make clear these are row vectors (1 x n matrices)
        
        final_outputs = np.reshape(final_outputs, (1,-1))
        hidden_outputs = np.reshape(hidden_outputs, (1,-1))
        X = np.reshape(X, (1,-1))
        y = np.reshape(y, (1,-1))  

        # Output error
        error = y - final_outputs
        
        # Error terms
        g_prime_output = 1             # output layer has no activation function, so g'(x) = 1
        output_error_term = error * g_prime_output
        
        g_prime_hidden = hidden_outputs * (1 - hidden_outputs) # Derivative of sigmoid
        hidden_error_term = output_error_term @ self.weights_hidden_to_output.T * g_prime_hidden
        
        # Weight updates
        delta_weights_i_h += self.lr * X.T @ hidden_error_term
        delta_weights_h_o += self.lr * hidden_outputs.T @ output_error_term
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        # Update weights
        self.weights_hidden_to_output += delta_weights_h_o / n_records
        self.weights_input_to_hidden += delta_weights_i_h / n_records 

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        # Convert to numpy array (in case features is a DataFrame)
        # This makes all array methods available.
        features = np.array(features)
        
        # Forward pass

        # Hidden layer
        hidden_inputs = features @ self.weights_input_to_hidden
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Output layer
        final_inputs = hidden_outputs @ self.weights_hidden_to_output
        final_outputs = final_inputs
        
        return final_outputs

    
################################################################################
#                                                                              #
# TRAINING                                                                     #
#                                                                              #
################################################################################


def MSE(targets, predictions):
    """Mean Square Error."""
    
    e = np.mean( (targets - predictions)**2 )
    
    return e


################################################################################
#                                                                              #
# UTILITY FUNCTIONS                                                            #
#                                                                              #
################################################################################


def get_timestamps_from_df_data(df) -> List[datetime.datetime]:
    """Get a list of timestamp from rows of a DataFrame containing
       raw data.
    """
    
    timestamps = []
    
    for index, row in df.iterrows():
    
        year  = int(row["dteday"][:4])
        month = int(row["dteday"][5:7])
        day   = int(row["dteday"][-2:])
        hour  = int(row["hr"])
    
        timestamp = datetime.datetime(year, month, day, hour)
        timestamps.append(timestamp)
    
    return timestamps


def plot_predictions(predictions: np.array,               # predictions made by nn (raw output - scaled values)
                     means: Dict,                         # mean values for unscaling predictions
                     stds: Dict,                          # standard deviations for unscaling predicitons
                     timestamps: List[datetime.datetime], # timestamps to use on x-axis instead of prediction indices [optional]
                     df_targets: pd.DataFrame = None,     # targets [optional]
                     ) -> None:
    
    # Values for x axis
    
    if timestamps is not None: # Use timestamps on x axis
        x = timestamps
    else: # Use prediction indices on x axis
        n = predictions.shape[0]
        x = [i for i in range(n)]
    
    # Values for y axis
    
    y = predictions[:,0] * stds["cnt"] + means["cnt"]
    
    # Plot figure    
    
    fig = go.Figure()
    
    trace = go.Scatter(x=x, y=y, mode="lines", name="Predictions", line_color="blue")
    fig.add_trace(trace)
    
    # Add targets on chart [optional]
    
    if df_targets is not None:

        targets = df_targets["cnt"] * stds["cnt"] + means["cnt"]
        trace = go.Scatter(x=x, y=targets, mode="lines", name="Targets", line_color="darkorange")
        fig.add_trace(trace)

    fig.layout.title = "Number of bike rentals"
    fig.layout.width = 600
    fig.layout.height = 500
        
    fig.show()