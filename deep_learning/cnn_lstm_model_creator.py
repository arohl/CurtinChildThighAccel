import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Class to create a CNN-LSTM model
class CreateCNNLSTM:
    def __init__(self, config) -> None:
        # Initialize the configuration parameters
        self.input_shape = config["input_shape"]  # Shape of input data
        self.num_classes = config["num_classes"]  # Number of classes for classification
        self.lr_params = config["lr_params"]  # Learning rate parameters
        self.loss_func = config["loss_func"]  # Loss function for training
        pass

    # Create the entire CNN-LSTM model
    def create_model(self, model_struc):
        model = models.Sequential()  # Create a sequential model

        # Add Conv1D layers
        self.create_conv1d_layers(model, **model_struc["conv1d_layers"])
        model.add(layers.MaxPooling1D(pool_size=3))  # Add MaxPooling layer
        # Add LSTM layers
        self.create_lstm_layers(model, **model_struc["lstm_layers"])
        # Add Dense layers
        self.create_dense_layers(model, **model_struc["dense_layers"])

        return model

    # Create Conv1D layers
    def create_conv1d_layers(self, model, parameters):
        for i in range(len(parameters)):
            if i == 0:
                # For the first Conv1D layer, specify input shape
                model.add(
                    layers.Conv1D(
                        filters=parameters[i]["filters"],
                        kernel_size=parameters[i]["kernel_size"],
                        activation="relu",
                        input_shape=self.input_shape,
                    )
                )
            else:
                # For subsequent Conv1D layers, no need to specify input shape
                model.add(
                    layers.Conv1D(
                        filters=parameters[i]["filters"],
                        kernel_size=parameters[i]["kernel_size"],
                        activation="relu",
                    )
                )
            model.add(
                layers.BatchNormalization()
            )  # Add BatchNormalization after each Conv1D layer

    # Create LSTM layers
    def create_lstm_layers(self, model, parameters):
        for i in range(len(parameters)):
            if i == 0:
                # For the first LSTM layer, return sequences (used for stacked LSTMs)
                model.add(
                    layers.LSTM(units=parameters[i]["units"], return_sequences=True)
                )
            else:
                # For subsequent LSTM layers, no need to return sequences
                model.add(layers.LSTM(units=parameters[i]["units"]))

        model.add(layers.Flatten())  # Flatten the output of LSTM layers

    # Create Dense layers
    def create_dense_layers(self, model, parameters):
        for i in range(len(parameters)):
            # Add a Dense layer with ReLU activation, L2 regularization, BatchNormalization, and Dropout
            model.add(
                layers.Dense(
                    parameters[i]["units"],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(parameters[i]["reg_rate"]),
                )
            )
            model.add(
                layers.BatchNormalization()
            )  # Add BatchNormalization after each Dense layer
            model.add(layers.Dropout(parameters[i]["dropout_rate"]))  # Apply Dropout

        model.add(
            layers.Dense(self.num_classes, activation="softmax")
        )  # Output layer with softmax activation

    # Compile the model with specified loss, optimizer, and metrics
    def compile_model(self, model):
        # Create a learning rate schedule for the optimizer
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.lr_params["learning_rate"],
            self.lr_params["decay_steps"],
            self.lr_params["decay_rate"],
        )

        # Compile the model with specified loss function, optimizer, and metrics
        model.compile(
            loss=self.loss_func,
            optimizer=tf.keras.optimizers.Adam(learning_rate_schedule),
            metrics=["accuracy"],
        )

        return model
