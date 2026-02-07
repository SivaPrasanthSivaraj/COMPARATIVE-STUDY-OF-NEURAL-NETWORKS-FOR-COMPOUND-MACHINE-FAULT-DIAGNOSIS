"""
Parallel CNN-LSTM Model for Bearing Fault Diagnosis
===================================================

This implementation provides a robust dual-branch architecture that processes 
vibration data through both temporal (LSTM) and spatial-frequency (CNN) pathways
before fusing the learned features for fault classification.

Key Features:
- Dual-branch parallel processing
- Robust regularization to prevent overfitting
- Comprehensive data preprocessing and augmentation
- Advanced training strategies with early stopping
- Extensive evaluation and visualization tools

Dataset Structure:
- Load Levels: H, L, M1, M2, M3, U1, U2, U3 (8 levels)
- Fault Types: B (Baseline), IR (Inner Race), OR (Outer Race), H (Hole)  
- Sampling Rates: 8kHz, 16kHz
- Rotating Speeds: 600, 800, 1000, 1200, 1400, 1600 RPM
"""

import os
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class BearingDataset:
    """
    Comprehensive dataset handler for bearing fault diagnosis data
    """
    
    def __init__(self, data_dir: str, sample_length: int = None):
        """
        Initialize the dataset handler
        
        Args:
            data_dir: Path to the root data directory
            sample_length: Length of time-series samples to extract (None = use full signal)
        """
        self.data_dir = Path(data_dir)
        self.sample_length = sample_length  # None means use full signal
        self.fault_type_mapping = {
            'B': 0,   # Baseline (Healthy)
            'IR': 1,  # Inner Race fault
            'OR': 2,  # Outer Race fault  
            'H': 3    # Hole fault
        }
        self.load_level_mapping = {
            'H': 0, 'L': 1, 'M1': 2, 'M2': 3, 'M3': 4, 'U1': 5, 'U2': 6, 'U3': 7
        }
        
        # Data storage
        self.raw_data = []
        self.spectrograms = []
        self.labels = []
        self.metadata = []
        
    def load_all_data(self, verbose: bool = True) -> None:
        """
        Load all .mat files from the dataset directory
        
        Args:
            verbose: Whether to print progress information
        """
        mat_files = list(self.data_dir.rglob("*.mat"))
        
        if verbose:
            print(f"Found {len(mat_files)} .mat files")
            print("Loading data...")
        
        for i, file_path in enumerate(mat_files):
            if verbose and (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(mat_files)} files")
                
            try:
                # Load MATLAB file
                mat_data = scipy.io.loadmat(file_path)
                
                # Extract data
                vibration_data = mat_data['Data'].flatten()
                spectrogram = mat_data['Spectrogram']  # Shape: (time_windows, freq_bins, time_bins)
                
                # Parse filename for metadata
                filename = file_path.name.replace('.mat', '')
                parts = filename.split('_')
                
                if len(parts) >= 5:
                    load_level = parts[0]
                    fault_type = parts[1]
                    sampling_rate = int(parts[2])
                    bearing_type = parts[3]
                    speed = int(parts[4])
                    
                    # Create multiple samples from long vibration signal
                    samples = self._create_samples(vibration_data)
                    
                    # Use all spectrogram time windows or sample them
                    spec_samples = self._process_spectrograms(spectrogram)
                    
                    # Ensure equal number of samples
                    min_samples = min(len(samples), len(spec_samples))
                    samples = samples[:min_samples]
                    spec_samples = spec_samples[:min_samples]
                    
                    # Store data
                    for j in range(min_samples):
                        self.raw_data.append(samples[j])
                        self.spectrograms.append(spec_samples[j])
                        self.labels.append(self.fault_type_mapping[fault_type])
                        
                        self.metadata.append({
                            'load_level': load_level,
                            'fault_type': fault_type,
                            'sampling_rate': sampling_rate,
                            'bearing_type': bearing_type,
                            'speed': speed,
                            'filename': filename,
                            'sample_idx': j
                        })
                        
            except Exception as e:
                if verbose:
                    print(f"Error loading {file_path}: {e}")
                continue
        
        # Convert to numpy arrays with memory optimization
        # Use float32 instead of float64 to halve memory usage
        self.raw_data = np.array(self.raw_data, dtype=np.float32)
        self.spectrograms = np.array(self.spectrograms, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int32)
        
        if verbose:
            print(f"\nDataset loaded successfully:")
            print(f"Raw data shape: {self.raw_data.shape}")
            print(f"Spectrogram shape: {self.spectrograms.shape}")
            print(f"Labels shape: {self.labels.shape}")
            print(f"Number of samples: {len(self.raw_data)}")
            
            # Print class distribution
            unique, counts = np.unique(self.labels, return_counts=True)
            print("\nClass distribution:")
            fault_names = ['Baseline', 'Inner Race', 'Outer Race', 'Hole']
            for label, count in zip(unique, counts):
                print(f"  {fault_names[label]}: {count} samples")
    
    def _create_samples(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple samples from a long vibration signal using sliding window
        
        Args:
            signal: Long vibration signal
            
        Returns:
            List of signal samples
        """
        samples = []
        
        # Use the complete signal - no data left unused
        # Convert to float32 immediately to save memory
        signal_f32 = signal.astype(np.float32)
        
        # Handle variable sequence lengths by padding/truncating to standard length
        expected_length = 1280000  # Standard length for all files
        if len(signal_f32) < expected_length:
            # Pad with zeros if shorter
            padded_signal = np.pad(signal_f32, (0, expected_length - len(signal_f32)), 'constant')
            samples.append(padded_signal)
        elif len(signal_f32) > expected_length:
            # Truncate if longer (take from middle)
            start = (len(signal_f32) - expected_length) // 2
            samples.append(signal_f32[start:start + expected_length])
        else:
            # Perfect length
            samples.append(signal_f32)
        
        return samples
    
    def _process_spectrograms(self, spectrogram: np.ndarray) -> List[np.ndarray]:
        """
        Process spectrograms to create individual samples
        
        Args:
            spectrogram: Input spectrogram with shape (time_windows, freq_bins, time_bins)
            
        Returns:
            List of spectrogram samples
        """
        # Use all spectrogram data by creating a large 2D spectrogram that preserves spatial structure
        # Stack all time windows horizontally to create one comprehensive spectrogram
        # This maintains the 2D nature required for CNN processing
        n_time_windows, freq_bins, time_bins = spectrogram.shape
        
        # Create a comprehensive 2D spectrogram: (freq_bins, total_time_bins)
        full_spectrogram = np.concatenate([spectrogram[i] for i in range(n_time_windows)], axis=1)
        
        # Convert to float32 to save memory
        full_spectrogram_f32 = full_spectrogram.astype(np.float32)
        samples = [full_spectrogram_f32]
        
        return samples
    
    def get_data_splits(self, test_size: float = 0.2, val_size: float = 0.2, 
                       stratify: bool = True) -> Tuple[Tuple[np.ndarray, ...], ...]:
        """
        Split data into train/validation/test sets
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            stratify: Whether to stratify splits based on labels
            
        Returns:
            ((X_train_raw, X_train_spec, y_train), 
             (X_val_raw, X_val_spec, y_val), 
             (X_test_raw, X_test_spec, y_test))
        """
        if len(self.raw_data) == 0:
            raise ValueError("No data loaded. Call load_all_data() first.")
        
        # First split: separate test set
        stratify_labels = self.labels if stratify else None
        
        (X_raw_temp, X_test_raw, 
         X_spec_temp, X_test_spec, 
         y_temp, y_test) = train_test_split(
            self.raw_data, self.spectrograms, self.labels,
            test_size=test_size, random_state=RANDOM_SEED,
            stratify=stratify_labels
        )
        
        # Second split: separate training and validation sets
        stratify_temp = y_temp if stratify else None
        
        (X_train_raw, X_val_raw,
         X_train_spec, X_val_spec,
         y_train, y_val) = train_test_split(
            X_raw_temp, X_spec_temp, y_temp,
            test_size=val_size, random_state=RANDOM_SEED,
            stratify=stratify_temp
        )
        
        return ((X_train_raw, X_train_spec, y_train),
                (X_val_raw, X_val_spec, y_val),
                (X_test_raw, X_test_spec, y_test))


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    """
    
    def __init__(self, normalize_raw: bool = True, normalize_spec: bool = True):
        """
        Initialize preprocessor
        
        Args:
            normalize_raw: Whether to normalize raw time-series data
            normalize_spec: Whether to normalize spectrogram data
        """
        self.normalize_raw = normalize_raw
        self.normalize_spec = normalize_spec
        self.raw_scaler = StandardScaler() if normalize_raw else None
        self.spec_scaler = StandardScaler() if normalize_spec else None
        
    def fit_transform_train(self, X_raw: np.ndarray, X_spec: np.ndarray, 
                           y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessors on training data and transform
        
        Args:
            X_raw: Raw time-series data
            X_spec: Spectrogram data
            y: Labels
            
        Returns:
            Preprocessed (X_raw, X_spec, y)
        """
        # Validate input shapes
        if len(X_raw.shape) != 2:
            raise ValueError(f"X_raw must be 2D (samples, timesteps), got shape {X_raw.shape}")
        if len(X_spec.shape) != 3:
            raise ValueError(f"X_spec must be 3D (samples, height, width), got shape {X_spec.shape}")
        if len(y.shape) != 1:
            raise ValueError(f"y must be 1D (samples,), got shape {y.shape}")
        # For raw data: fit scaler on flattened data across all samples
        if self.normalize_raw:
            # Flatten all samples into one long sequence for proper normalization
            X_raw_flat = np.concatenate([sample.flatten() for sample in X_raw])
            self.raw_scaler.fit(X_raw_flat.reshape(-1, 1))
            
            # Transform each sample individually
            X_raw_norm = np.array([
                self.raw_scaler.transform(sample.reshape(-1, 1)).flatten()
                for sample in X_raw
            ], dtype=np.float32)
        else:
            X_raw_norm = X_raw
            
        # For spectrogram data: fit scaler on flattened spectrogram data
        if self.normalize_spec:
            # Flatten all spectrograms for proper normalization
            X_spec_flat = np.concatenate([sample.flatten() for sample in X_spec])
            self.spec_scaler.fit(X_spec_flat.reshape(-1, 1))
            
            # Transform each spectrogram individually and reshape back
            X_spec_norm = []
            for sample in X_spec:
                original_shape = sample.shape
                normalized = self.spec_scaler.transform(sample.flatten().reshape(-1, 1))
                X_spec_norm.append(normalized.flatten().reshape(original_shape))
            X_spec_norm = np.array(X_spec_norm, dtype=np.float32)
        else:
            X_spec_norm = X_spec
        
        # Expand dimensions for model input
        X_raw_norm = np.expand_dims(X_raw_norm, axis=-1)  # Add channel dimension
        X_spec_norm = np.expand_dims(X_spec_norm, axis=-1)  # Add channel dimension
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=4)
        
        return X_raw_norm, X_spec_norm, y_categorical
    
    def transform(self, X_raw: np.ndarray, X_spec: np.ndarray, 
                  y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessors
        
        Args:
            X_raw: Raw time-series data
            X_spec: Spectrogram data
            y: Labels
            
        Returns:
            Preprocessed (X_raw, X_spec, y)
        """
        # Validate input shapes
        if len(X_raw.shape) != 2:
            raise ValueError(f"X_raw must be 2D (samples, timesteps), got shape {X_raw.shape}")
        if len(X_spec.shape) != 3:
            raise ValueError(f"X_spec must be 3D (samples, height, width), got shape {X_spec.shape}")
        if len(y.shape) != 1:
            raise ValueError(f"y must be 1D (samples,), got shape {y.shape}")
        # Transform raw data using fitted scaler
        if self.normalize_raw and self.raw_scaler is not None:
            X_raw_norm = np.array([
                self.raw_scaler.transform(sample.reshape(-1, 1)).flatten()
                for sample in X_raw
            ], dtype=np.float32)
        else:
            X_raw_norm = X_raw
            
        # Transform spectrogram data using fitted scaler
        if self.normalize_spec and self.spec_scaler is not None:
            X_spec_norm = []
            for sample in X_spec:
                original_shape = sample.shape
                normalized = self.spec_scaler.transform(sample.flatten().reshape(-1, 1))
                X_spec_norm.append(normalized.flatten().reshape(original_shape))
            X_spec_norm = np.array(X_spec_norm, dtype=np.float32)
        else:
            X_spec_norm = X_spec
        
        # Expand dimensions
        X_raw_norm = np.expand_dims(X_raw_norm, axis=-1)
        X_spec_norm = np.expand_dims(X_spec_norm, axis=-1)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=4)
        
        return X_raw_norm, X_spec_norm, y_categorical


def create_data_augmentation_layer():
    """
    Create data augmentation layer for training robustness
    """
    return keras.Sequential([
        layers.GaussianNoise(0.01),  # Add small amount of noise
    ])


class ParallelCNNLSTMModel:
    """
    Parallel CNN-LSTM model for bearing fault diagnosis as described in model.md
    """
    
    def __init__(self, raw_input_shape: Tuple[int, int], 
                 spec_input_shape: Tuple[int, int, int], 
                 num_classes: int = 4,
                 dropout_rate: float = 0.3,
                 l2_reg: float = 1e-4):
        """
        Initialize the parallel model architecture
        
        Args:
            raw_input_shape: Shape of raw time-series input (timesteps, features)
            spec_input_shape: Shape of spectrogram input (height, width, channels)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
        """
        self.raw_input_shape = raw_input_shape
        self.spec_input_shape = spec_input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def build_lstm_branch(self, input_layer):
        """
        Build LSTM branch for temporal processing
        
        Args:
            input_layer: Input tensor for raw time-series data
            
        Returns:
            LSTM branch output tensor
        """
        # Data augmentation for training robustness
        x = layers.GaussianNoise(0.01)(input_layer)
        
        # Hierarchical convolution to progressively reduce sequence length
        # First reduction: 1,280,000 -> ~40,000
        x = layers.Conv1D(32, 64, strides=32, activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Second reduction: ~40,000 -> ~5,000
        x = layers.Conv1D(64, 16, strides=8, activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Third reduction: ~5,000 -> ~1,250
        x = layers.Conv1D(128, 8, strides=4, activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Fourth reduction: ~1,250 -> ~300 (manageable for LSTM)
        x = layers.Conv1D(256, 4, strides=4, activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Now apply LSTM to the manageable sequence length
        # Bidirectional LSTM Layer (with return sequences)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, 
                       kernel_regularizer=l2(self.l2_reg),
                       recurrent_regularizer=l2(self.l2_reg))
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # LSTM Layer (without return sequences) 
        x = layers.LSTM(64, return_sequences=False,
                       kernel_regularizer=l2(self.l2_reg),
                       recurrent_regularizer=l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense Layer (ReLU activation)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        
        return x
    
    def build_cnn_branch(self, input_layer):
        """
        Build CNN branch for spatial-frequency processing
        
        Args:
            input_layer: Input tensor for spectrogram data
            
        Returns:
            CNN branch output tensor
        """
        # Data augmentation
        x = layers.GaussianNoise(0.01)(input_layer)
        
        # Convolutional Block 1 - Handle wide spectrogram (freq_bins, time_bins_combined)
        # Use wider kernels in time dimension to capture temporal patterns
        x = layers.Conv2D(32, (7, 15), activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 4))(x)  # More aggressive pooling in time dimension
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Convolutional Block 2 - Continue processing
        x = layers.Conv2D(64, (5, 11), activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 4))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Convolutional Block 3 - Fine-grained features
        x = layers.Conv2D(128, (3, 7), activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 4))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Additional block to handle the large time dimension
        x = layers.Conv2D(256, (3, 5), activation='relu', padding='same',
                         kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((1, 4))(x)  # Only pool in time dimension
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense Layer (ReLU activation)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        
        return x
    
    def build_fusion_and_classification_head(self, lstm_output, cnn_output):
        """
        Build fusion layer and classification head
        
        Args:
            lstm_output: Output tensor from LSTM branch
            cnn_output: Output tensor from CNN branch
            
        Returns:
            Final output tensor with class probabilities
        """
        # Fusion Layer - Concatenation
        fused_features = layers.Concatenate()([lstm_output, cnn_output])
        
        # Classification Head
        x = layers.Dropout(self.dropout_rate)(fused_features)
        
        # Dense Layer 1 (ReLU activation)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense Layer 2 (ReLU activation)
        x = layers.Dense(128, activation='relu',
                        kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output Layer (Softmax activation)
        output = layers.Dense(self.num_classes, activation='softmax',
                             name='fault_classification')(x)
        
        return output
    
    def build_model(self) -> Model:
        """
        Build the complete parallel CNN-LSTM model
        
        Returns:
            Compiled Keras model
        """
        # Validate input shapes
        if len(self.raw_input_shape) != 2:
            raise ValueError(f"Raw input shape must be 2D (timesteps, features), got {self.raw_input_shape}")
        if len(self.spec_input_shape) != 3:
            raise ValueError(f"Spectrogram input shape must be 3D (height, width, channels), got {self.spec_input_shape}")
        
        # Define inputs
        raw_input = layers.Input(shape=self.raw_input_shape, name='raw_signal')
        spec_input = layers.Input(shape=self.spec_input_shape, name='spectrogram')
        
        # Build branches
        lstm_output = self.build_lstm_branch(raw_input)
        cnn_output = self.build_cnn_branch(spec_input)
        
        # Build fusion and classification
        final_output = self.build_fusion_and_classification_head(lstm_output, cnn_output)
        
        # Create model
        self.model = Model(
            inputs=[raw_input, spec_input],
            outputs=final_output,
            name='Parallel_CNN_LSTM_Model'
        )
        
        return self.model
    
    def compile_model(self, learning_rate: float = 1e-3):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Use Adam optimizer with better configuration for complex model
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,  # More stable for float32
            clipnorm=1.0   # Gradient clipping for stability
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        return self.model.summary()


class ModelTrainer:
    """
    Comprehensive training pipeline with robust regularization strategies
    """
    
    def __init__(self, model: Model, save_dir: str = "model_checkpoints"):
        """
        Initialize trainer
        
        Args:
            model: Compiled Keras model
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.history = None
        
    def get_callbacks(self, patience: int = 15, monitor: str = 'val_loss') -> List[callbacks.Callback]:
        """
        Get training callbacks for robust training
        
        Args:
            patience: Early stopping patience
            monitor: Metric to monitor for callbacks
            
        Returns:
            List of Keras callbacks
        """
        callback_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau - less aggressive for complex model
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.85,  # Less aggressive reduction
                patience=8,  # More patience for complex model
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                filepath=str(self.save_dir / 'best_model.h5'),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            
        ]
        
        return callback_list
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> keras.callbacks.History:
        """
        Train the model with robust training strategies
        
        Args:
            train_data: (X_raw_train, X_spec_train, y_train)
            val_data: (X_raw_val, X_spec_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        X_raw_train, X_spec_train, y_train = train_data
        X_raw_val, X_spec_val, y_val = val_data
        
        # Get callbacks
        callback_list = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            x=[X_raw_train, X_spec_train],
            y=y_train,
            validation_data=([X_raw_val, X_spec_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose,
            shuffle=True
        )
        
        return self.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Plot accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Plot precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Plot recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization utilities
    """
    
    def __init__(self, model: Model, class_names: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            class_names: List of class names for visualization
        """
        self.model = model
        self.class_names = class_names or ['Baseline', 'Inner Race', 'Outer Race', 'Hole']
        
    def evaluate_model(self, test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            test_data: (X_raw_test, X_spec_test, y_test)
            verbose: Whether to print results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        X_raw_test, X_spec_test, y_test = test_data
        
        # Get predictions
        y_pred_proba = self.model.predict([X_raw_test, X_spec_test])
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            [X_raw_test, X_spec_test], y_test, verbose=0
        )
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        if verbose:
            print("=== Model Evaluation Results ===")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print("\n=== Classification Report ===")
            print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return results
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            conf_matrix: Confusion matrix array
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       save_path: Optional[str] = None):
        """
        Plot ROC curves for multiclass classification
        
        Args:
            y_true: True labels (integer encoded)
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        # Convert to one-hot encoding for ROC calculation
        y_true_onehot = to_categorical(y_true, num_classes=len(self.class_names))
        
        plt.figure(figsize=(12, 8))
        
        # Calculate ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, 
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Each Class', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Plot prediction confidence distribution
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            # Get samples for this class
            class_mask = y_true == i
            class_probs = y_pred_proba[class_mask, i]
            
            # Plot histogram
            axes[i].hist(class_probs, bins=20, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[i].set_title(f'{class_name} Prediction Confidence', fontweight='bold')
            axes[i].set_xlabel('Predicted Probability')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_prob = np.mean(class_probs)
            std_prob = np.std(class_probs)
            axes[i].axvline(mean_prob, color='red', linestyle='--', 
                           label=f'Mean: {mean_prob:.3f}±{std_prob:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_complete_pipeline(data_dir: str = ".", sample_length: int = None,
                         epochs: int = 100, batch_size: int = 16,
                         test_size: float = 0.2, val_size: float = 0.2):
    """
    Run the complete training and evaluation pipeline
    
    Args:
        data_dir: Directory containing the dataset
        sample_length: Length of time-series samples
        epochs: Number of training epochs
        batch_size: Training batch size
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
    """
    print("=== Parallel CNN-LSTM Model for Bearing Fault Diagnosis ===\n")
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    dataset = BearingDataset(data_dir, sample_length=sample_length)
    dataset.load_all_data()
    
    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    train_data, val_data, test_data = dataset.get_data_splits(
        test_size=test_size, val_size=val_size
    )
    
    # Step 3: Preprocess data
    print("\nStep 3: Preprocessing data...")
    preprocessor = DataPreprocessor(normalize_raw=True, normalize_spec=True)
    
    # Preprocess training data (fit scalers)
    X_raw_train, X_spec_train, y_train = preprocessor.fit_transform_train(*train_data)
    
    # Preprocess validation and test data (transform only)
    X_raw_val, X_spec_val, y_val = preprocessor.transform(*val_data)
    X_raw_test, X_spec_test, y_test = preprocessor.transform(*test_data)
    
    print(f"Preprocessed data shapes:")
    print(f"  Raw train: {X_raw_train.shape}")
    print(f"  Spec train: {X_spec_train.shape}")
    print(f"  Labels train: {y_train.shape}")
    
    # Step 4: Build model
    print("\nStep 4: Building model...")
    model_builder = ParallelCNNLSTMModel(
        raw_input_shape=X_raw_train.shape[1:],
        spec_input_shape=X_spec_train.shape[1:],
        num_classes=4,
        dropout_rate=0.3,
        l2_reg=1e-4
    )
    
    model = model_builder.build_model()
    model_builder.compile_model(learning_rate=1e-3)
    
    print("Model architecture:")
    model_builder.get_model_summary()
    
    # Step 5: Train model
    print("\nStep 5: Training model...")
    trainer = ModelTrainer(model)
    
    history = trainer.train(
        train_data=(X_raw_train, X_spec_train, y_train),
        val_data=(X_raw_val, X_spec_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    # Step 6: Evaluate model
    print("\nStep 6: Evaluating model...")
    evaluator = ModelEvaluator(model)
    
    results = evaluator.evaluate_model(
        test_data=(X_raw_test, X_spec_test, y_test)
    )
    
    # Create visualizations
    evaluator.plot_confusion_matrix(results['confusion_matrix'], 'confusion_matrix.png')
    evaluator.plot_roc_curves(results['y_true'], results['y_pred_proba'], 'roc_curves.png')
    evaluator.plot_prediction_distribution(results['y_true'], 
                                          results['y_pred_proba'], 
                                          'prediction_distribution.png')
    
    print("\nPipeline completed successfully!")
    print("Generated files:")
    print("  - best_model.h5 (trained model)")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - roc_curves.png")
    print("  - prediction_distribution.png")
    
    return model, results


if __name__ == "__main__":
    # Run the complete pipeline using full signals - no data left unused
    model, results = run_complete_pipeline(
        data_dir=".",
        sample_length=None,  # Use complete signals - 100% data utilization
        epochs=50,  # Reduced for testing
        batch_size=10,   # Increased batch size for better gradient stability
        test_size=0.2,
        val_size=0.2
    )