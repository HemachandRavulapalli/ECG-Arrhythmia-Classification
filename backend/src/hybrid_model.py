# hybrid_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


class AdvancedHybridModel:
    """
    Advanced hybrid model combining multiple CNN architectures with ensemble methods
    to achieve 99%+ accuracy on ECG classification
    """
    
    def __init__(self, input_shape, num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        self.ensemble_weights = {}
        
    def build_ensemble_cnn_1d(self):
        """Build ensemble of 1D CNN models with different architectures"""
        models = {}
        
        # Model 1: Deep Residual CNN
        models['residual_cnn'] = self._build_residual_cnn_1d()
        
        # Model 2: DenseNet-like CNN
        models['densenet_cnn'] = self._build_densenet_cnn_1d()
        
        # Model 3: Attention-based CNN
        models['attention_cnn'] = self._build_attention_cnn_1d()
        
        # Model 4: Multi-scale CNN
        models['multiscale_cnn'] = self._build_multiscale_cnn_1d()
        
        return models
    
    def _build_residual_cnn_1d(self):
        """Deep residual CNN with skip connections"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv1D(64, 15, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Residual blocks
        for i, filters in enumerate([64, 128, 256, 512]):
            residual = x
            x = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Skip connection
            if residual.shape[-1] != filters:
                residual = layers.Conv1D(filters, 1, padding='same')(residual)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(0.2)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_densenet_cnn_1d(self):
        """DenseNet-like CNN with dense connections"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Dense blocks
        for block in range(4):
            for layer in range(4):
                # Dense connection
                y = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
                y = layers.BatchNormalization()(y)
                y = layers.Conv1D(32, 3, padding='same', activation='relu')(y)
                y = layers.BatchNormalization()(y)
                x = layers.Concatenate()([x, y])
            
            # Transition layer
            x = layers.Conv1D(x.shape[-1] // 2, 1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(0.2)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_attention_cnn_1d(self):
        """CNN with self-attention mechanism"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Feature extraction
        x = layers.Conv1D(64, 15, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, 11, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(256, 7, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Self-attention mechanism
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(1, activation='softmax')(attention)
        attention = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling1D()(attention)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_multiscale_cnn_1d(self):
        """Multi-scale CNN with different kernel sizes"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Multi-scale feature extraction
        branches = []
        for kernel_size in [5, 11, 21, 41]:
            branch = layers.Conv1D(64, kernel_size, padding='same', activation='relu')(inputs)
            branch = layers.BatchNormalization()(branch)
            branch = layers.MaxPooling1D(2)(branch)
            branch = layers.Conv1D(128, kernel_size//2, padding='same', activation='relu')(branch)
            branch = layers.BatchNormalization()(branch)
            branch = layers.MaxPooling1D(2)(branch)
            branch = layers.GlobalAveragePooling1D()(branch)
            branches.append(branch)
        
        # Concatenate multi-scale features
        x = layers.Concatenate()(branches)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_ensemble_cnn_2d(self):
        """Build ensemble of 2D CNN models"""
        models = {}
        
        # Model 1: Deep 2D CNN with attention
        models['attention_2d'] = self._build_attention_cnn_2d()
        
        # Model 2: Multi-scale 2D CNN
        models['multiscale_2d'] = self._build_multiscale_cnn_2d()
        
        return models
    
    def _build_attention_cnn_2d(self):
        """2D CNN with attention mechanism - FIXED"""
        # Start from the same 1D input shape
        base_inputs = layers.Input(shape=self.input_shape)   # (1000, 1)
        
        # Reshape to 2D representation, consistent with train_ensemble
        x = layers.Reshape((100, 10, 1))(base_inputs)        # (100, 10, 1)
        
        # Feature extraction
        x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism (channel-wise)
        attention = layers.GlobalAveragePooling2D()(x)
        attention = layers.Dense(256 // 4, activation='relu')(attention)
        attention = layers.Dense(256, activation='sigmoid')(attention)
        attention = layers.Reshape((1, 1, 256))(attention)
        x = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(base_inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_multiscale_cnn_2d(self):
        """Multi-scale 2D CNN - FIXED"""
        # Start from the same 1D input shape
        base_inputs = layers.Input(shape=self.input_shape)   # (1000, 1)
        
        # Reshape to 2D representation, consistent with train_ensemble
        x_in = layers.Reshape((100, 10, 1))(base_inputs)     # (100, 10, 1)
        
        # Multi-scale feature extraction
        branches = []
        for kernel_size in [(3, 3), (5, 5), (7, 7)]:
            branch = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(x_in)
            branch = layers.BatchNormalization()(branch)
            branch = layers.MaxPooling2D((2, 2))(branch)
            branch = layers.Conv2D(128, kernel_size, padding='same', activation='relu')(branch)
            branch = layers.BatchNormalization()(branch)
            branch = layers.MaxPooling2D((2, 2))(branch)
            branch = layers.GlobalAveragePooling2D()(branch)
            branches.append(branch)
        
        # Concatenate multi-scale features
        x = layers.Concatenate()(branches)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(base_inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the ensemble of models"""
        print("🚀 Training ensemble of CNN models...")
        
        # Train 1D CNN ensemble
        cnn1d_models = self.build_ensemble_cnn_1d()
        for name, model in cnn1d_models.items():
            print(f"Training {name}...")
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            self.models[f'cnn1d_{name}'] = model
        
        # Train 2D CNN ensemble - NOW WORKS WITH SAME X_train!
        cnn2d_models = self.build_ensemble_cnn_2d()
        for name, model in cnn2d_models.items():
            print(f"Training {name}...")
            model.fit(
                X_train, y_train,  # No reshape needed anymore!
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            self.models[f'cnn2d_{name}'] = model
    
    def predict_ensemble(self, X_test):
        """Make predictions using the ensemble"""
        predictions = []
        
        # 1D CNN predictions
        for name, model in self.models.items():
            if name.startswith('cnn1d_'):
                pred = model.predict(X_test, verbose=0)
                predictions.append(pred)
        
        # 2D CNN predictions - NOW WORKS WITH SAME X_test!
        for name, model in self.models.items():
            if name.startswith('cnn2d_'):
                pred = model.predict(X_test, verbose=0)  # No reshape needed!
                predictions.append(pred)
        
        # Weighted ensemble prediction
        if predictions:
            # Equal weights for now, can be optimized
            weights = np.ones(len(predictions)) / len(predictions)
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, weights):
                ensemble_pred += weight * pred
            return ensemble_pred
        else:
            return None


class HybridEnsemble:
    """
    Traditional hybrid ensemble combining ML and DL models
    """
    def __init__(self, ml_models, dl_models, classes, weights=None):
        self.ml_models = ml_models or {}
        self.dl_models = dl_models or {}
        self.classes = classes
        self.weights = weights or {}
        # unified view for convenience
        self.models = {**self.ml_models, **self.dl_models}
    
    def predict_proba(self, X_ml, X_dl):
        """Get ensemble predictions from ML and DL models"""
        predictions = []
        weights_list = []
        
        # ML model predictions
        for name, model in self.ml_models.items():
            try:
                proba = model.predict_proba(X_ml)
                predictions.append(proba)
                weights_list.append(self.weights.get(name, 1.0))
            except Exception as e:
                print(f"⚠️ Error predicting with {name}: {e}")
        
        # DL model predictions
        for name, model in self.dl_models.items():
            try:
                proba = model.predict(X_dl, verbose=0)
                predictions.append(proba)
                weights_list.append(self.weights.get(name, 1.0))
            except Exception as e:
                print(f"⚠️ Error predicting with {name}: {e}")
        
        if not predictions:
            raise ValueError("❌ No models available for prediction")
        
        # Weighted average
        total_weight = sum(weights_list)
        if total_weight == 0:
            weights_list = [1.0] * len(predictions)
            total_weight = len(predictions)
        
        ensemble_proba = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights_list):
            # Ensure same number of classes
            if pred.shape[1] != ensemble_proba.shape[1]:
                # Truncate or pad
                min_classes = min(pred.shape[1], ensemble_proba.shape[1])
                ensemble_proba[:, :min_classes] += (weight / total_weight) * pred[:, :min_classes]
            else:
                ensemble_proba += (weight / total_weight) * pred
        
        return ensemble_proba
    
    def evaluate(self, X_ml, X_dl, y_true):
        """Evaluate the ensemble model on aligned ML/DL inputs"""
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        probs = self.predict_proba(X_ml, X_dl)
        y_pred = np.argmax(probs, axis=1)
            
        accuracy = accuracy_score(y_true, y_pred)
        print(f"🎯 Ensemble Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
            
        return accuracy, probs
    
    def save_models(self, save_dir):
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save DL models
        for name, model in self.dl_models.items():
            model_path = os.path.join(save_dir, f"{name}.keras")
            model.save(model_path)
            print(f"💾 Saved DL model {name} to {model_path}")

        # Save ML models
        for name, model in self.ml_models.items():
            model_path = os.path.join(save_dir, f"{name}.joblib")
            try:
                joblib.dump(model, model_path)
                print(f"💾 Saved ML model {name} to {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to save {name}: {e}")
    
    def load_models(self, save_dir):
        """Load pre-trained models"""
        import os
        import glob
        
        # Load DL models
        model_files = glob.glob(os.path.join(save_dir, "*.keras"))
        for model_file in model_files:
            name = os.path.basename(model_file).replace('.keras', '')
            model = tf.keras.models.load_model(model_file)
            self.dl_models[name] = model
            print(f"📂 Loaded DL model {name} from {model_file}")

        # Load ML models
        joblib_files = glob.glob(os.path.join(save_dir, "*.joblib"))
        for joblib_file in joblib_files:
            name = os.path.basename(joblib_file).replace('.joblib', '')
            try:
                model = joblib.load(joblib_file)
                self.ml_models[name] = model
                print(f"📂 Loaded ML model {name} from {joblib_file}")
            except Exception as e:
                print(f"⚠️ Failed to load {joblib_file}: {e}")

        # refresh unified view
        self.models = {**self.ml_models, **self.dl_models}
