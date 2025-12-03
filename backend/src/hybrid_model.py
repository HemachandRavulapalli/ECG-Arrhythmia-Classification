# hybrid_model.py
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Try to import tensorflow, but handle if missing
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, initializers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    # Create dummy objects for type hints
    layers = None
    models = None
    regularizers = None
    initializers = None

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
        """2D CNN with attention mechanism"""
        inputs = layers.Input(shape=(100, 10, 1))
        
        # Feature extraction
        x = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.GlobalAveragePooling2D()(x)
        attention = layers.Dense(256//4, activation='relu')(attention)
        attention = layers.Dense(256, activation='sigmoid')(attention)
        attention = layers.Reshape((1, 1, 256))(attention)
        x = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_multiscale_cnn_2d(self):
        """Multi-scale 2D CNN"""
        inputs = layers.Input(shape=(100, 10, 1))
        
        # Multi-scale feature extraction
        branches = []
        for kernel_size in [(3, 3), (5, 5), (7, 7)]:
            branch = layers.Conv2D(64, kernel_size, padding='same', activation='relu')(inputs)
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
        
        model = models.Model(inputs, outputs)
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
        
        # Train 2D CNN ensemble
        X_train_2d = X_train.reshape(-1, 100, 10, 1)
        X_val_2d = X_val.reshape(-1, 100, 10, 1)
        
        cnn2d_models = self.build_ensemble_cnn_2d()
        for name, model in cnn2d_models.items():
            print(f"Training {name}...")
            model.fit(
                X_train_2d, y_train,
                validation_data=(X_val_2d, y_val),
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
        
        # 2D CNN predictions
        X_test_2d = X_test.reshape(-1, 100, 10, 1)
        for name, model in self.models.items():
            if name.startswith('cnn2d_'):
                pred = model.predict(X_test_2d, verbose=0)
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
    
    def evaluate(self, X_test, y_test):
        """Evaluate the ensemble model"""
        predictions = self.predict_ensemble(X_test)
        if predictions is not None:
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            accuracy = accuracy_score(y_true, y_pred)
            print(f"🎯 Ensemble Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred))
            
            return accuracy, predictions
        else:
            print("❌ No predictions available")
            return 0, None
    
    def save_models(self, save_dir):
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name}.keras")
            model.save(model_path)
            print(f"💾 Saved {name} to {model_path}")
    
    def load_models(self, save_dir):
        """Load pre-trained models"""
        import os
        import glob
        
        model_files = glob.glob(os.path.join(save_dir, "*.keras"))
        for model_file in model_files:
            name = os.path.basename(model_file).replace('.keras', '')
            model = tf.keras.models.load_model(model_file)
            self.models[name] = model
            print(f"📂 Loaded {name} from {model_file}")


# ------------------------
# Hybrid Ensemble Class
# ------------------------
class HybridEnsemble:
    """Combine ML and DL models for improved predictions"""
    
    def __init__(self, ml_models=None, dl_models=None, classes=None, weights=None):
        self.ml_models = ml_models or {}
        self.dl_models = dl_models or {}
        self.classes = classes or []
        self.weights = weights or {}
    
    def predict_proba(self, X_ml, X_dl):
        """Get probability predictions from all models"""
        all_probs = []
        
        # ML model predictions
        for name, model in self.ml_models.items():
            try:
                prob = model.predict_proba(X_ml)
                all_probs.append(prob)
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {e}")
        
        # DL model predictions
        for name, model in self.dl_models.items():
            try:
                if name == "CNN2D" or "CNN2D" in name:
                    X_input = X_dl.reshape(-1, 100, 10, 1)
                else:
                    X_input = X_dl
                prob = model.predict(X_input, verbose=0)
                all_probs.append(prob)
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {e}")
        
        if not all_probs:
            raise ValueError("No valid predictions available")
        
        # Weighted averaging
        if self.weights:
            weights_list = [self.weights.get(name, 1.0) for name in list(self.ml_models.keys()) + list(self.dl_models.keys())]
            weights_array = np.array(weights_list[:len(all_probs)])
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones(len(all_probs)) / len(all_probs)
        
        # Ensemble prediction
        ensemble_prob = np.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, weights_array):
            ensemble_prob += weight * prob
        
        return ensemble_prob
    
    def evaluate(self, X_ml, X_dl, y_true):
        """Evaluate ensemble performance"""
        y_proba = self.predict_proba(X_ml, X_dl)
        y_pred = np.argmax(y_proba, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"🎯 Hybrid Ensemble Accuracy: {accuracy:.4f}")
        
        if self.classes:
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=self.classes))
        
        return accuracy