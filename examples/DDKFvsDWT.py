"""
ML Example: DDKF vs DWT Feature Extraction for Time Series Classification

This script demonstrates using DDKF for feature extraction in a time series
classification task using the aeon library. We compare:
1. DDKF-based features → CNN
2. DWT-based features → CNN

Dataset: Aeon univariate time series classification dataset

UPDATED: Uses corrected DDKF API with proper parameter names
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

print("=" * 80)
print("DDKF vs DWT for Time Series Classification")
print("(Using Corrected DDKF Implementation)")
print("=" * 80)

# Check dependencies
print("\nChecking dependencies...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    print("✓ PyTorch available")
except ImportError:
    print("✗ PyTorch not available. Install with: pip install torch")
    print("  Continuing with numpy-only implementation...")
    torch = None

try:
    from aeon.datasets import load_classification
    print("✓ Aeon available")
except ImportError:
    print("✗ Aeon not available. Install with: pip install aeon")
    print("  Will use synthetic data instead...")
    load_classification = None

try:
    import pywt
    print("✓ PyWavelets available")
except ImportError:
    print("✗ PyWavelets not available. Install with: pip install PyWavelets")
    pywt = None

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    print("✓ Scikit-learn available")
except ImportError:
    print("✗ Scikit-learn not available. Install with: pip install scikit-learn")
    raise

# Import DDKF - UPDATED to use corrected implementation
from ddkf import DDKFLayer

print("✓ DDKF available (corrected implementation)")


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_ddkf_features(
    time_series: np.ndarray,
    window_size: int = 20,
    step_size: int = 8,
    c_smoothing: float = 0.15,      # ← RENAMED from alpha
    c_smart_min: float = 0.85,      # ← RENAMED from beta
    kernel: List[str] = None
) -> np.ndarray:
    """Extract DDKF-based features from a time series.
    
    Returns the time-frequency representation as a 2D feature map.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    window_size : int
        DDKF window size
    step_size : int
        DDKF step size
    c_smoothing : float
        Beta threshold for final smoothing (was alpha)
    c_smart_min : float
        Alpha threshold for smart minimum (was beta)
    kernel : List[str]
        Kernel names to use
    
    Returns
    -------
    np.ndarray
        Time-frequency representation
    """
    if kernel is None:
        kernel = ["polynomial", "gaussian"]  # Hybrid kernel
    
    # Create DDKFLayer with CORRECTED parameter names
    layer = DDKFLayer(
        kernel_names=kernel,
        gamma=[0.5, 0.5],  # Equal weights for hybrid
        window_size=window_size,
        step_size=step_size,
        c_smoothing=c_smoothing,    # ← CORRECTED: was alpha
        c_smart_min=c_smart_min,    # ← CORRECTED: was beta
        interp_factor=1.5
    )

    # Optional: Freeze parameters if you don't want them to be learnable
    # (Uncomment if needed)
    # layer.c_smoothing.requires_grad = False  # ← CORRECTED: was alpha
    # layer.c_smart_min.requires_grad = False  # ← CORRECTED: was beta
    # layer._gamma.requires_grad = False

    try:
        # Convert to torch tensor
        signal_torch = torch.from_numpy(time_series).float()
        
        # Forward pass (no gradients needed for feature extraction)
        with torch.no_grad():
            tfr = layer(signal_torch)
        
        # Convert back to numpy
        tfr = tfr.numpy()
        
        # Normalize to [0, 1] range
        if tfr.max() > 0:
            tfr = tfr / tfr.max()
        
        return tfr
    except Exception as e:
        print(f"Warning: DDKF extraction failed: {e}")
        import traceback
        traceback.print_exc()
        # Return zero features
        return np.zeros((10, 10))


def extract_dwt_features(
    time_series: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> np.ndarray:
    """Extract DWT-based features from a time series.
    
    Returns multi-level wavelet coefficients as a 2D feature map.
    """
    if pywt is None:
        # Fallback: simple binning
        n_bins = 10
        features = []
        bin_size = len(time_series) // n_bins
        for i in range(n_bins):
            start = i * bin_size
            end = min(start + bin_size, len(time_series))
            if end > start:
                features.append([
                    np.mean(time_series[start:end]),
                    np.std(time_series[start:end]),
                    np.max(time_series[start:end]),
                    np.min(time_series[start:end])
                ])
        return np.array(features)
    
    try:
        # Perform multilevel DWT
        coeffs = pywt.wavedec(time_series, wavelet, level=level)
        
        # Organize coefficients into a 2D matrix
        max_len = max(len(c) for c in coeffs)
        feature_matrix = np.zeros((len(coeffs), max_len))
        
        for i, coeff in enumerate(coeffs):
            feature_matrix[i, :len(coeff)] = coeff
        
        # Normalize
        if feature_matrix.max() > 0:
            feature_matrix = feature_matrix / np.abs(feature_matrix).max()
        
        return feature_matrix
    except Exception as e:
        print(f"Warning: DWT extraction failed: {e}")
        return np.zeros((4, 10))


def pad_or_crop_features(features: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Pad or crop features to match target shape."""
    h, w = features.shape
    th, tw = target_shape
    
    # Create output array
    output = np.zeros(target_shape)
    
    # Copy data
    h_copy = min(h, th)
    w_copy = min(w, tw)
    output[:h_copy, :w_copy] = features[:h_copy, :w_copy]
    
    return output


# =============================================================================
# CNN Classifier
# =============================================================================

if torch is not None:
    class SimpleCNN(nn.Module):
        """Simple CNN for 2D feature classification."""
        
        def __init__(self, input_shape: Tuple[int, int], num_classes: int):
            super(SimpleCNN, self).__init__()
            
            h, w = input_shape
            
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(2, 2)
            
            # Calculate flattened size
            h_out = h // 4
            w_out = w // 4
            self.flat_size = 32 * h_out * w_out
            
            self.fc1 = nn.Linear(self.flat_size, 64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, num_classes)
        
        def forward(self, x):
            # x shape: (batch, 1, height, width)
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, self.flat_size)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001
) -> Tuple[nn.Module, dict]:
    """Train CNN classifier on 2D features."""
    
    if torch is None:
        print("PyTorch not available, skipping CNN training")
        return None, {"train_acc": [], "test_acc": []}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(1)
    y_test_t = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_shape = X_train.shape[1:]
    model = SimpleCNN(input_shape, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {"train_acc": [], "test_acc": []}
    
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Test accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t.to(device))
            _, predicted = torch.max(outputs.data, 1)
            test_acc = 100 * (predicted.cpu() == y_test_t).sum().item() / len(y_test)
        
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    return model, history


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment():
    """Run the full DDKF vs DWT comparison experiment."""
    
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)
    
    # Load dataset
    if load_classification is not None:
        try:
            # Try to load a small dataset
            print("Attempting to load ECG200 dataset...")
            X_train, y_train = load_classification("ECG200", split="train")
            X_test, y_test = load_classification("ECG200", split="test")
            
            # Convert to numpy if needed
            if hasattr(X_train, 'to_numpy'):
                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()
            
            # If 3D, take first dimension
            if len(X_train.shape) == 3:
                X_train = X_train.squeeze()
                X_test = X_test.squeeze()
            
            print(f"✓ Loaded ECG200 dataset")
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
            
        except Exception as e:
            print(f"Could not load ECG200: {e}")
            print("Generating synthetic dataset...")
            X_train, y_train, X_test, y_test = generate_synthetic_dataset()
    else:
        print("Generating synthetic dataset...")
        X_train, y_train, X_test, y_test = generate_synthetic_dataset()
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution (train): {np.bincount(y_train_encoded)}")
    
    # =============================================================================
    # Extract DDKF Features
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("Extracting DDKF Features (Corrected Implementation)")
    print("=" * 80)
    
    ddkf_features_train = []
    for i, ts in enumerate(X_train):
        if (i + 1) % 20 == 0:
            print(f"Processing {i+1}/{len(X_train)}...", end='\r')
        features = extract_ddkf_features(
            ts,
            c_smoothing=0.15,  # ← CORRECTED parameter name
            c_smart_min=0.85   # ← CORRECTED parameter name
        )
        ddkf_features_train.append(features)
    print(f"Processed {len(X_train)} training samples" + " " * 20)
    
    ddkf_features_test = []
    for i, ts in enumerate(X_test):
        features = extract_ddkf_features(
            ts,
            c_smoothing=0.15,  # ← CORRECTED parameter name
            c_smart_min=0.85   # ← CORRECTED parameter name
        )
        ddkf_features_test.append(features)
    print(f"Processed {len(X_test)} test samples")
    
    # Determine target shape (use median dimensions)
    shapes = [f.shape for f in ddkf_features_train]
    target_h = int(np.median([s[0] for s in shapes]))
    target_w = int(np.median([s[1] for s in shapes]))
    target_shape_ddkf = (target_h, target_w)
    
    print(f"Target DDKF feature shape: {target_shape_ddkf}")
    
    # Pad/crop to uniform size
    X_train_ddkf = np.array([
        pad_or_crop_features(f, target_shape_ddkf) for f in ddkf_features_train
    ])
    X_test_ddkf = np.array([
        pad_or_crop_features(f, target_shape_ddkf) for f in ddkf_features_test
    ])
    
    print(f"DDKF features shape: {X_train_ddkf.shape}")
    
    # =============================================================================
    # Extract DWT Features
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("Extracting DWT Features")
    print("=" * 80)
    
    dwt_features_train = []
    for i, ts in enumerate(X_train):
        if (i + 1) % 20 == 0:
            print(f"Processing {i+1}/{len(X_train)}...", end='\r')
        features = extract_dwt_features(ts)
        dwt_features_train.append(features)
    print(f"Processed {len(X_train)} training samples" + " " * 20)
    
    dwt_features_test = []
    for i, ts in enumerate(X_test):
        features = extract_dwt_features(ts)
        dwt_features_test.append(features)
    print(f"Processed {len(X_test)} test samples")
    
    # Determine target shape
    shapes_dwt = [f.shape for f in dwt_features_train]
    target_h_dwt = int(np.median([s[0] for s in shapes_dwt]))
    target_w_dwt = int(np.median([s[1] for s in shapes_dwt]))
    target_shape_dwt = (target_h_dwt, target_w_dwt)
    
    print(f"Target DWT feature shape: {target_shape_dwt}")
    
    # Pad/crop to uniform size
    X_train_dwt = np.array([
        pad_or_crop_features(f, target_shape_dwt) for f in dwt_features_train
    ])
    X_test_dwt = np.array([
        pad_or_crop_features(f, target_shape_dwt) for f in dwt_features_test
    ])
    
    print(f"DWT features shape: {X_train_dwt.shape}")
    
    # =============================================================================
    # Train CNN on DDKF Features
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("Training CNN on DDKF Features")
    print("=" * 80)
    
    if torch is not None:
        model_ddkf, history_ddkf = train_cnn(
            X_train_ddkf, y_train_encoded,
            X_test_ddkf, y_test_encoded,
            num_classes=num_classes,
            epochs=30,
            batch_size=16,
            lr=0.001
        )
        
        final_test_acc_ddkf = history_ddkf["test_acc"][-1]
        print(f"\nFinal DDKF Test Accuracy: {final_test_acc_ddkf:.2f}%")
    else:
        history_ddkf = None
        final_test_acc_ddkf = 0
    
    # =============================================================================
    # Train CNN on DWT Features
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("Training CNN on DWT Features")
    print("=" * 80)
    
    if torch is not None:
        model_dwt, history_dwt = train_cnn(
            X_train_dwt, y_train_encoded,
            X_test_dwt, y_test_encoded,
            num_classes=num_classes,
            epochs=30,
            batch_size=16,
            lr=0.001
        )
        
        final_test_acc_dwt = history_dwt["test_acc"][-1]
        print(f"\nFinal DWT Test Accuracy: {final_test_acc_dwt:.2f}%")
    else:
        history_dwt = None
        final_test_acc_dwt = 0
    
    # =============================================================================
    # Visualize Results
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Example time series and features
    idx = 0
    
    plt.subplot(3, 3, 1)
    plt.plot(X_train[idx])
    plt.title(f'Example Time Series (Class {y_train[idx]})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 3, 2)
    plt.imshow(X_train_ddkf[idx], aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title('DDKF Features (Corrected)')
    plt.xlabel('Frequency')
    plt.ylabel('Time Window')
    
    plt.subplot(3, 3, 3)
    plt.imshow(X_train_dwt[idx], aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('DWT Features')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Decomposition Level')
    
    # Plot 2-3: More examples
    for i, idx in enumerate([5, 10]):
        plt.subplot(3, 3, 4 + i*3)
        plt.plot(X_train[idx])
        plt.title(f'Time Series (Class {y_train[idx]})')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 5 + i*3)
        plt.imshow(X_train_ddkf[idx], aspect='auto', cmap='hot')
        plt.title('DDKF Features')
        
        plt.subplot(3, 3, 6 + i*3)
        plt.imshow(X_train_dwt[idx], aspect='auto', cmap='viridis')
        plt.title('DWT Features')
    
    plt.suptitle('Feature Extraction Comparison (Corrected DDKF)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('ml_features_comparison_corrected.png', dpi=150, bbox_inches='tight')
    print("✓ Saved ml_features_comparison_corrected.png")
    
    # Plot training curves
    if history_ddkf is not None and history_dwt is not None:
        fig2 = plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_ddkf["train_acc"], 'b-', label='DDKF Train', linewidth=2)
        plt.plot(history_ddkf["test_acc"], 'b--', label='DDKF Test', linewidth=2)
        plt.plot(history_dwt["train_acc"], 'r-', label='DWT Train', linewidth=2)
        plt.plot(history_dwt["test_acc"], 'r--', label='DWT Test', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Curves (Corrected DDKF)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        methods = ['DDKF\n(Corrected)', 'DWT']
        accuracies = [final_test_acc_ddkf, final_test_acc_dwt]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = plt.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        plt.ylabel('Test Accuracy (%)')
        plt.title('Final Test Accuracy Comparison')
        plt.ylim([0, 100])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('ml_training_results_corrected.png', dpi=150, bbox_inches='tight')
        print("✓ Saved ml_training_results_corrected.png")
    
    # =============================================================================
    # Summary
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Using Corrected DDKF Implementation)")
    print("=" * 80)
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test")
    print(f"Number of classes: {num_classes}")
    print(f"\nDDKF Features (Corrected):")
    print(f"  Shape: {X_train_ddkf.shape}")
    print(f"  Test Accuracy: {final_test_acc_ddkf:.2f}%")
    print(f"\nDWT Features:")
    print(f"  Shape: {X_train_dwt.shape}")
    print(f"  Test Accuracy: {final_test_acc_dwt:.2f}%")
    
    if final_test_acc_ddkf > final_test_acc_dwt:
        improvement = final_test_acc_ddkf - final_test_acc_dwt
        print(f"\n✓ DDKF outperforms DWT by {improvement:.2f}%")
    elif final_test_acc_dwt > final_test_acc_ddkf:
        improvement = final_test_acc_dwt - final_test_acc_ddkf
        print(f"\n✓ DWT outperforms DDKF by {improvement:.2f}%")
    else:
        print(f"\n= DDKF and DWT perform equally")
    
    print("=" * 80)
    
    plt.show()


def generate_synthetic_dataset(n_train=100, n_test=30, n_classes=2):
    """Generate synthetic time series dataset."""
    print("Generating synthetic dataset...")
    
    def generate_class_signal(class_id, length=150):
        t = np.linspace(0, 10, length)
        if class_id == 0:
            # Low frequency signal
            signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
        else:
            # High frequency signal
            signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 8 * t)
        
        # Add noise
        signal += 0.1 * np.random.randn(length)
        return signal
    
    # Generate training data
    X_train = []
    y_train = []
    for i in range(n_train):
        class_id = i % n_classes
        X_train.append(generate_class_signal(class_id))
        y_train.append(class_id)
    
    # Generate test data
    X_test = []
    y_test = []
    for i in range(n_test):
        class_id = i % n_classes
        X_test.append(generate_class_signal(class_id))
        y_test.append(class_id)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


if __name__ == "__main__":
    run_experiment()
    print("\nExperiment complete!")
