import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# Update imports to match your new structure
from utils.data_loader import load_fashion_mnist
from utils.cnn_model import prepare_data_for_cnn, build_cnn_model
from utils.data_augmentation import create_data_augmentation_generator

class TestDataLoading:
    # Update patch paths to match your new module structure
    @patch('utils.data_loader.pickle.load')
    @patch('utils.data_loader.open')
    @patch('utils.data_loader.os.path.exists')
    def test_load_fashion_mnist_from_cache(self, mock_exists, mock_open, mock_pickle_load):
        # Setup mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock data
        x_train = np.random.rand(1000, 28, 28)
        y_train = np.random.randint(0, 10, 1000)
        x_test = np.random.rand(200, 28, 28)
        y_test = np.random.randint(0, 10, 200)
        mock_pickle_load.return_value = (x_train, y_train, x_test, y_test)

        # Call function
        with patch('utils.data_loader.train_test_split', return_value=(x_train[:800], x_train[800:], y_train[:800], y_train[800:])):
            result = load_fashion_mnist(cache_dir='./test_data')

        # Assertions
        assert mock_exists.called
        assert mock_open.called
        assert len(result) == 7  # Returns 7 items
        assert result[0].shape == (800, 28, 28)  # x_train
        assert result[2].shape == (200, 28, 28)  # x_val
        assert result[4].shape == (200, 28, 28)  # x_test

    def test_prepare_data_for_cnn(self):
        # Create test data
        x_train = np.random.rand(100, 28, 28)
        x_val = np.random.rand(20, 28, 28)
        x_test = np.random.rand(30, 28, 28)

        # Call function
        x_train_prep, x_val_prep, x_test_prep = prepare_data_for_cnn(x_train, x_val, x_test)

        # Assertions
        assert x_train_prep.shape == (100, 28, 28, 1)
        assert x_val_prep.shape == (20, 28, 28, 1)
        assert x_test_prep.shape == (30, 28, 28, 1)


class TestModelArchitecture:
    def test_build_cnn_model(self):
        # Build model
        model = build_cnn_model()

        # Assertions
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 10)

        # Check layer types (at least one of each important type)
        layer_types = [layer.__class__.__name__ for layer in model.layers]
        assert 'Conv2D' in layer_types
        assert 'MaxPooling2D' in layer_types
        assert 'Dropout' in layer_types
        assert 'Dense' in layer_types
        assert 'BatchNormalization' in layer_types

        # Test dummy forward pass
        test_input = np.random.rand(1, 28, 28, 1)
        output = model.predict(test_input)
        assert output.shape == (1, 10)
        assert np.isclose(np.sum(output[0]), 1.0, atol=1e-5)  # Softmax output sums to 1


class TestDataAugmentation:
    def test_create_data_augmentation_generator(self):
        # Create generator
        datagen = create_data_augmentation_generator(
            rotation_range=15,
            width_shift_range=0.2
        )

        # Assertions
        assert datagen.rotation_range == 15
        assert datagen.width_shift_range == 0.2
        assert datagen.horizontal_flip == True


class TestImageProcessing:
    def test_preprocess_image(self):
        # Create a test image
        test_img_path = 'test_image.png'
        img = Image.new('RGB', (50, 50), color='white')

        # Draw a simple shape to simulate a clothing item
        for x in range(15, 35):
            for y in range(15, 35):
                img.putpixel((x, y), (0, 0, 0))

        try:
            img.save(test_img_path)

            # Fix the import to match your new module structure
            from tkinter_example import preprocess_image
            img_array, _ = preprocess_image(test_img_path)

            # Assertions
            assert img_array is not None
            assert img_array.shape == (1, 28, 28, 1)
            # Update to accept either float32 or float64
            assert img_array.dtype in [np.float32, np.float64]
            assert np.max(img_array) <= 1.0
            assert np.min(img_array) >= 0.0

        finally:
            # Clean up
            if os.path.exists(test_img_path):
                os.remove(test_img_path)


if __name__ == '__main__':
    pytest.main(['-xvs', 'test_functions.py'])