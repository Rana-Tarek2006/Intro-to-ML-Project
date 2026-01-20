import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import numpy as np

class FeatureExtractor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.feature_extractors = {}
        
    def extract_handcrafted_features(self, image_path):
        """
        Extract traditional handcrafted features
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_size)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. Histogram features
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        features['hist_mean'] = np.mean(hist)
        features['hist_std'] = np.std(hist)
        
        # 2. Edge features (Canny)
        edges = cv2.Canny(img_gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 3. Texture features (GLCM-like)
        from skimage.feature import graycomatrix, graycoprops
        # Convert to uint8 for GLCM
        img_uint8 = (img_gray / img_gray.max() * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
        features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        features['energy'] = graycoprops(glcm, 'energy')[0, 0]
        
        # 4. Color moments (if color image)
        if len(img.shape) == 3:
            for i in range(3):
                channel = img[:, :, i]
                features[f'channel_{i}_mean'] = np.mean(channel)
                features[f'channel_{i}_std'] = np.std(channel)
        
        return np.array(list(features.values()))
    
    def extract_cnn_features(self, image_path, model_name='vgg16'):
        """
        Extract deep features using pre-trained CNN models
        """
        if model_name not in self.feature_extractors:
            if model_name == 'vgg16':
                base_model = VGG16(weights='imagenet', include_top=False, 
                                  input_shape=(*self.img_size, 3))
            elif model_name == 'resnet50':
                base_model = ResNet50(weights='imagenet', include_top=False,
                                     input_shape=(*self.img_size, 3))
            elif model_name == 'efficientnet':
                base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                           input_shape=(*self.img_size, 3))
            
            # Create feature extraction model
            model = Model(inputs=base_model.input, 
                         outputs=base_model.output)
            model = Model(inputs=model.input, 
                         outputs=tf.keras.layers.GlobalAveragePooling2D()(model.output))
            self.feature_extractors[model_name] = model
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize
        
        if model_name == 'vgg16':
            # VGG16 preprocessing
            img = tf.keras.applications.vgg16.preprocess_input(img)
        elif model_name == 'resnet50':
            # ResNet50 preprocessing
            img = tf.keras.applications.resnet50.preprocess_input(img)
        
        # Extract features
        features = self.feature_extractors[model_name].predict(np.expand_dims(img, axis=0))
        return features.flatten()
