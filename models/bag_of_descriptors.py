from sklearn.cluster import KMeans
import numpy as np
import cv2


class BagOfDescriptors:
    
    def __init__(self, clusters=5, regression_model = None):
        self.clusters = clusters
        self.kmeans = None
        self.weights = None
        self.model = regression_model
        pass
    
    def fit(self, X, y):
        # Get SIFT descriptors for all training images
        descriptors_list = self.get_sift_descriptors(X)
        
        # Filter out images without descriptors and corresponding labels
        valid_indices = [i for i, d in enumerate(descriptors_list) if d is not None]
        descriptors_list = [descriptors_list[i] for i in valid_indices]
        y = [y[i] for i in valid_indices]       
             
        # Train KMeans clustering and extract features
        self.kmeans = self.train_kmeans(descriptors_list, max_per_sample=12, num_clusters= self.clusters)
        
        # Convert the descriptors to a normalized bag of words histogram
        features = self.get_features(descriptors_list)
        print(f"Features extracted for {len(features)} images")

        # Check if a model is provided
        if self.model is not None:
            self.model.fit(features, y)
        else:
            raise ValueError("No regression model provided")

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not trained")
        
        test_features = self.get_features(self.get_sift_descriptors(X_test))

        return self.model.predict_proba(test_features)
        
    def get_sift_descriptors(self, X):
        """
        Extract SIFT descriptors from the images in X.
        
        Parameters
        ----------
        X : numpy.ndarray
            Array of images, with shape (n_images, n_pixels_x, n_pixels_y, n_channels).
        
        Returns
        -------
        descriptors : numpy.ndarray
            Array of SIFT descriptors, with shape (n_descriptors, n_features).
        """
        # Initialize SIFT
        sift = cv2.SIFT_create()
        
        # Initialize array of descriptors
        descriptors_list = []
        
        # Iterate through all images to extract descriptors
        for image in X:
            _, descriptor = sift.detectAndCompute(image, None)
                            
            if  descriptor is not None:
                descriptor = descriptor[:100]
                descriptors_list.append(descriptor)
            else:
                #Some images do not have any descriptors, they can't be classified
                # Mark as None so we can remove them later
                descriptors_list.append(None) 

        return descriptors_list

    def train_kmeans(self, descriptors_list, max_per_sample, num_clusters):  
        # Train the KMeans clustering model to quantize the SIFT descriptors 
        
        # Sample descriptors from each image to train the KMeans model
        sampled_descriptors_list = []             
       
        for descriptors in descriptors_list:
            num_descriptors = len(descriptors)
            if num_descriptors > max_per_sample:
                # Randomly sample descriptors if there are more than the limit
                sampled_indices = np.random.choice(num_descriptors, max_per_sample, replace=False)
                sampled_descriptors = descriptors[sampled_indices]
            else:
                # Use all descriptors if less than the limit
                sampled_descriptors = descriptors
                
            sampled_descriptors_list.append(sampled_descriptors)
        
        # Flatten the list of descriptors to feed to kmeans
        all_descriptors = np.vstack(sampled_descriptors_list)
        print(f"Training on {len(all_descriptors)} descriptors")
        
        # Train a kmeans model with num_clusters clusters
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', max_iter=1000, random_state=0)
        kmeans.fit(all_descriptors)     
       
        print(f"KMeans trained with {num_clusters} clusters")         
        return kmeans 

    
    def get_features(self, descriptors_list):
        """
        Convert the descriptors to a normalized bag of words histogram.
        Use kmeans for quantization.
        
        Parameters
        ----------
        kmeans : sklearn.cluster.KMeans
            The trained KMeans clustering model.
        descriptors : numpy.ndarray
            Array of SIFT descriptors, with shape (n_descriptors, 128).
            
        Returns
        -------
        bow_features : numpy.ndarray
            Normalized bag of words histogram, with shape (1, n_clusters).    
        """
        
        # Initialize the bag of words histogram
        bow_features = np.zeros((len(descriptors_list), self.kmeans.n_clusters))

        # Iterate over the descriptors
        for i, d in enumerate(descriptors_list):
            if d is not None and len(d) > 0:
                # Predict the label for the descriptor
                labels = self.kmeans.predict(d)
                
                for label in labels:
                    # Increment the count in the histogram
                    bow_features[i, label] += 1

                # Normalize this histogram
                bow_features[i, :] /= len(d)
            # Else, leave the feature as zero vector if no descriptors are found

        return bow_features