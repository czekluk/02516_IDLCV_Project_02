import numpy as np
import random

class WeakLabelsCreator:
    
    def __init__(self, num_points_per_label=10, strategy="random"):
        self.num_points = num_points_per_label
        strategy_map = {
            "random": self.random_sample,
            "extreme_clicks": self.extreme_clicks,
            "central_clicks": self.central_clicks
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"strategy must be one of {strategy_map.keys()}, but got '{strategy}' instead")
        
        self.create_points = strategy_map[strategy]
    
    def random_sample(self, binary_mask):
        point_mask = np.full(binary_mask.shape, -1, dtype=np.float32)
        
        foreground_points = np.argwhere(binary_mask == 1)
        selected_foreground_points = self.sample_points(foreground_points, self.num_points)
        
        background_points = np.argwhere(binary_mask == 0)
        selected_background_points = self.sample_points(background_points, self.num_points)
        
        for point in selected_foreground_points:
            point_mask[tuple(point)] = 1
        for point in selected_background_points:
            point_mask[tuple(point)] = 0
        
        return point_mask

    def sample_points(self, points, num_samples, probabilities=None):
        """Helper function to sample a subset of points with optional probability distribution."""
        if len(points) == 0:
            return []
        if probabilities is None:
            probabilities = np.ones(len(points))  # Uniform distribution if not provided
        probabilities /= np.sum(probabilities)
        selected_indices = np.random.choice(len(points), num_samples, p=probabilities, replace=False)
        return points[selected_indices]

    def extreme_clicks(self, binary_mask):
        """
        Find extreme points (left-most, right-most, top-most, and bottom-most)
        for the lesion (foreground = 1), then sample the remaining points randomly.
        """
        point_mask = np.full(binary_mask.shape, -1, dtype=np.float32)
        
        foreground_points = np.argwhere(binary_mask == 1)
        
        if len(foreground_points) == 0:
            return point_mask
        
        left_most = foreground_points[np.argmin(foreground_points[:, 1])] 
        right_most = foreground_points[np.argmax(foreground_points[:, 1])]
        top_most = foreground_points[np.argmin(foreground_points[:, 0])]
        bottom_most = foreground_points[np.argmax(foreground_points[:, 0])]

        extreme_points = [left_most, right_most, top_most, bottom_most]
        for point in extreme_points:
            point_mask[tuple(point)] = 1

        remaining_points = self.sample_points(foreground_points, max(self.num_points - len(extreme_points), 0))
        for point in remaining_points:
            point_mask[tuple(point)] = 1

        background_points = np.argwhere(binary_mask == 0)
        selected_background_points = self.sample_points(background_points, self.num_points)
        for point in selected_background_points:
            point_mask[tuple(point)] = 0
        return point_mask
    
    def central_clicks(self, binary_mask):
        """
        Sample points with a higher probability of selecting points near the center of the lesion.
        """
        point_mask = np.full(binary_mask.shape, -1, dtype=np.float32)
        
        foreground_points = np.argwhere(binary_mask == 1)
        
        if len(foreground_points) == 0:
            return point_mask
        centroid = np.mean(foreground_points, axis=0)
        distances = np.linalg.norm(foreground_points - centroid, axis=1)
        probabilities = 1 / (distances + 1e-6)
        selected_foreground_points = self.sample_points(foreground_points, self.num_points, probabilities)
        
        for point in selected_foreground_points:
            point_mask[tuple(point)] = 1
        
        background_points = np.argwhere(binary_mask == 0)
        selected_background_points = self.sample_points(background_points, self.num_points)
        for point in selected_background_points:
            point_mask[tuple(point)] = 0
        return point_mask