import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import logging

class CrimeDetectionSystem:
    """
    Advanced crime detection system using computer vision and machine learning
    to identify suspicious activities and criminal behaviors.
    """
    
    def __init__(self):
        # Initialize detection parameters
        self.confidence_threshold = 0.75
        self.detection_history = []
        self.alert_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds
        
        # Initialize detection models
        self.person_detector = self._load_person_detection_model()
        self.action_classifier = self._load_action_classification_model()
        self.anomaly_detector = self._load_anomaly_detection_model()
        
        # Initialize tracking system
        self.tracked_objects = {}
        self.next_track_id = 0
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CrimeDetection")
        
    def _load_person_detection_model(self):
        """Load pre-trained person detection model"""
        # In a real implementation, this would load a trained model
        # For simulation, we'll return a dummy model
        self.logger.info("Loading person detection model")
        return "person_detection_model"
        
    def _load_action_classification_model(self):
        """Load pre-trained action classification model"""
        # In a real implementation, this would load a trained model
        # For simulation, we'll return a dummy model
        self.logger.info("Loading action classification model")
        return "action_classification_model"
        
    def _load_anomaly_detection_model(self):
        """Load pre-trained anomaly detection model"""
        # In a real implementation, this would load a trained model
        # For simulation, we'll return a dummy model
        self.logger.info("Loading anomaly detection model")
        return "anomaly_detection_model"
        
    def detect_persons(self, frame):
        """
        Detect persons in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected persons with bounding boxes and confidence scores
        """
        # In a real implementation, this would use the person detection model
        # For simulation, we'll generate random detections
        num_persons = np.random.randint(0, 10)
        detections = []
        
        for _ in range(num_persons):
            x = np.random.randint(0, frame.shape[1] - 100)
            y = np.random.randint(0, frame.shape[0] - 200)
            w = np.random.randint(50, 100)
            h = np.random.randint(100, 200)
            confidence = np.random.uniform(0.6, 0.99)
            
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'class': 'person'
            })
            
        return detections
        
    def classify_actions(self, frame, detections):
        """
        Classify actions of detected persons
        
        Args:
            frame: Input image frame
            detections: List of detected persons
            
        Returns:
            List of actions for each detected person
        """
        # In a real implementation, this would use the action classification model
        # For simulation, we'll assign random actions
        actions = []
        
        suspicious_actions = [
            'running', 'hiding', 'climbing', 'throwing', 'fighting',
            'loitering', 'trespassing', 'vandalism', 'theft', 'assault'
        ]
        
        normal_actions = [
            'walking', 'standing', 'sitting', 'talking', 'waiting',
            'reading', 'using phone', 'carrying', 'shopping', 'working'
        ]
        
        for detection in detections:
            # 20% chance of suspicious action
            if np.random.random() < 0.2:
                action = np.random.choice(suspicious_actions)
                confidence = np.random.uniform(0.7, 0.95)
            else:
                action = np.random.choice(normal_actions)
                confidence = np.random.uniform(0.6, 0.9)
                
            actions.append({
                'action': action,
                'confidence': confidence,
                'bbox': detection['bbox']
            })
            
        return actions
        
    def detect_anomalies(self, frame, detections, actions):
        """
        Detect anomalies in the scene
        
        Args:
            frame: Input image frame
            detections: List of detected persons
            actions: List of classified actions
            
        Returns:
            List of detected anomalies
        """
        # In a real implementation, this would use the anomaly detection model
        # For simulation, we'll generate random anomalies
        anomalies = []
        
        # Check for suspicious actions
        for action in actions:
            if action['action'] in ['running', 'hiding', 'climbing', 'throwing', 'fighting',
                                   'loitering', 'trespassing', 'vandalism', 'theft', 'assault']:
                if action['confidence'] > self.confidence_threshold:
                    anomalies.append({
                        'type': 'suspicious_action',
                        'action': action['action'],
                        'confidence': action['confidence'],
                        'bbox': action['bbox'],
                        'timestamp': time.time()
                    })
                    
        # Random chance of additional anomalies
        if np.random.random() < 0.1:
            anomaly_types = [
                'unusual_crowd', 'abandoned_object', 'suspicious_vehicle',
                'unauthorized_access', 'property_damage', 'fire_hazard'
            ]
            
            anomalies.append({
                'type': np.random.choice(anomaly_types),
                'confidence': np.random.uniform(0.7, 0.95),
                'bbox': [np.random.randint(0, frame.shape[1]), 
                         np.random.randint(0, frame.shape[0]), 
                         np.random.randint(50, 200), 
                         np.random.randint(50, 200)],
                'timestamp': time.time()
            })
            
        return anomalies
        
    def track_objects(self, detections):
        """
        Track objects across frames
        
        Args:
            detections: List of detected objects
            
        Returns:
            Updated tracking information
        """
        # Simple tracking implementation
        # In a real system, this would use more sophisticated tracking algorithms
        
        current_tracks = {}
        
        for detection in detections:
            bbox = detection['bbox']
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            
            # Find closest existing track
            min_dist = float('inf')
            matched_id = None
            
            for track_id, track in self.tracked_objects.items():
                if track['active']:
                    track_center_x = track['bbox'][0] + track['bbox'][2] / 2
                    track_center_y = track['bbox'][1] + track['bbox'][3] / 2
                    
                    dist = np.sqrt((center_x - track_center_x)**2 + (center_y - track_center_y)**2)
                    
                    if dist < min_dist and dist < 100:  # Maximum distance threshold
                        min_dist = dist
                        matched_id = track_id
                        
            if matched_id is not None:
                # Update existing track
                self.tracked_objects[matched_id]['bbox'] = bbox
                self.tracked_objects[matched_id]['last_seen'] = time.time()
                current_tracks[matched_id] = self.tracked_objects[matched_id]
            else:
                # Create new track
                new_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_objects[new_id] = {
                    'bbox': bbox,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'active': True,
                    'history': [bbox]
                }
                
                current_tracks[new_id] = self.tracked_objects[new_id]
                
        # Mark tracks as inactive if not updated
        for track_id, track in self.tracked_objects.items():
            if track_id not in current_tracks and track['active']:
                if time.time() - track['last_seen'] > 5:  # 5 seconds timeout
                    self.tracked_objects[track_id]['active'] = False
                    
        return current_tracks
        
    def analyze_behavior_patterns(self, tracks):
        """
        Analyze behavior patterns of tracked objects
        
        Args:
            tracks: Dictionary of tracked objects
            
        Returns:
            List of suspicious behavior patterns
        """
        suspicious_patterns = []
        
        for track_id, track in tracks.items():
            if len(track['history']) > 10:  # Need enough history for pattern analysis
                # Check for loitering (staying in one area for too long)
                bbox_history = track['history'][-10:]
                center_points = [(bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2) for bbox in bbox_history]
                
                # Calculate movement
                total_movement = 0
                for i in range(1, len(center_points)):
                    dx = center_points[i][0] - center_points[i-1][0]
                    dy = center_points[i][1] - center_points[i-1][1]
                    total_movement += np.sqrt(dx*dx + dy*dy)
                    
                # If movement is very small, might be loitering
                if total_movement < 50 and time.time() - track['first_seen'] > 60:
                    suspicious_patterns.append({
                        'type': 'loitering',
                        'track_id': track_id,
                        'confidence': 0.8,
                        'duration': time.time() - track['first_seen'],
                        'bbox': track['bbox']
                    })
                    
                # Check for erratic movement
                if total_movement > 500 and time.time() - track['first_seen'] < 10:
                    suspicious_patterns.append({
                        'type': 'erratic_movement',
                        'track_id': track_id,
                        'confidence': 0.75,
                        'bbox': track['bbox']
                    })
                    
        return suspicious_patterns
        
    def detect_crowd_anomalies(self, detections):
        """
        Detect anomalies in crowd behavior
        
        Args:
            detections: List of detected persons
            
        Returns:
            List of crowd anomalies
        """
        crowd_anomalies = []
        
        # Count people in different regions of the frame
        if len(detections) > 10:  # Only analyze if there's a crowd
            # Divide frame into grid
            grid_size = 4
            grid_counts = np.zeros((grid_size, grid_size))
            
            for detection in detections:
                bbox = detection['bbox']
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                
                # Determine grid cell
                grid_x = int(center_x / (1920 / grid_size))
                grid_y = int(center_y / (1080 / grid_size))
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid_counts[grid_y, grid_x] += 1
                    
            # Check for unusual crowd density
            mean_density = np.mean(grid_counts)
            std_density = np.std(grid_counts)
            
            for y in range(grid_size):
                for x in range(grid_size):
                    if grid_counts[y, x] > mean_density + 2 * std_density:
                        crowd_anomalies.append({
                            'type': 'unusual_crowd_density',
                            'grid_position': (x, y),
                            'density': grid_counts[y, x],
                            'confidence': 0.85,
                            'bbox': [x * (1920 / grid_size), 
                                    y * (1080 / grid_size), 
                                    (1920 / grid_size), 
                                    (1080 / grid_size)]
                        })
                        
        return crowd_anomalies
        
    def detect_vehicle_anomalies(self, frame):
        """
        Detect anomalies related to vehicles
        
        Args:
            frame: Input image frame
            
        Returns:
            List of vehicle anomalies
        """
        vehicle_anomalies = []
        
        # In a real implementation, this would use vehicle detection and tracking
        # For simulation, we'll generate random vehicle anomalies
        
        # 10% chance of vehicle anomaly
        if np.random.random() < 0.1:
            anomaly_types = [
                'speeding_vehicle', 'reckless_driving', 'unauthorized_vehicle',
                'vehicle_theft', 'suspicious_parking', 'vehicle_break_in'
            ]
            
            vehicle_anomalies.append({
                'type': np.random.choice(anomaly_types),
                'confidence': np.random.uniform(0.7, 0.95),
                'bbox': [np.random.randint(0, frame.shape[1]), 
                         np.random.randint(0, frame.shape[0]), 
                         np.random.randint(100, 300), 
                         np.random.randint(50, 150)],
                'timestamp': time.time()
            })
            
        return vehicle_anomalies
        
    def detect_property_crimes(self, frame):
        """
        Detect property-related crimes
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected property crimes
        """
        property_crimes = []
        
        # In a real implementation, this would use specialized models
        # For simulation, we'll generate random property crimes
        
        # 5% chance of property crime
        if np.random.random() < 0.05:
            crime_types = [
                'vandalism', 'theft', 'breaking_and_entering', 
                'trespassing', 'property_damage', 'graffiti'
            ]
            
            property_crimes.append({
                'type': np.random.choice(crime_types),
                'confidence': np.random.uniform(0.7, 0.95),
                'bbox': [np.random.randint(0, frame.shape[1]), 
                         np.random.randint(0, frame.shape[0]), 
                         np.random.randint(50, 200), 
                         np.random.randint(50, 200)],
                'timestamp': time.time()
            })
            
        return property_crimes
        
    def detect_violent_crimes(self, frame, detections, actions):
        """
        Detect violent crimes
        
        Args:
            frame: Input image frame
            detections: List of detected persons
            actions: List of classified actions
            
        Returns:
            List of detected violent crimes
        """
        violent_crimes = []
        
        # Check for fighting actions
        for action in actions:
            if action['action'] == 'fighting' and action['confidence'] > self.confidence_threshold:
                violent_crimes.append({
                    'type': 'assault',
                    'confidence': action['confidence'],
                    'bbox': action['bbox'],
                    'timestamp': time.time()
                })
                
        # Random chance of additional violent crime
        if np.random.random() < 0.05:
            crime_types = [
                'assault', 'robbery', 'threat', 'weapon_possession',
                'hostage_situation', 'public_disorder'
            ]
            
            violent_crimes.append({
                'type': np.random.choice(crime_types),
                'confidence': np.random.uniform(0.7, 0.95),
                'bbox': [np.random.randint(0, frame.shape[1]), 
                         np.random.randint(0, frame.shape[0]), 
                         np.random.randint(50, 200), 
                         np.random.randint(50, 200)],
                'timestamp': time.time()
            })
            
        return violent_crimes
        
    def process_frame(self, frame):
        """
        Process a single frame for crime detection
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing all detection results
        """
        # Detect persons in the frame
        detections = self.detect_persons(frame)
        
        # Classify actions of detected persons
        actions = self.classify_actions(frame, detections)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(frame, detections, actions)
        
        # Track objects
        tracks = self.track_objects(detections)
        
        # Analyze behavior patterns
        behavior_patterns = self.analyze_behavior_patterns(tracks)
        
        # Detect crowd anomalies
        crowd_anomalies = self.detect_crowd_anomalies(detections)
        
        # Detect vehicle anomalies
        vehicle_anomalies = self.detect_vehicle_anomalies(frame)
        
        # Detect property crimes
        property_crimes = self.detect_property_crimes(frame)
        
        # Detect violent crimes
        violent_crimes = self.detect_violent_crimes(frame, detections, actions)
        
        # Combine all detections
        all_detections = {
            'persons': detections,
            'actions': actions,
            'anomalies': anomalies,
            'tracks': tracks,
            'behavior_patterns': behavior_patterns,
            'crowd_anomalies': crowd_anomalies,
            'vehicle_anomalies': vehicle_anomalies,
            'property_crimes': property_crimes,
            'violent_crimes': violent_crimes
        }
        
        # Check if any suspicious activity detected
        suspicious_activity = False
        alert_details = None
        
        # Check for high-confidence suspicious actions
        for action in actions:
            if action['action'] in ['running', 'hiding', 'climbing', 'throwing', 'fighting',
                                   'loitering', 'trespassing', 'vandalism', 'theft', 'assault']:
                if action['confidence'] > self.confidence_threshold:
                    suspicious_activity = True
                    alert_details = {
                        'type': 'suspicious_action',
                        'action': action['action'],
                        'confidence': action['confidence'],
                        'bbox': action['bbox']
                    }
                    break
                    
        # Check for anomalies
        if not suspicious_activity and len(anomalies) > 0:
            suspicious_activity = True
            alert_details = anomalies[0]
            
        # Check for behavior patterns
        if not suspicious_activity and len(behavior_patterns) > 0:
            suspicious_activity = True
            alert_details = behavior_patterns[0]
            
        # Check for crowd anomalies
        if not suspicious_activity and len(crowd_anomalies) > 0:
            suspicious_activity = True
            alert_details = crowd_anomalies[0]
            
        # Check for vehicle anomalies
        if not suspicious_activity and len(vehicle_anomalies) > 0:
            suspicious_activity = True
            alert_details = vehicle_anomalies[0]
            
        # Check for property crimes
        if not suspicious_activity and len(property_crimes) > 0:
            suspicious_activity = True
            alert_details = property_crimes[0]
            
        # Check for violent crimes
        if not suspicious_activity and len(violent_crimes) > 0:
            suspicious_activity = True
            alert_details = violent_crimes[0]
            
        # Apply alert cooldown
        current_time = time.time()
        if suspicious_activity:
            if current_time - self.last_alert_time > self.alert_cooldown:
                self.alert_count += 1
                self.last_alert_time = current_time
                self.logger.info(f"ALERT: Suspicious activity detected! Type: {alert_details['type']}")
            else:
                suspicious_activity = False
                self.logger.info("Alert suppressed due to cooldown")
                
        return {
            'suspicious_activity': suspicious_activity,
            'alert_details': alert_details,
            'detections': all_detections
        }
        
    def get_alert_summary(self):
        """
        Get a summary of alerts generated
        
        Returns:
            Dictionary containing alert summary
        """
        return {
            'total_alerts': self.alert_count,
            'last_alert_time': self.last_alert_time,
            'alert_cooldown': self.alert_cooldown
        } 