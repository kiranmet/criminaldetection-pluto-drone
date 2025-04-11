import numpy as np
import matplotlib.pyplot as plt
from pluto import Drone, Environment
import cv2
import time
import logging
from detection import CrimeDetectionSystem

class SecurityDroneSimulation:
    def __init__(self):
        # Initialize drone with security-specific parameters
        self.drone = Drone(
            mass=2.0,  # kg
            max_thrust=25.0,  # N
            max_velocity=20.0,  # m/s
            battery_capacity=8000  # mAh
        )
        
        # Create simulation environment
        self.env = Environment(
            gravity=9.81,
            air_density=1.225,
            wind_speed=0.0
        )
        
        # Initialize position and velocity
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Security-specific parameters
        self.patrol_points = []
        self.suspicious_activities = []
        self.alert_threshold = 0.8
        self.night_mode = False
        self.thermal_mode = False
        
        # Initialize crime detection system
        self.detection_system = CrimeDetectionSystem()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SecurityDrone")
        
    def setup_patrol_route(self, points):
        """Define patrol route with waypoints"""
        self.patrol_points = points
        self.logger.info(f"Patrol route established with {len(points)} waypoints")
        
    def toggle_night_vision(self):
        """Toggle night vision mode"""
        self.night_mode = not self.night_mode
        self.logger.info(f"Night vision mode: {'ON' if self.night_mode else 'OFF'}")
        
    def toggle_thermal_imaging(self):
        """Toggle thermal imaging mode"""
        self.thermal_mode = not self.thermal_mode
        self.logger.info(f"Thermal imaging mode: {'ON' if self.thermal_mode else 'OFF'}")
        
    def generate_simulation_frame(self):
        """Generate a simulated camera frame for processing"""
        # In a real implementation, this would capture from a camera
        # For simulation, we'll create a blank frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Add some random elements to simulate a scene
        # Add ground
        cv2.rectangle(frame, (0, 800), (1920, 1080), (100, 100, 100), -1)
        
        # Add some buildings
        for _ in range(10):
            x = np.random.randint(0, 1920)
            y = np.random.randint(200, 800)
            w = np.random.randint(50, 200)
            h = np.random.randint(100, 600)
            color = (np.random.randint(100, 200), np.random.randint(100, 200), np.random.randint(100, 200))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
            
        # Add some roads
        cv2.rectangle(frame, (0, 750), (1920, 770), (50, 50, 50), -1)
        cv2.rectangle(frame, (950, 0), (970, 1080), (50, 50, 50), -1)
        
        return frame
        
    def process_surveillance(self):
        """Process surveillance data for suspicious activities"""
        # Generate a simulated frame
        frame = self.generate_simulation_frame()
        
        # Process the frame with the crime detection system
        detection_results = self.detection_system.process_frame(frame)
        
        # Check if suspicious activity was detected
        if detection_results['suspicious_activity']:
            alert_details = detection_results['alert_details']
            self.suspicious_activities.append({
                'position': self.position.copy(),
                'timestamp': time.time(),
                'alert_details': alert_details
            })
            self.alert_authorities(alert_details)
            
        return detection_results
        
    def alert_authorities(self, alert_details):
        """Alert law enforcement of suspicious activity"""
        self.logger.info(f"ALERT: Notifying law enforcement of suspicious activity")
        self.logger.info(f"Location: {self.position}")
        self.logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Alert Type: {alert_details['type']}")
        
        if 'action' in alert_details:
            self.logger.info(f"Action: {alert_details['action']}")
            
        self.logger.info(f"Confidence: {alert_details['confidence']:.2f}")
        
    def patrol(self, duration=60.0):
        """Execute patrol route"""
        if not self.patrol_points:
            self.logger.warning("No patrol route defined!")
            return
            
        time_elapsed = 0
        current_point = 0
        
        while time_elapsed < duration:
            # Move to next patrol point
            target = self.patrol_points[current_point]
            direction = target - self.position
            distance = np.linalg.norm(direction)
            
            if distance < 0.1:  # Close enough to current waypoint
                current_point = (current_point + 1) % len(self.patrol_points)
            else:
                # Move towards target
                self.velocity = direction / distance * 5.0  # 5 m/s patrol speed
                self.position += self.velocity * 0.1
                
                # Process surveillance data
                detection_results = self.process_surveillance()
                
                # Log detection summary
                if detection_results['suspicious_activity']:
                    self.logger.info(f"Patrolling... Position: {self.position}")
                    self.logger.info(f"Detected suspicious activity: {detection_results['alert_details']['type']}")
                else:
                    self.logger.info(f"Patrolling... Position: {self.position}")
                    
            time_elapsed += 0.1
            
    def run_simulation(self):
        """Run complete security simulation"""
        self.logger.info("Starting security drone simulation...")
        
        # Setup patrol route (example points)
        patrol_route = [
            np.array([0.0, 0.0, 10.0]),
            np.array([10.0, 10.0, 10.0]),
            np.array([10.0, -10.0, 10.0]),
            np.array([-10.0, -10.0, 10.0]),
            np.array([-10.0, 10.0, 10.0])
        ]
        self.setup_patrol_route(patrol_route)
        
        # Enable night vision and thermal imaging
        self.toggle_night_vision()
        self.toggle_thermal_imaging()
        
        # Execute patrol
        self.patrol(duration=30.0)
        
        # Get alert summary
        alert_summary = self.detection_system.get_alert_summary()
        
        self.logger.info("Simulation completed successfully!")
        self.logger.info(f"Detected {len(self.suspicious_activities)} suspicious activities")
        self.logger.info(f"Total alerts generated: {alert_summary['total_alerts']}")
        
        # Print detailed detection summary
        self.logger.info("\nDetection Summary:")
        self.logger.info(f"Person detections: {len(self.detection_system.tracked_objects)}")
        self.logger.info(f"Suspicious actions: {sum(1 for a in self.suspicious_activities if 'action' in a.get('alert_details', {}))}")
        self.logger.info(f"Anomalies detected: {sum(1 for a in self.suspicious_activities if a.get('alert_details', {}).get('type') == 'suspicious_action')}")
        self.logger.info(f"Behavior patterns: {sum(1 for a in self.suspicious_activities if a.get('alert_details', {}).get('type') in ['loitering', 'erratic_movement'])}")
        self.logger.info(f"Crowd anomalies: {sum(1 for a in self.suspicious_activities if a.get('alert_details', {}).get('type') == 'unusual_crowd_density')}")
        self.logger.info(f"Vehicle anomalies: {sum(1 for a in self.suspicious_activities if 'vehicle' in a.get('alert_details', {}).get('type', ''))}")
        self.logger.info(f"Property crimes: {sum(1 for a in self.suspicious_activities if a.get('alert_details', {}).get('type') in ['vandalism', 'theft', 'breaking_and_entering', 'trespassing', 'property_damage', 'graffiti'])}")
        self.logger.info(f"Violent crimes: {sum(1 for a in self.suspicious_activities if a.get('alert_details', {}).get('type') in ['assault', 'robbery', 'threat', 'weapon_possession', 'hostage_situation', 'public_disorder'])}")

if __name__ == "__main__":
    # Create and run simulation
    sim = SecurityDroneSimulation()
    sim.run_simulation() 