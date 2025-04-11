import numpy as np
import cv2
import time
import datetime
import random
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

class DroneSimulator:
    def __init__(self, duration_days: int = 12, time_acceleration: float = 60.0):
        """
        Initialize the drone simulator.
        
        Args:
            duration_days: Number of days to simulate
            time_acceleration: Speed multiplier for simulation (e.g., 60.0 means 1 real second = 1 simulated minute)
        """
        self.duration_days = duration_days
        self.time_acceleration = time_acceleration
        self.start_time = datetime.datetime.now()
        self.current_time = self.start_time
        self.end_time = self.start_time + datetime.timedelta(days=duration_days)
        
        # Simulation parameters
        self.drone_position = (0, 0, 50)  # (x, y, altitude)
        self.drone_speed = 10  # meters per second
        self.battery_level = 100.0
        self.battery_drain_rate = 0.1  # % per minute
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the simulator"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=f"logs/simulation_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def generate_telemetry(self) -> Dict:
        """Generate simulated drone telemetry data"""
        return {
            "timestamp": self.current_time.isoformat(),
            "position": self.drone_position,
            "battery_level": self.battery_level,
            "speed": self.drone_speed,
            "altitude": self.drone_position[2],
            "temperature": random.uniform(20, 30),
            "humidity": random.uniform(40, 60),
            "wind_speed": random.uniform(0, 5)
        }
    
    def generate_synthetic_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generate a synthetic camera frame with random events"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background noise
        frame = cv2.randn(frame, (0, 0, 0), (30, 30, 30))
        
        # Randomly add simulated objects/events
        if random.random() < 0.1:  # 10% chance of event
            event_type = random.choice(["person", "vehicle", "animal"])
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(20, 100)
            
            if event_type == "person":
                color = (0, 255, 0)  # Green
            elif event_type == "vehicle":
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 0, 255)  # Red
                
            cv2.rectangle(frame, (x, y), (x + size, y + size), color, 2)
            
        return frame
    
    def update_drone_state(self):
        """Update drone state based on time elapsed"""
        # Update position (simple circular pattern for demo)
        angle = (self.current_time - self.start_time).total_seconds() / 3600
        radius = 100
        self.drone_position = (
            radius * np.cos(angle),
            radius * np.sin(angle),
            50 + 10 * np.sin(angle/2)
        )
        
        # Update battery
        self.battery_level = max(0, self.battery_level - self.battery_drain_rate)
        
    def run_simulation(self):
        """Run the simulation for the specified duration"""
        logging.info(f"Starting simulation for {self.duration_days} days")
        
        while self.current_time < self.end_time:
            # Update simulation state
            self.update_drone_state()
            
            # Generate and log telemetry
            telemetry = self.generate_telemetry()
            logging.info(f"Telemetry: {json.dumps(telemetry)}")
            
            # Generate synthetic frame
            frame = self.generate_synthetic_frame()
            
            # Save frame periodically
            if random.random() < 0.01:  # Save 1% of frames
                frame_dir = Path("simulation_frames")
                frame_dir.mkdir(exist_ok=True)
                filename = f"frame_{self.current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(frame_dir / filename), frame)
            
            # Advance simulation time
            self.current_time += datetime.timedelta(seconds=1/self.time_acceleration)
            
            # Sleep to control real-time simulation speed
            time.sleep(1/self.time_acceleration)
            
        logging.info("Simulation completed")

if __name__ == "__main__":
    # Create and run a 12-day simulation
    simulator = DroneSimulator(duration_days=12, time_acceleration=60.0)
    simulator.run_simulation() 