import time
import datetime
import random
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import psutil
import platform

class AppTestSimulator:
    def __init__(self, num_accounts: int = 12, duration_days: int = 14, time_acceleration: float = 60.0):
        """
        Initialize the app testing simulator.
        
        Args:
            num_accounts: Number of user accounts to simulate (default: 12)
            duration_days: Number of days to simulate (default: 14)
            time_acceleration: Speed multiplier for simulation (e.g., 60.0 means 1 real second = 1 simulated minute)
        """
        self.num_accounts = num_accounts
        self.duration_days = duration_days
        self.time_acceleration = time_acceleration
        self.start_time = datetime.datetime.now()
        self.current_time = self.start_time
        self.end_time = self.start_time + datetime.timedelta(days=duration_days)
        
        # Initialize user accounts
        self.user_accounts = self.initialize_user_accounts()
        
        # Test parameters
        self.test_sessions = []
        self.crash_count = 0
        self.performance_metrics = []
        
        # Setup logging
        self.setup_logging()
        
    def initialize_user_accounts(self) -> List[Dict]:
        """Initialize user accounts with different characteristics"""
        accounts = []
        for i in range(self.num_accounts):
            account = {
                "account_id": f"user_{i+1}",
                "name": f"Test User {i+1}",
                "preferences": {
                    "theme": random.choice(["light", "dark"]),
                    "language": random.choice(["en", "es", "fr", "de"]),
                    "notifications": random.choice([True, False])
                },
                "usage_pattern": random.choice(["casual", "power", "business"]),
                "session_history": [],
                "crash_history": []
            }
            accounts.append(account)
        return accounts
    
    def setup_logging(self):
        """Configure logging for the simulator"""
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            filename=f"test_logs/app_test_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def simulate_user_session(self, account: Dict) -> Dict:
        """Simulate a user session with the app for a specific account"""
        # Adjust session duration based on usage pattern
        if account["usage_pattern"] == "casual":
            session_duration = random.randint(30, 300)  # 30 seconds to 5 minutes
            actions = random.randint(5, 20)
        elif account["usage_pattern"] == "power":
            session_duration = random.randint(300, 1200)  # 5 to 20 minutes
            actions = random.randint(20, 50)
        else:  # business
            session_duration = random.randint(600, 1800)  # 10 to 30 minutes
            actions = random.randint(30, 100)
        
        # Simulate different types of user interactions
        interaction_types = [
            "screen_navigation",
            "button_click",
            "form_submission",
            "content_view",
            "settings_change",
            "search_query",
            "purchase_attempt",
            "data_sync",
            "file_upload",
            "chat_message"
        ]
        
        session_actions = []
        for _ in range(actions):
            action = {
                "type": random.choice(interaction_types),
                "timestamp": self.current_time.isoformat(),
                "duration": random.uniform(0.1, 5.0),
                "success": random.random() > 0.05  # 95% success rate
            }
            session_actions.append(action)
            
        return {
            "account_id": account["account_id"],
            "session_id": len(account["session_history"]) + 1,
            "start_time": self.current_time.isoformat(),
            "duration_seconds": session_duration,
            "actions": session_actions,
            "device_info": self.get_device_info(),
            "usage_pattern": account["usage_pattern"]
        }
    
    def get_device_info(self) -> Dict:
        """Get simulated device information"""
        android_versions = ["11.0", "12.0", "13.0", "14.0"]
        device_models = [
            "Samsung Galaxy S21",
            "Google Pixel 6",
            "OnePlus 9",
            "Xiaomi Mi 11",
            "Sony Xperia 1 III",
            "Samsung Galaxy S22",
            "Google Pixel 7",
            "OnePlus 10",
            "Xiaomi Mi 12",
            "Sony Xperia 5 IV"
        ]
        
        return {
            "model": random.choice(device_models),
            "android_version": random.choice(android_versions),
            "screen_resolution": f"{random.choice([1080, 1440])}x{random.choice([1920, 2560])}",
            "ram_gb": random.choice([4, 6, 8, 12]),
            "storage_gb": random.choice([64, 128, 256]),
            "battery_level": random.randint(20, 100)
        }
    
    def check_app_performance(self) -> Dict:
        """Simulate performance metrics"""
        return {
            "timestamp": self.current_time.isoformat(),
            "cpu_usage": random.uniform(5, 30),
            "memory_usage_mb": random.uniform(100, 500),
            "response_time_ms": random.uniform(50, 300),
            "frame_rate": random.uniform(45, 60)
        }
    
    def simulate_crash(self, account: Dict) -> Optional[Dict]:
        """Simulate app crashes with a low probability"""
        if random.random() < 0.001:  # 0.1% chance of crash
            self.crash_count += 1
            crash_types = [
                "NullPointerException",
                "OutOfMemoryError",
                "NetworkTimeoutException",
                "UIThreadBlocked",
                "PermissionDenied"
            ]
            
            crash = {
                "account_id": account["account_id"],
                "timestamp": self.current_time.isoformat(),
                "crash_type": random.choice(crash_types),
                "stack_trace": "Simulated stack trace for testing",
                "device_info": self.get_device_info()
            }
            account["crash_history"].append(crash)
            return crash
        return None
    
    def run_simulation(self):
        """Run the app testing simulation"""
        logging.info(f"Starting app test simulation for {self.duration_days} days with {self.num_accounts} accounts")
        
        while self.current_time < self.end_time:
            # Simulate sessions for each account
            for account in self.user_accounts:
                # Adjust session probability based on usage pattern
                session_probability = 0.3
                if account["usage_pattern"] == "power":
                    session_probability = 0.5
                elif account["usage_pattern"] == "business":
                    session_probability = 0.7
                
                if random.random() < session_probability:
                    session = self.simulate_user_session(account)
                    account["session_history"].append(session)
                    self.test_sessions.append(session)
                    logging.info(f"New user session for {account['account_id']}: {json.dumps(session)}")
                
                # Check for crashes
                crash = self.simulate_crash(account)
                if crash:
                    logging.error(f"App crash detected for {account['account_id']}: {json.dumps(crash)}")
            
            # Check performance
            performance = self.check_app_performance()
            self.performance_metrics.append(performance)
            
            # Save session data periodically
            if random.random() < 0.01:  # Save 1% of sessions
                self.save_test_data()
            
            # Advance simulation time
            self.current_time += datetime.timedelta(seconds=1/self.time_acceleration)
            
            # Sleep to control real-time simulation speed
            time.sleep(1/self.time_acceleration)
        
        logging.info("Simulation completed")
        self.generate_test_report()
    
    def save_test_data(self):
        """Save test data to file"""
        data_dir = Path("test_data")
        data_dir.mkdir(exist_ok=True)
        
        data = {
            "accounts": self.user_accounts,
            "sessions": self.test_sessions,
            "performance_metrics": self.performance_metrics,
            "crash_count": self.crash_count
        }
        
        filename = f"test_data/test_data_{self.current_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_test_report(self):
        """Generate final test report"""
        # Calculate account-specific metrics
        account_metrics = {}
        for account in self.user_accounts:
            sessions = account["session_history"]
            crashes = account["crash_history"]
            
            account_metrics[account["account_id"]] = {
                "total_sessions": len(sessions),
                "total_crashes": len(crashes),
                "average_session_duration": sum(
                    s["duration_seconds"] for s in sessions
                ) / len(sessions) if sessions else 0,
                "crash_rate": len(crashes) / len(sessions) if sessions else 0,
                "usage_pattern": account["usage_pattern"]
            }
        
        report = {
            "simulation_duration_days": self.duration_days,
            "total_accounts": self.num_accounts,
            "total_sessions": len(self.test_sessions),
            "total_crashes": self.crash_count,
            "account_metrics": account_metrics,
            "performance_summary": {
                "average_cpu_usage": sum(
                    m["cpu_usage"] for m in self.performance_metrics
                ) / len(self.performance_metrics) if self.performance_metrics else 0,
                "average_memory_usage": sum(
                    m["memory_usage_mb"] for m in self.performance_metrics
                ) / len(self.performance_metrics) if self.performance_metrics else 0,
                "average_response_time": sum(
                    m["response_time_ms"] for m in self.performance_metrics
                ) / len(self.performance_metrics) if self.performance_metrics else 0
            }
        }
        
        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)
        
        filename = f"test_reports/final_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Test report generated: {filename}")

if __name__ == "__main__":
    # Create and run a 14-day simulation with 12 accounts
    simulator = AppTestSimulator(num_accounts=12, duration_days=14, time_acceleration=60.0)
    simulator.run_simulation() 