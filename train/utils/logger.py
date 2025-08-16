import csv
import os
from datetime import datetime

class Logger:
    """A simple logger to save training data to a CSV file."""
    def __init__(self, log_dir, agent_name, env_name):
        self.log_dir = os.path.join(log_dir, f"{agent_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file_path = os.path.join(self.log_dir, 'training_log.csv')
        self.fieldnames = [
            'episode', 'score', 'steps', 'duration_sec', 
            'avg_speed_steps_sec', 'success'
        ]
        
        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_episode(self, data):
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)
        
        # Print a summary to console
        print(f"Episode {data['episode']}: Score={data['score']:.2f}, Steps={data['steps']}, Duration={data['duration_sec']:.2f}s, Success={data['success']}")

    def get_log_dir(self):
        return self.log_dir