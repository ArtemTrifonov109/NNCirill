import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from config import STATISTICS_PATH

class Statistics:
    def __init__(self, model_name):
        """Initialize statistics tracker"""
        self.model_name = model_name
        self.stats_dir = os.path.join(STATISTICS_PATH, model_name)
        os.makedirs(self.stats_dir, exist_ok=True)

        self.usage_file = os.path.join(self.stats_dir, "usage_stats.csv")
        if not os.path.exists(self.usage_file):
            pd.DataFrame(columns=[
                'timestamp', 'model_type', 'input_letter', 'predicted_letter',
                'confidence', 'correct', 'response_time'
            ]).to_csv(self.usage_file, index=False)

        self.training_file = os.path.join(self.stats_dir, "training_stats.json")

    def record_training_stats(self, ffnn_history, tdnn_history, config, training_time):
        """Record training statistics"""
        ffnn_stats = {
            'final_accuracy': float(ffnn_history.history['accuracy'][-1]),
            'final_val_accuracy': float(ffnn_history.history['val_accuracy'][-1]),
            'final_loss': float(ffnn_history.history['loss'][-1]),
            'final_val_loss': float(ffnn_history.history['val_loss'][-1]),
            'accuracy_history': [float(acc) for acc in ffnn_history.history['accuracy']],
            'val_accuracy_history': [float(acc) for acc in ffnn_history.history['val_accuracy']],
            'loss_history': [float(loss) for loss in ffnn_history.history['loss']],
            'val_loss_history': [float(loss) for loss in ffnn_history.history['val_loss']]
        }

        tdnn_stats = {
            'final_accuracy': float(tdnn_history.history['accuracy'][-1]),
            'final_val_accuracy': float(tdnn_history.history['val_accuracy'][-1]),
            'final_loss': float(tdnn_history.history['loss'][-1]),
            'final_val_loss': float(tdnn_history.history['val_loss'][-1]),
            'accuracy_history': [float(acc) for acc in tdnn_history.history['accuracy']],
            'val_accuracy_history': [float(acc) for acc in tdnn_history.history['val_accuracy']],
            'loss_history': [float(loss) for loss in tdnn_history.history['loss']],
            'val_loss_history': [float(loss) for loss in tdnn_history.history['val_loss']]
        }

        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': config,
            'training_time_seconds': training_time,
            'ffnn_model': ffnn_stats,
            'tdnn_model': tdnn_stats
        }

        with open(self.training_file, 'w') as f:
            json.dump(stats, f, indent=4)

    def record_prediction(self, model_type, true_letter, predicted_letter, confidence, response_time):
        """Record prediction statistics"""
        correct = (true_letter == predicted_letter) if true_letter else None

        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'input_letter': true_letter if true_letter else 'unknown',
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'correct': correct,
            'response_time': response_time
        }

        df = pd.DataFrame([entry])
        df.to_csv(self.usage_file, mode='a', header=False, index=False)

    def generate_usage_report(self):
        """Generate usage statistics report"""
        if not os.path.exists(self.usage_file):
            return None

        df = pd.read_csv(self.usage_file)
        if df.empty:
            return None

        report = {
            'total_predictions': len(df),
            'ffnn_predictions': len(df[df['model_type'] == 'FFNN']),
            'tdnn_predictions': len(df[df['model_type'] == 'TDNN']),
            'correct_predictions': df['correct'].sum() if 'correct' in df.columns else 0,
            'accuracy': float(df['correct'].mean()) if 'correct' in df.columns else 0,
            'average_confidence': float(df['confidence'].mean()),
            'average_response_time': float(df['response_time'].mean()),
            'last_usage': df['timestamp'].max()
        }

        letter_stats = {}
        for letter in df['predicted_letter'].unique():
            letter_df = df[df['predicted_letter'] == letter]
            letter_stats[letter] = {
                'count': len(letter_df),
                'avg_confidence': float(letter_df['confidence'].mean())
            }

        report['letter_statistics'] = letter_stats

        report_file = os.path.join(self.stats_dir, "usage_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)

        return report