import numpy as np

class UncertaintyAwareConfidence:
    """
    Calculates session confidence with uncertainty decomposition.
    """
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.bpm_history = []
        self.sqi_history = []
        self.agreement_history = []
        
    def update(self, bpm, sqi, agreement):
        if bpm > 0:
            self.bpm_history.append(bpm)
            self.sqi_history.append(sqi)
            self.agreement_history.append(agreement)
            
            if len(self.bpm_history) > self.window_size:
                self.bpm_history.pop(0)
                self.sqi_history.pop(0)
                self.agreement_history.pop(0)
                
    def get_confidence_metrics(self):
        if len(self.bpm_history) < 10:
            return 0.0, {}
            
        # 1. Aleatoric Uncertainty (Signal Noise)
        # Inversely proportional to SQI
        aleatoric = 1.0 - (np.mean(self.sqi_history) / 100.0)
        
        # 2. Epistemic Uncertainty (Model/Data Consistency)
        # Proportional to BPM variance and ROI disagreement
        bpm_std = np.std(self.bpm_history)
        agreement_err = 1.0 - np.mean(self.agreement_history)
        epistemic = (bpm_std / 20.0) + agreement_err
        
        # 3. Total Uncertainty
        total_uncertainty = aleatoric + epistemic
        
        # 4. Confidence Interval (95%)
        # Simplified: 1.96 * total_uncertainty
        ci_range = 1.96 * (bpm_std + 2.0) # Base CI on BPM std
        
        # 5. Final Confidence Score
        confidence = np.exp(-total_uncertainty / 2.0) * 100.0
        
        return float(confidence), {
            "aleatoric_uncertainty": float(aleatoric),
            "epistemic_uncertainty": float(epistemic),
            "total_uncertainty": float(total_uncertainty),
            "confidence_interval_95": float(ci_range)
        }

def propagate_variance(roi_variances, weights):
    """
    Propagate variance through weighted fusion.
    Var(sum(w_i * X_i)) = sum(w_i^2 * Var(X_i)) assuming independence.
    """
    weights = np.array(weights)
    weights /= np.sum(weights)
    
    total_var = np.sum((weights**2) * np.array(roi_variances))
    return float(total_var)
