# Lyapunov-based Emotion Stability Switching System

This project implements a switching decision system for AI customer service based on **Lyapunov stability theory**, aiming to detect user emotional instability and decide whether to escalate from AI to human agents.

## 🧠 Overview

The system models the user's emotional dynamics using PAD (Pleasure-Arousal-Dominance) vectors and fits a personalized state-transition matrix $A_i$. Then, using Lyapunov stability theory, it determines whether the user's current emotional state lies inside a “safe region.” If the state is outside, a switch to a human agent is triggered.

## 🧠 Key Features

- Personalized emotional dynamics modeling with system identification.
- Lyapunov-based emotion stability judgment.
- Adaptive thresholding based on historical user emotions.
- Integrated with DeepSeek API for synthetic dialog generation.
- Evaluation via precision, recall, F1 score, switch rate, and false switch rate.

## 🗂️ Project Structure

.
├── fixed_emotion_dataset_20250526_205726.csv # Historical PAD dataset (training)
├── Updated_Lyapunov_Dataset_with_Speaker.csv # Test dataset with ground truth
├── datafix.py # Data cleaning
├── deepseek_generate_text.py # Uses DeepSeek API to generate user dialog
├── Updated_Lyapunov_Switch_Dataset_with_Speaker.py # Adds 'speaker' column to dataset
├── Lyapunov_Adaptive_Threshold.py # Computes V(x_t) and adaptive threshold
├── Comparative test.py # Strategy comparison experiment
└── README.md

## 🔌 DeepSeek API Integration

This project uses [DeepSeek](https://deepseek.com/) to generate synthetic emotional dialogues. Before running the generation script, configure your `.env` file with your API key:

```env
DEEPSEEK_API_KEY=your_deepseek_key_here

## 🧪 Switching Strategies

| Strategy Name   | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Oracle`        | Ideal, uses ground-truth switching labels                                   |
| `A_lyapunov`    | Adaptive Lyapunov-based switching using personalized user models            |
| `B_fixed`       | Lyapunov switching with a fixed threshold (e.g., c=3.0)                     |
| `C_threshold`   | Simple arousal-threshold based rule (e.g., arousal > 0.7)                   |
| `D_none`        | No switching strategy (baseline)                                            |

## 📊 Evaluation Metrics

- **Precision**
- **Recall**
- **F1-score**
- **Average V(x_T)**: Average Lyapunov function value
- **Switch Rate**
- **False Switch Rate**

| Strategy     | Precision | Recall | F1    | Avg. V(x\_T) | Switch Rate | False Switch Rate |
| ------------ | --------- | ------ | ----- | ------------ | ----------- | ----------------- |
| Oracle       | 1.000     | 1.000  | 1.000 | 0.002961     | 0.275       | 0.000             |
| A\_lyapunov  | 0.303     | 0.479  | 0.371 | 0.002961     | 0.436       | 0.304             |
| B\_fixed     | 0.000     | 0.000  | 0.000 | 0.002961     | 0.000       | 0.000             |
| C\_threshold | 0.277     | 0.438  | 0.340 | 0.002961     | 0.436       | 0.315             |
| D\_none      | 0.000     | 0.000  | 0.000 | 0.002961     | 0.000       | 0.000             |

##✨ Conclusion
The A_lyapunov strategy achieves the best trade-off between detection precision and recall. It adapts to individual users’ emotional characteristics and offers a more stable and explainable switching mechanism compared to fixed or rule-based approaches.

## 🚀 How to Run

1. Place all CSV and script files in the same directory.
2. Ensure you have Python ≥3.8 installed and install dependencies:
   pip install pandas numpy scikit-learn
3. Run the comparison test:
python "Comparative test.py"Ensure your DeepSeek key is configured in .env if running deepseek_generate_text.py.

