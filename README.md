# TypingDNA Analytics

TypingDNA Analytics is an end-to-end pure data science project that synthesizes typing-behavior data, explores the patterns behind typing styles, trains multiple classifiers, and packages the final artifacts for portfolio use.

The project classifies sessions into four behavioral styles:

- Fast Typist
- Balanced Typist
- Careful Typist
- Inconsistent Typist

## Highlights

- 5,000 realistic synthetic typing sessions with rule-based labels
- Reproducible cleaning, feature engineering, and exploratory analysis
- Three classification models evaluated on weighted F1-score
- Best model automatically selected and saved with the label encoder
- Publication-ready plots, metrics, and a written project report

## Workflow

1. Generate a synthetic dataset with realistic typing ranges.
2. Clean and validate the session-level records.
3. Explore the behavioral patterns and class balance.
4. Train Logistic Regression, Random Forest, and Gradient Boosting models.
5. Select the best model by weighted F1-score.
6. Save artifacts, reports, and visualizations for reuse.

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

## Project Structure

```text
typingdna-analytics/
├── data/
│   ├── typing_behavior.csv
│   └── cleaned_typing_behavior.csv
├── notebooks/
│   └── typingdna_analysis.ipynb
├── models/
│   ├── typing_style_classifier.pkl
│   └── label_encoder.pkl
├── visuals/
│   ├── style_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── reports/
│   ├── model_metrics.json
│   └── project_report.md
├── metrics/
│   └── classification_report.txt
├── src/
│   └── typingdna_analytics/
│       └── pipeline.py
├── run_pipeline.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The synthetic dataset contains the following features:

- `wpm`
- `accuracy`
- `error_rate`
- `backspace_count`
- `pause_time_ms`
- `session_duration_min`
- `words_typed`

The target label `typing_style` is assigned through realistic behavioral rules derived from the generated features.

## Results

The current benchmark from the generated dataset is:

- Best model: Random Forest
- Weighted F1-score: 0.9830
- Accuracy: 0.9830

| Model | Weighted F1 | Accuracy |
| --- | ---: | ---: |
| Logistic Regression | 0.9640 | 0.9640 |
| Random Forest | 0.9830 | 0.9830 |
| Gradient Boosting | 0.9790 | 0.9790 |

## Quick Start

1. Create or activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the full pipeline:

```bash
python run_pipeline.py
```

This will generate the dataset, clean the data, train and compare models, save the best model, and export all charts and reports.

## Output Artifacts

- `data/typing_behavior.csv` and `data/cleaned_typing_behavior.csv`
- `models/typing_style_classifier.pkl` and `models/label_encoder.pkl`
- `visuals/style_distribution.png`
- `visuals/correlation_heatmap.png`
- `visuals/feature_importance.png`
- `visuals/confusion_matrix.png`
- `reports/model_metrics.json`
- `reports/project_report.md`
- `metrics/classification_report.txt`

## Notes

The pipeline is fully reproducible from the repository root. The root entrypoint adds `src/` to the import path so the project can be run directly without installing the package first.

If you want to inspect the analysis in notebook form, open `notebooks/typingdna_analysis.ipynb` after running the pipeline once.

## Troubleshooting

- If the notebook shows stale outputs, rerun `python run_pipeline.py` before reopening it.
- If you want to regenerate every artifact from scratch, delete the generated folders and rerun the pipeline:

```bash
python run_pipeline.py
```
