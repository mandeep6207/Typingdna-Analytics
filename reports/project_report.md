# TypingDNA Analytics Project Report

## Overview

This project generated 5,000 synthetic typing sessions and trained three supervised classification models to identify typing behavior styles.

## Target Distribution

typing_style
Fast Typist            1336
Balanced Typist        1657
Careful Typist          869
Inconsistent Typist    1138

## Best Model

- Model: Random Forest
- Weighted F1: 0.9830
- Accuracy: 0.9830

## Dataset Summary

           wpm  accuracy  error_rate  backspace_count  pause_time_ms  session_duration_min  words_typed
count  5000.00   5000.00     5000.00          5000.00        5000.00               5000.00      5000.00
mean     75.90     91.63       10.80            17.06         517.96                 19.60        24.03
std      20.89      6.09        7.44            12.37         310.73                  9.27        11.91
min      20.00     70.00        0.00             0.00          50.00                  2.00         1.00
25%      61.50     89.30        5.40             9.00         276.00                 13.08        16.00
50%      74.20     93.10        9.00            14.00         445.00                 19.00        23.00
75%      92.00     95.90       13.80            21.00         712.25                 25.50        31.00
max     120.00    100.00       30.00            79.00        1500.00                 60.00        86.00

## Key Findings

- Faster sessions tend to pair high WPM with lower pause time and fewer corrections.
- Careful typists show the highest accuracy and the longest pause times.
- Inconsistent typists are separated by wider spreads in error rate, backspace usage, and session rhythm.
- Tree-based models are expected to outperform the linear baseline because the classes are rule-driven and non-linear.

## Recommended Use

Use the saved best model for batch scoring or portfolio demonstrations. The artifacts in `models/`, `visuals/`, `metrics/`, and `reports/` are all reproducible from the pipeline.
