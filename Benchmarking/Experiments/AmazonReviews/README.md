# AmazonReviews Experiment Pipeline

This directory contains the complete experimentation pipeline for the AmazonReviews dataset.

## 🏗️ **Directory Structure**

```
Benchmarking/Experiments/AmazonReviews/
├── run_amazon_reviews_experiment.py    # Main orchestration script
├── phase_scripts/                      # Individual phase implementations
│   ├── phase_1_data_validation.py     # Data validation & preprocessing
│   ├── phase_2_baseline.py            # Baseline model evaluation
│   ├── phase_3_mainmodel.py           # MainModel hyperparameter optimization
│   ├── phase_4_ablation.py            # Ablation studies
│   ├── phase_5_interpretability.py    # Interpretability studies
│   ├── phase_6_robustness.py          # Robustness tests
│   ├── phase_7_comparative.py         # Comparative analysis
│   └── phase_8_production.py          # Production readiness assessment
├── results/                            # Experiment results (created automatically)
│   └── amazon_reviews_[mode]_[timestamp]/
│       ├── seed_[X]/
│       │   ├── phase_1_data_validation/
│       │   ├── phase_2_baseline/
│       │   ├── phase_3_mainmodel/
│       │   ├── phase_4_ablation/
│       │   ├── phase_5_interpretability/
│       │   ├── phase_6_robustness/
│       │   ├── phase_7_comparative/
│       │   └── phase_8_production/
│       └── comprehensive_report/
└── README.md                           # This file
```

## 🚀 **Usage**

### **Quick Test Mode**
```bash
cd Benchmarking/Experiments/AmazonReviews
python run_amazon_reviews_experiment.py --mode quick
```

**Quick Mode Features:**
- Uses 30% subset of full dataset
- 25 hyperparameter trials (half of full)
- Single seed (42)
- Reduced CV folds (3)
- Basic interpretability and robustness tests
- Skips production tests

### **Full Test Mode**
```bash
cd Benchmarking/Experiments/AmazonReviews
python run_amazon_reviews_experiment.py --mode full
```

**Full Mode Features:**
- Uses complete dataset
- 50 hyperparameter trials
- Multiple seeds (42, 123, 456, 789, 999)
- Full CV folds (5)
- Comprehensive interpretability and robustness tests
- Includes production readiness assessment

### **Verbose Logging**
```bash
python run_amazon_reviews_experiment.py --mode quick --verbose
```

## 📊 **Test Modes Comparison**

| Feature | Quick Mode | Full Mode |
|---------|------------|-----------|
| Dataset Size | 30% subset | 100% full |
| Hyperparameter Trials | 25 | 50 |
| Seeds | 1 (42) | 5 (42, 123, 456, 789, 999) |
| CV Folds | 3 | 5 |
| Max Epochs | 20 | 50 |
| Ablation Trials | 2 | 5 |
| Robustness Tests | Core only | All tests |
| Production Tests | No | Yes |
| Estimated Duration | ~2-4 hours | ~8-16 hours |

## 🎯 **Phase Descriptions**

### **Phase 1: Data Validation & Preprocessing**
- Sample count verification
- Feature dimension consistency
- Missing value analysis
- Cross-modal alignment
- Memory usage optimization

### **Phase 2: Baseline Model Evaluation**
- Unimodal baselines (text, metadata)
- Traditional fusion methods
- Ensemble baselines
- Performance comparison

### **Phase 3: MainModel Hyperparameter Optimization**
- Ensemble configuration optimization
- Modality dropout strategy tuning
- Base learner selection optimization
- Training parameter tuning
- Aggregation strategy optimization

### **Phase 4: Ablation Studies**
- Modality dropout ablation
- Adaptive learner selection ablation
- Cross-modal denoising ablation
- Transformer meta-learner ablation
- Ensemble size ablation

### **Phase 5: Interpretability Studies**
- Modality importance analysis
- Learner contribution analysis
- Decision path analysis
- Uncertainty quantification

### **Phase 6: Robustness Tests**
- Missing modality robustness
- Noise robustness
- Adversarial robustness
- Distribution shift robustness
- Scalability robustness

### **Phase 7: Comparative Analysis**
- Performance comparison
- Statistical analysis
- Computational analysis

### **Phase 8: Production Readiness Assessment**
- API performance testing
- Deployment testing
- Maintenance testing

## 📁 **Output Files**

Each phase generates specific output files:

- `[phase]_results.json`: Main results summary
- `data_quality_report.json`: Data validation results (Phase 1)
- `baseline_results.json`: Baseline model results (Phase 2)
- `mainmodel_trials.json`: Hyperparameter trial results (Phase 3)
- `mainmodel_best.json`: Best configuration summary (Phase 3)
- `ablation_results.json`: Ablation study results (Phase 4)
- `interpretability_report.json`: Interpretability analysis (Phase 5)
- `robustness_results.json`: Robustness test results (Phase 6)
- `comparative_analysis.json`: Comparative analysis (Phase 7)
- `production_assessment.json`: Production readiness (Phase 8)

## 🔧 **Configuration**

The experiment configuration is automatically generated based on the test mode and saved to `experiment_config.json` in the results directory.

Key configuration parameters:
- `dataset_subset_ratio`: Dataset size (0.3 for quick, 1.0 for full)
- `hyperparameter_trials`: Number of trials (25 for quick, 50 for full)
- `seeds`: Random seeds to use
- `cross_validation_folds`: CV folds (3 for quick, 5 for full)
- `max_epochs`: Training epochs (20 for quick, 50 for full)
- `ensemble_bags`: Bag size options
- `dropout_rates`: Modality dropout rates
- `learning_rates`: Learning rate options

## 📈 **Expected Results**

### **Performance Targets**
- **Accuracy Improvement**: >5% over best baseline
- **Robustness**: <10% performance drop under missing modalities
- **Efficiency**: <2x training time compared to best baseline
- **Interpretability**: >80% explanation accuracy

### **Statistical Requirements**
- **Significance Level**: p < 0.05 for all comparisons
- **Effect Size**: Cohen's d > 0.5 for meaningful differences
- **Confidence Intervals**: 95% CI for all performance metrics

## 🐛 **Troubleshooting**

### **Common Issues**
1. **Phase script not found**: Create the missing phase script in `phase_scripts/`
2. **Dataset path error**: Ensure AmazonReviews data exists in `../../ProcessedData/AmazonReviews/`
3. **Memory issues**: Use quick mode for testing, full mode requires more memory
4. **Long execution time**: Full mode can take 8-16 hours, use quick mode for testing

### **Logs**
- Experiment logs: `amazon_reviews_experiment.log`
- Individual phase logs: Available in each phase directory

## 🎯 **Next Steps**

1. **Implement Phase Scripts**: Create actual implementations for each phase
2. **Test Quick Mode**: Run quick mode to verify pipeline works
3. **Run Full Mode**: Execute complete experiment for publication
4. **Analyze Results**: Review comprehensive report for insights
5. **Iterate**: Refine based on results and run additional experiments

This pipeline ensures thorough, publication-ready evaluation of the multimodal ensemble model! 🚀
