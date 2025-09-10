Question
How can a modality-aware adaptive bagging ensemble—integrating strategic modality dropout, adaptive base learner selection, and transformer-based dynamic fusion—improve predictive accuracy, robustness to missing modalities, and interpretability in multimodal learning tasks compared to conventional fusion and ensemble methods?

Hypothesis
We hypothesize that the proposed Modality-Aware Adaptive Bagging Ensemble will:
1. Performance Gains – Achieve statistically significant improvements in predictive accuracy, F1-score, and AUROC over traditional early fusion, late fusion, and unimodal baselines, with the performance advantage increasing as the number and diversity of modalities grows.
2. Robustness – Maintain higher accuracy and lower performance degradation than baselines under simulated missing modality conditions due to strategic modality dropout during training.
3. Interpretability – Provide more actionable modality importance and learner confidence insights than existing multimodal fusion methods through attention-based meta-learning and integrated uncertainty estimation.
4. Scalability – Deliver these benefits without prohibitive computational overhead by leveraging adaptive learner selection and modular architecture design.
5. Novel Feature Impact – Each novel architectural component (modality-aware dropout bagging, adaptive base learner selection, cross-modal denoising auxiliary tasks, and transformer-based meta-learner) will contribute measurable performance improvements, with ablation studies demonstrating statistically significant degradation when any component is removed.

Novel Features:

Stage 1:
//Summary
Stage 1 begins with data loading through the load_from_files() method, which accepts file paths for training and testing labels along with modality-specific data files (supporting .npy, .csv, and .txt formats) and automatically detects data types while storing everything in memory within the SimpleDataLoader object. The data validation step ensures shape consistency across modalities, validates data types, and performs basic integrity checks to confirm that training and testing data have compatible dimensions and that all required modalities are present. Next comes data cleaning via the process_data() method, which handles missing values using configurable strategies (handle_nan: 'fill_mean', 'fill_zero', or 'drop'), manages infinite values (handle_inf: 'fill_max', 'fill_zero', or 'drop'), and optionally removes statistical outliers based on a configurable standard deviation threshold (outlier_std, default 3.0). The data preprocessing step applies optional z-score normalization (normalize parameter) to scale features to zero mean and unit variance, ensuring all modalities are processed consistently while preserving the original data structure. Finally, data preparation occurs through the get_processed_data() method, which packages all cleaned and processed data (train_data, test_data, train_labels, test_labels, modality_configs, and metadata) into a dictionary format that can be directly passed to Stage 2, with the entire process being memory-only (no physical file creation) for efficiency. The data quality reporting feature via get_data_summary() and print_summary() provides comprehensive statistics about the loaded and processed data for validation and debugging purposes.

//Hyperparameters
🔧 CORE DATA INTEGRATION HYPERPARAMETERS
verbose
Description: Enables verbose logging and progress information during data loading and processing
Range: [True, False]
Default: True
Testing: FIXED - Keep True for debugging and monitoring
�� DATA CLEANING HYPERPARAMETERS
handle_nan
Description: Strategy for handling missing values (NaN) in the data - can fill with mean values, zeros, or drop samples with missing data
Range: ['fill_mean', 'fill_zero', 'drop']
Default: 'fill_mean'
Testing: FIXED - Keep 'fill_mean' for robust handling
handle_inf
Description: Strategy for handling infinite values (Inf) in the data - can fill with maximum values, zeros, or drop samples with infinite values
Range: ['fill_max', 'fill_zero', 'drop']
Default: 'fill_max'
Testing: FIXED - Keep 'fill_max' for robust handling
🔧 DATA PREPROCESSING HYPERPARAMETERS
normalize
Description: Enables data normalization (standardization) to scale features to zero mean and unit variance, which can improve model performance
Range: [True, False]
Default: False
Testing: FIXED - Keep False (let models handle normalization)
remove_outliers
Description: Enables outlier removal using statistical thresholds to remove samples that are more than N standard deviations from the mean
Range: [True, False]
Default: False
Testing: FIXED - Keep False (preserve data integrity)
outlier_std
Description: Standard deviation threshold for outlier detection - samples beyond this threshold are considered outliers and removed
Range: [1.0, 2.0, 3.0, 4.0, 5.0]
Default: 3.0
Testing: FIXED - Keep 3.0 (standard threshold)

Stage 2:
//Summary
Stage 2 begins with Initialization where the ModalityDropoutBagger class is instantiated with training data, labels, modality configurations, and hyperparameters including the number of bags, dropout strategy, maximum dropout rate, minimum modalities, sample ratio, and feature sampling ratio. The Dropout Rate Calculation step implements four strategies: linear (gradual increase), exponential (exponential curve), random (uniform distribution), and the novel Adaptive strategy which computes variance-based modality importance scores, calculates inverse proportional dropout probabilities, enforces distinct modality combinations across bags, and maximizes ensemble diversity through dynamic probability adjustments. The Modality & Feature Mask Generation step creates boolean masks for each bag, determining which modalities are active based on dropout probabilities and which specific features are selected within each active modality using the feature sampling ratio, with the adaptive strategy ensuring high-importance modalities are retained more frequently while maintaining diversity. Bootstrap Sampling extracts random subsets of training data for each bag using the specified sample ratio, creating diverse training sets that prevent overfitting and improve ensemble robustness. Bag Configuration Creation generates BagConfig dataclass objects containing bag metadata including bag ID, data indices, modality masks, feature masks, dropout rates, sample ratios, diversity scores, creation timestamps, and additional metadata for tracking and analysis. Data Access & Analytics provides methods for retrieving bag data (get_bag_data()), bag information (get_bag_info()), ensemble statistics (get_ensemble_stats()), and interpretability data (get_stage2_interpretability_data()), enabling comprehensive analysis of the generated ensemble structure. Finally, Interpretability Data captures modality importance scores, feature selection patterns, dropout rate distributions, diversity metrics, and bag metadata, storing everything in memory for seamless access by subsequent stages while maintaining the complete ensemble configuration needed for training and prediction phases.

//Novel Features Present
The Adaptive Modality + Feature Dropout Strategy is a novel ensemble generation approach that dynamically optimizes ensemble diversity and robustness through intelligent modality and feature selection. The strategy begins with Modality Importance Computation where the _compute_modality_importance() function calculates variance-based importance scores for each modality by computing the variance of each modality's features, normalizing these scores to sum to 1, and storing them as _modality_importance for subsequent use. The Inverse Proportional Dropout Calculation step uses the _adaptive_dropout_strategy() function to compute dropout probabilities that are inversely proportional to modality importance, ensuring high-impact modalities are retained more frequently while maintaining diversity through the formula dropout_prob = max_dropout_rate * (1 - importance_score), with additional constraints to prevent extreme dropout rates.
The Distinct Modality Combination Enforcement ensures that each bag has a unique combination of active modalities by tracking previously used modality combinations and adjusting dropout probabilities to avoid repetition, creating diverse ensemble members that explore different modality subspaces. Feature-Level Sampling implements the _create_feature_masks() function with the feature_sampling_ratio parameter, randomly selecting a subset of features within each active modality (e.g., 40 out of 50 text features, 24 out of 30 image features) to create intra-modality diversity and prevent overfitting to specific feature patterns. The Ensemble Diversity Maximization step calculates diversity scores using Hamming distance between modality masks across bags, dynamically adjusting sampling probabilities to maximize ensemble diversity while preserving predictive performance, ensuring that the generated bags explore different regions of the modality-feature space.
The Dynamic Probability Adjustment mechanism continuously updates modality importance scores and dropout probabilities during bag generation, incorporating feedback from diversity metrics to optimize the balance between modality retention and ensemble diversity. Metadata Capture and Storage records all adaptive decisions including modality importance scores, feature selection patterns, dropout rate distributions, and diversity metrics in the BagConfig objects and interpretability data structures, providing complete traceability of the adaptive process. Finally, Cross-Bag Optimization ensures that the ensemble as a whole benefits from the adaptive strategy by considering inter-bag relationships, preventing redundant bag configurations, and optimizing the overall ensemble composition for maximum predictive performance and robustness, creating a novel approach that goes beyond traditional static dropout strategies by incorporating data-driven, dynamic, and diversity-aware modality and feature selection.

//Hyperparameters
1. n_bags
Description: Number of ensemble bags to create - determines the size of the ensemble and affects diversity and computational cost
Range: [3, 5, 10, 15, 20, 25, 30]
Default: 10
Testing: VARY - Key parameter for ensemble size optimization
2. dropout_strategy
Description: Modality dropout strategy for creating diverse bags - includes your novel adaptive strategy that dynamically adjusts based on ensemble diversity
Range: ['linear', 'exponential', 'random', 'adaptive']
Default: 'adaptive' (your novel feature)
Testing: FIXED - Keep 'adaptive' for main experiments, vary in ablation studies
3. max_dropout_rate
Description: Maximum dropout rate for modality removal - controls the upper bound of how many modalities can be dropped from each bag
Range: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
Default: 0.5
Testing: VARY - Affects ensemble diversity and robustness
4. min_modalities
Description: Minimum number of modalities that must remain in each bag - ensures no bag becomes too sparse
Range: [1, 2, 3] (depends on total modalities)
Default: 1
Testing: FIXED - Keep 1 for maximum flexibility
🎯 DIVERSITY OPTIMIZATION HYPERPARAMETERS
5. sample_ratio
Description: Bootstrap sampling ratio for each bag - controls how much of the training data is sampled for each bag
Range: [0.6, 0.7, 0.8, 0.9]
Default: 0.8
Testing: VARY - Affects bag diversity and training data coverage
🎯 FEATURE SAMPLING HYPERPARAMETERS
6. feature_sampling_ratio
Description: Ratio of features to sample within each active modality - enables hierarchical feature sampling within modalities
Range: [0.3, 0.5, 0.7, 0.8, 0.9]
Default: 0.8
Testing: VARY - Novel feature testing
🎯 VALIDATION AND CONTROL HYPERPARAMETERS
7. random_state
Description: Random seed for reproducible ensemble generation and bag creation - set based on test run number during experiments
Range: [42, 123, 456, 789, 1000, ...] (any integer)
Default: 42
Testing: EXPERIMENT-DEPENDENT - Set based on test run number, not varied per test

//Abalation Studies
1. ADAPTIVE MODALITY DROPOUT ABLATION STUDY
Study Name: Adaptive_Modality_Dropout_Ablation
Feature Being Tested:
Adaptive Modality Dropout Strategy - Your novel diversity-driven adaptive dropout that dynamically adjusts dropout rates based on real-time ensemble diversity monitoring, variance-based modality importance computation, inverse proportional dropout calculation, distinct modality combination enforcement, and ensemble diversity maximization
Control (Baseline):
Static Dropout Strategies: linear, exponential, random dropout strategies that use fixed dropout rates without diversity optimization or modality importance consideration
Variables for the Study:
ablation_variables = [
    # Control group (baselines)
    {'name': 'Linear_Baseline', 'dropout_strategy': 'linear'},
    {'name': 'Exponential_Baseline', 'dropout_strategy': 'exponential'},
    {'name': 'Random_Baseline', 'dropout_strategy': 'random'},
    
    # Treatment group (your novel feature)
    {'name': 'Adaptive_Novel', 'dropout_strategy': 'adaptive'},
    
    # Additional adaptive variants for sensitivity analysis
    {'name': 'Adaptive_Low_Dropout', 'dropout_strategy': 'adaptive', 'max_dropout_rate': 0.3},
    {'name': 'Adaptive_High_Dropout', 'dropout_strategy': 'adaptive', 'max_dropout_rate': 0.7},
]
Metrics to Compare:
Ensemble diversity (Hamming distance between modality masks)
Predictive performance (accuracy, F1-score, AUC-ROC)
Modality utilization patterns
Feature selection diversity
Computational efficiency
Robustness to missing modalities
Expected Outcome:
Adaptive strategy should demonstrate superior ensemble diversity, better predictive performance, and more intelligent modality utilization compared to static strategies

2. FEATURE-LEVEL SAMPLING ABLATION STUDY
Study Name: Feature_Level_Sampling_Ablation
Feature Being Tested:
Feature-level sampling within modalities - Your novel two-level sampling system that randomly selects subsets of features within each active modality to create intra-modality diversity
Control (Baseline):
Full feature utilization - All features within active modalities are used without sampling
Variables for the Study:
ablation_variables = [
    # Control group (baseline)
    {'name': 'Full_Features', 'feature_sampling_ratio': 1.0},
    
    # Treatment groups (feature sampling variants)
    {'name': 'Low_Feature_Sampling', 'feature_sampling_ratio': 0.3},
    {'name': 'Medium_Feature_Sampling', 'feature_sampling_ratio': 0.5},
    {'name': 'High_Feature_Sampling', 'feature_sampling_ratio': 0.8},
    {'name': 'Very_High_Feature_Sampling', 'feature_sampling_ratio': 0.9},
]
Metrics to Compare:
Feature utilization diversity across bags
Intra-modality diversity scores
Predictive performance with reduced features
Overfitting resistance
Computational efficiency
Feature importance stability
Expected Outcome:
Feature sampling should improve ensemble diversity and overfitting resistance while maintaining or improving predictive performance

3. COMBINED ADAPTIVE STRATEGY ABLATION STUDY
Study Name: Combined_Adaptive_Strategy_Ablation
Feature Being Tested:
Combined effect of adaptive modality dropout + feature-level sampling - Testing the synergistic impact of both novel features working together
Control (Baseline):
Static strategies with full feature utilization
Variables for the Study:
ablation_variables = [
    # Control group (static + full features)
    {'name': 'Linear_Full_Features', 'dropout_strategy': 'linear', 'feature_sampling_ratio': 1.0},
    {'name': 'Random_Full_Features', 'dropout_strategy': 'random', 'feature_sampling_ratio': 1.0},
    
    # Treatment groups (adaptive + feature sampling combinations)
    {'name': 'Adaptive_Full_Features', 'dropout_strategy': 'adaptive', 'feature_sampling_ratio': 1.0},
    {'name': 'Adaptive_Feature_Sampling', 'dropout_strategy': 'adaptive', 'feature_sampling_ratio': 0.8},
    {'name': 'Linear_Feature_Sampling', 'dropout_strategy': 'linear', 'feature_sampling_ratio': 0.8},
    {'name': 'Random_Feature_Sampling', 'dropout_strategy': 'random', 'feature_sampling_ratio': 0.8},
]
Metrics to Compare:
Overall ensemble diversity (modality + feature level)
Predictive performance across different combinations
Synergistic effects of both novel features
Computational efficiency trade-offs
Robustness to various data conditions
Expected Outcome:
The combination of adaptive modality dropout + feature sampling should demonstrate the strongest performance, showing synergistic benefits of both novel features

//Intepretability Tests
1. Ensemble Diversity Analysis
Test Name: analyze_ensemble_diversity
Description: Analyze ensemble diversity patterns in Stage 2 to understand how different dropout strategies and feature sampling affect ensemble composition and diversity
Implementation:
def analyze_ensemble_diversity(model):
    """Analyze ensemble diversity patterns in Stage 2"""
    ensemble_stats = model.get_ensemble_stats()
    interpretability_data = model.get_stage2_interpretability_data()
    
    # Calculate diversity metrics
    bags = interpretability_data['bags']
    modality_coverage = ensemble_stats['modality_coverage']
    dropout_statistics = ensemble_stats['dropout_statistics']
    
    # Compute Hamming distance between modality masks
    diversity_scores = []
    for i in range(len(bags)):
        for j in range(i+1, len(bags)):
            bag_i_mask = bags[i]['modality_mask']
            bag_j_mask = bags[j]['modality_mask']
            hamming_dist = sum(1 for k in bag_i_mask if bag_i_mask[k] != bag_j_mask[k])
            diversity_scores.append(hamming_dist)
    
    return {
        'ensemble_diversity': np.mean(diversity_scores),
        'mean_bag_diversity': np.mean(diversity_scores),
        'std_bag_diversity': np.std(diversity_scores),
        'diversity_distribution': diversity_scores,
        'modality_coverage': modality_coverage,
        'dropout_statistics': dropout_statistics
    }
Interpretability Questions:
How diverse is the ensemble across all bags?
What's the variance in diversity across bags?
How does diversity change with different dropout strategies?
Which modalities are most/least frequently used?

2. Modality Importance Analysis
Test Name: analyze_modality_importance
Description: Analyze how the adaptive strategy computes and utilizes modality importance scores, and how this affects ensemble composition
Implementation:
def analyze_modality_importance(model):
    """Analyze modality importance computation and utilization"""
    interpretability_data = model.get_stage2_interpretability_data()
    bags = interpretability_data['bags']
    
    # Extract modality importance patterns
    modality_usage = {}
    dropout_rates = []
    
    for bag in bags:
        for modality, is_active in bag['modality_mask'].items():
            if modality not in modality_usage:
                modality_usage[modality] = {'active': 0, 'total': 0}
            modality_usage[modality]['total'] += 1
            if is_active:
                modality_usage[modality]['active'] += 1
        dropout_rates.append(bag['dropout_rate'])
    
    # Calculate importance scores
    importance_scores = {}
    for modality, usage in modality_usage.items():
        importance_scores[modality] = usage['active'] / usage['total']
    
    return {
        'modality_importance_scores': importance_scores,
        'modality_usage_patterns': modality_usage,
        'dropout_rate_distribution': dropout_rates,
        'importance_variance': np.var(list(importance_scores.values()))
    }
Interpretability Questions:
How does the adaptive strategy rank modality importance?
Which modalities are retained most frequently?
How consistent are importance scores across bags?
Does importance ranking match expected data characteristics?

3. Feature Selection Analysis
Test Name: analyze_feature_selection
Description: Analyze feature-level sampling patterns to understand how features are selected within each modality and their impact on ensemble diversity
Implementation:
def analyze_feature_selection(model):
    """Analyze feature selection patterns within modalities"""
    interpretability_data = model.get_stage2_interpretability_data()
    bags = interpretability_data['bags']
    
    # Analyze feature selection patterns
    feature_selection_stats = {}
    
    for bag in bags:
        for modality, feature_mask in bag['feature_mask'].items():
            if modality not in feature_selection_stats:
                feature_selection_stats[modality] = {
                    'total_features': len(feature_mask),
                    'selected_features': [],
                    'selection_ratios': []
                }
            
            selected_count = np.sum(feature_mask)
            selection_ratio = selected_count / len(feature_mask)
            
            feature_selection_stats[modality]['selected_features'].append(selected_count)
            feature_selection_stats[modality]['selection_ratios'].append(selection_ratio)
    
    # Calculate statistics
    for modality in feature_selection_stats:
        stats = feature_selection_stats[modality]
        stats['mean_selection_ratio'] = np.mean(stats['selection_ratios'])
        stats['std_selection_ratio'] = np.std(stats['selection_ratios'])
        stats['min_selection_ratio'] = np.min(stats['selection_ratios'])
        stats['max_selection_ratio'] = np.max(stats['selection_ratios'])
    
    return {
        'feature_selection_statistics': feature_selection_stats,
        'overall_feature_diversity': {
            mod: stats['std_selection_ratio'] for mod, stats in feature_selection_stats.items()
        }
    }
Interpretability Questions:
How many features are typically selected per modality?
What's the variance in feature selection across bags?
Are certain features consistently selected or avoided?
How does feature sampling ratio affect selection patterns?

4. Dropout Strategy Comparison
Test Name: compare_dropout_strategies
Description: Compare the effectiveness of different dropout strategies (linear, exponential, random, adaptive) in terms of diversity, modality utilization, and ensemble composition
Implementation:
def compare_dropout_strategies(models_dict):
    """Compare different dropout strategies across multiple models"""
    comparison_results = {}
    
    for strategy_name, model in models_dict.items():
        ensemble_stats = model.get_ensemble_stats()
        interpretability_data = model.get_stage2_interpretability_data()
        
        # Extract key metrics
        bags = interpretability_data['bags']
        modality_coverage = ensemble_stats['modality_coverage']
        dropout_stats = ensemble_stats['dropout_statistics']
        
        # Calculate strategy-specific metrics
        dropout_rates = [bag['dropout_rate'] for bag in bags]
        modality_usage = {}
        
        for bag in bags:
            for modality, is_active in bag['modality_mask'].items():
                if modality not in modality_usage:
                    modality_usage[modality] = 0
                if is_active:
                    modality_usage[modality] += 1
        
        comparison_results[strategy_name] = {
            'mean_dropout_rate': np.mean(dropout_rates),
            'std_dropout_rate': np.std(dropout_rates),
            'modality_usage': modality_usage,
            'modality_coverage': modality_coverage,
            'strategy': ensemble_stats['strategy']
        }
    
    return {
        'strategy_comparison': comparison_results,
        'best_diversity_strategy': max(comparison_results.keys(), 
                                     key=lambda x: comparison_results[x]['modality_coverage']),
        'most_balanced_strategy': min(comparison_results.keys(),
                                    key=lambda x: comparison_results[x]['std_dropout_rate'])
    }
Interpretability Questions:
Which dropout strategy produces the most diverse ensemble?
How do dropout rates vary across different strategies?
Which strategy provides the most balanced modality utilization?
How does the adaptive strategy compare to static strategies?

5. Bootstrap Sampling Analysis
Test Name: analyze_bootstrap_sampling
Description: Analyze bootstrap sampling patterns to understand how data is distributed across bags and identify potential sampling biases
Implementation:
def analyze_bootstrap_sampling(model):
    """Analyze bootstrap sampling patterns and data distribution"""
    interpretability_data = model.get_stage2_interpretability_data()
    bags = interpretability_data['bags']
    
    # Analyze sampling patterns
    sample_counts = []
    sample_overlap = []
    unique_samples = set()
    
    for bag in bags:
        sample_count = len(bag['data_indices'])
        sample_counts.append(sample_count)
        unique_samples.update(bag['data_indices'])
    
    # Calculate overlap between bags
    for i in range(len(bags)):
        for j in range(i+1, len(bags)):
            overlap = len(set(bags[i]['data_indices']) & set(bags[j]['data_indices']))
            sample_overlap.append(overlap)
    
    return {
        'sample_count_distribution': {
            'mean': np.mean(sample_counts),
            'std': np.std(sample_counts),
            'min': np.min(sample_counts),
            'max': np.max(sample_counts)
        },
        'sample_overlap_statistics': {
            'mean_overlap': np.mean(sample_overlap),
            'std_overlap': np.std(sample_overlap),
            'max_overlap': np.max(sample_overlap)
        },
        'unique_sample_coverage': len(unique_samples),
        'total_samples': len(bags[0]['data_indices']) if bags else 0
    }
Interpretability Questions:
How evenly are samples distributed across bags?
What's the overlap between different bags?
Are there any sampling biases or patterns?
How does sample ratio affect data coverage?

//Robustness Tests
1. Modality Dropout Robustness
Test Name: test_modality_dropout_robustness
Description: Test how well the ensemble generation handles different dropout scenarios including missing modalities, partial modality availability, and extreme dropout rates
Implementation:
def test_modality_dropout_robustness(model, test_scenarios):
    """Test robustness to different modality dropout scenarios."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        # Create modified model with specific dropout configuration
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=scenario_config.get('dropout_strategy', 'adaptive'),
            max_dropout_rate=scenario_config.get('max_dropout_rate', 0.5),
            feature_sampling_ratio=model.feature_sampling_ratio,
            random_state=model.random_state,
            verbose=False
        )
        
        # Test with scenario-specific configuration
        test_model.fit(model.train_data, model.train_labels)
        
        # Collect robustness metrics
        ensemble_stats = test_model.get_ensemble_statistics()
        modality_importance = test_model.get_modality_importance()
        feature_stats = test_model.get_feature_statistics()
        
        results[scenario_name] = {
            'ensemble_stats': ensemble_stats,
            'modality_importance': modality_importance,
            'feature_stats': feature_stats,
            'robustness_score': calculate_robustness_score(ensemble_stats, modality_importance)
        }
    
    return results
Test Scenarios:
Missing Modality Robustness: Test with 1, 2, or all modalities missing
Partial Modality Robustness: Test with reduced feature dimensions per modality
Dropout Rate Sensitivity: Test with extreme dropout rates (0.1, 0.5, 0.9)
Dropout Strategy Robustness: Compare performance across different dropout strategies
Robustness Questions:
How does the ensemble handle missing modalities?
What's the impact of extreme dropout rates on ensemble diversity?
Which dropout strategy is most robust to missing data?
How does feature sampling affect robustness to partial modality availability?

2. Feature Sampling Robustness
Test Name: test_feature_sampling_robustness
Description: Test robustness of the feature-level sampling mechanism under different sampling ratios and feature availability scenarios
Implementation:
def test_feature_sampling_robustness(model, sampling_scenarios):
    """Test robustness of feature sampling under different conditions."""
    results = {}
    
    for scenario_name, sampling_ratio in sampling_scenarios.items():
        # Create model with specific feature sampling ratio
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            max_dropout_rate=model.max_dropout_rate,
            feature_sampling_ratio=sampling_ratio,
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze feature selection patterns
        feature_stats = test_model.get_feature_statistics()
        interpretability_data = test_model.get_stage2_interpretability_data()
        
        # Calculate feature diversity metrics
        feature_diversity = calculate_feature_diversity(interpretability_data)
        
        results[scenario_name] = {
            'sampling_ratio': sampling_ratio,
            'feature_stats': feature_stats,
            'feature_diversity': feature_diversity,
            'robustness_metrics': {
                'selection_consistency': calculate_selection_consistency(feature_stats),
                'diversity_stability': calculate_diversity_stability(feature_diversity)
            }
        }
    
    return results
Test Scenarios:
Low Feature Sampling (0.1, 0.3): Test with minimal feature selection
Medium Feature Sampling (0.5, 0.7): Test with moderate feature selection
High Feature Sampling (0.9, 1.0): Test with near-complete feature selection
Variable Feature Sampling: Test with different ratios per modality
Robustness Questions:
How does feature sampling ratio affect ensemble stability?
What's the minimum feature sampling ratio for robust performance?
Are there optimal feature sampling ratios for different modalities?
How does feature diversity change with sampling ratios?

3. Bootstrap Sampling Robustness
Test Name: test_bootstrap_sampling_robustness
Description: Test robustness of bootstrap sampling under different sample ratios and data distribution scenarios
Implementation:
def test_bootstrap_sampling_robustness(model, sampling_scenarios):
    """Test robustness of bootstrap sampling mechanisms."""
    results = {}
    
    for scenario_name, sample_ratio in sampling_scenarios.items():
        # Create model with specific sample ratio
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            max_dropout_rate=model.max_dropout_rate,
            feature_sampling_ratio=model.feature_sampling_ratio,
            sample_ratio=sample_ratio,
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze bootstrap sampling patterns
        interpretability_data = test_model.get_stage2_interpretability_data()
        detailed_bags = interpretability_data['detailed_bags']
        
        # Calculate sampling robustness metrics
        sampling_metrics = calculate_sampling_robustness(detailed_bags)
        
        results[scenario_name] = {
            'sample_ratio': sample_ratio,
            'sampling_metrics': sampling_metrics,
            'data_coverage': calculate_data_coverage(detailed_bags),
            'robustness_score': calculate_sampling_robustness_score(sampling_metrics)
        }
    
    return results
Test Scenarios:
Low Sample Ratios (0.3, 0.5): Test with minimal data sampling
Medium Sample Ratios (0.7, 0.8): Test with standard data sampling
High Sample Ratios (0.9, 1.0): Test with near-complete data sampling
Variable Sample Ratios: Test with different ratios across bags
Robustness Questions:
How does sample ratio affect data coverage across bags?
What's the minimum sample ratio for robust ensemble generation?
How does bootstrap sampling handle class imbalance?
Are there optimal sample ratios for different data sizes?

4. Ensemble Size Robustness
Test Name: test_ensemble_size_robustness
Description: Test robustness of ensemble generation with different numbers of bags and ensemble sizes
Implementation:
def test_ensemble_size_robustness(model, ensemble_sizes):
    """Test robustness with different ensemble sizes."""
    results = {}
    
    for scenario_name, n_bags in ensemble_sizes.items():
        # Create model with specific ensemble size
        test_model = MultiModalEnsembleModel(
            n_bags=n_bags,
            dropout_strategy=model.dropout_strategy,
            max_dropout_rate=model.max_dropout_rate,
            feature_sampling_ratio=model.feature_sampling_ratio,
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze ensemble size effects
        ensemble_stats = test_model.get_ensemble_statistics()
        interpretability_data = test_model.get_stage2_interpretability_data()
        
        # Calculate ensemble diversity and stability
        ensemble_diversity = calculate_ensemble_diversity(interpretability_data)
        ensemble_stability = calculate_ensemble_stability(ensemble_stats)
        
        results[scenario_name] = {
            'n_bags': n_bags,
            'ensemble_stats': ensemble_stats,
            'ensemble_diversity': ensemble_diversity,
            'ensemble_stability': ensemble_stability,
            'robustness_metrics': {
                'diversity_scaling': calculate_diversity_scaling(ensemble_diversity, n_bags),
                'stability_trend': calculate_stability_trend(ensemble_stability, n_bags)
            }
        }
    
    return results
Test Scenarios:
Small Ensembles (3, 5 bags): Test with minimal ensemble size
Medium Ensembles (10, 15 bags): Test with standard ensemble size
Large Ensembles (25, 50 bags): Test with large ensemble size
Variable Ensemble Sizes: Test with different sizes across runs
Robustness Questions:
How does ensemble size affect diversity and stability?
What's the minimum ensemble size for robust performance?
Are there diminishing returns with larger ensembles?
How does ensemble size interact with dropout strategies?

5. Adaptive Strategy Robustness
Test Name: test_adaptive_strategy_robustness
Description: Test robustness of the novel adaptive dropout strategy under different data conditions and parameter settings
Implementation:
def test_adaptive_strategy_robustness(model, adaptive_scenarios):
    """Test robustness of the adaptive dropout strategy."""
    results = {}
    
    for scenario_name, scenario_config in adaptive_scenarios.items():
        # Create model with adaptive strategy and specific parameters
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy='adaptive',
            max_dropout_rate=scenario_config.get('max_dropout_rate', 0.5),
            feature_sampling_ratio=scenario_config.get('feature_sampling_ratio', 0.8),
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze adaptive strategy performance
        modality_importance = test_model.get_modality_importance()
        ensemble_stats = test_model.get_ensemble_statistics()
        interpretability_data = test_model.get_stage2_interpretability_data()
        
        # Calculate adaptive strategy robustness metrics
        adaptive_metrics = calculate_adaptive_robustness(
            modality_importance, ensemble_stats, interpretability_data
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'modality_importance': modality_importance,
            'ensemble_stats': ensemble_stats,
            'adaptive_metrics': adaptive_metrics,
            'robustness_score': calculate_adaptive_robustness_score(adaptive_metrics)
        }
    
    return results
Test Scenarios:
Adaptive Parameter Sensitivity: Test with different max_dropout_rate values
Feature Sampling Integration: Test adaptive strategy with different feature_sampling_ratio
Data Distribution Robustness: Test with different data distributions
Modality Importance Stability: Test consistency of importance calculations
Robustness Questions:
How robust is the adaptive strategy to parameter changes?
Does the adaptive strategy maintain performance across different data conditions?
How consistent are modality importance calculations?
What's the impact of feature sampling on adaptive strategy robustness?

Stage 3:

//summary
Stage 3 begins with Initialization & Configuration where the BaseLearnerSelector class is instantiated with task type (classification/regression), optimization mode (accuracy/performance/efficiency), number of classes, and random state parameters, establishing the foundation for intelligent learner selection. The Bag Data Processing step receives saved bags from Stage 2 with their sampled training data and labels, accessing bag data through bagger.get_bag_data() to retrieve modality-specific data and labels for each bag, ensuring complete data availability for analysis. Modality Weightage Analysis implements the _analyze_modality_weightage() function that calculates the importance of each modality within each bag by analyzing feature selection ratios (how many features are selected), data variance (importance of the data), and dimensionality (size of the modality), then normalizes these weights to sum to 1.0, providing quantitative measures of modality importance per bag. Learner Type Selection uses the _select_learner_type() function to intelligently assign learner types based on bag characteristics, mapping single modalities to modality-specific learners (text→text_learner, image→image_learner, tabular→tabular_learner), two modalities to simple fusion learners, and multiple modalities to advanced fusion learners, ensuring optimal learner selection for each bag's composition. Learner Configuration implements the _configure_learner() function that customizes learner parameters based on the selected optimization mode, with accuracy mode using high complexity (200 estimators, depth 20, early stopping), performance mode using medium complexity (100 estimators, depth 10), and efficiency mode using low complexity (50 estimators, depth 5), while also adding modality-specific configurations for different learner types. Performance Prediction uses the _predict_performance() function to estimate expected performance for each bag-learner combination by calculating base performance (0.6) plus bonuses for modality diversity, weightage, learner type, and optimization mode, minus penalties for dropout rates, providing performance scores between 0.0 and 1.0. Bag-Learner Configuration Creation generates BagLearnerConfig dataclass objects that store complete bag information including bag ID, sampled training data and labels, modality and feature masks, assigned learner type and configuration, modality weights, optimization mode, task type, and expected performance, creating comprehensive configurations for each bag. Storage & Management stores all bag-learner configurations in self.bag_learner_configs list, providing methods like get_bag_learner_summary() for overall statistics, get_bag_learner_config(bag_id) for individual bag configurations, and get_stage3_interpretability_data() for comprehensive analysis data, enabling easy access to all Stage 3 results. Finally, API Integration seamlessly integrates Stage 3 into the main pipeline through the _select_base_learners() method, automatically running after Stage 2 completion, with all Stage 3 functionality accessible through the unified API, ensuring smooth data flow from Stage 2 bag generation to Stage 3 learner selection, with complete bag data, assigned learners, and configuration metadata ready for Stage 4 training.

//novel features
Stage 3 represents a completely novel approach to ensemble learning that fundamentally transforms how base learners are selected and configured in multimodal ensemble systems. Unlike traditional ensemble methods that use identical learners for all bags or select learners based on performance metrics, this stage introduces intelligent modality-aware learner selection that analyzes the specific composition and characteristics of each bag to assign optimal learners. The novel feature begins with modality weightage analysis that quantifies the importance and contribution of each modality within each individual bag by calculating feature selection ratios, data variance, and dimensionality, then normalizes these weights to create a comprehensive understanding of how different modalities contribute to each bag's learning potential. This weightage analysis enables bag-specific learner type selection where the system intelligently maps modality combinations to optimal learner architectures, assigning single-modality learners for bags with one dominant modality, simple fusion learners for two-modality combinations, and advanced fusion learners for complex multi-modality scenarios, ensuring each bag receives a learner specifically designed for its data characteristics. The system further incorporates optimization mode integration that customizes learner configurations based on performance goals, with accuracy mode prioritizing complex models with high capacity, performance mode balancing complexity and speed, and efficiency mode focusing on fast, lightweight models, allowing the same modality analysis to produce different learner configurations based on deployment requirements. Complete bag-learner pairing stores each bag with its sampled training data, labels, assigned learner configuration, modality weights, and performance predictions in unified BagLearnerConfig objects, creating a comprehensive mapping between data characteristics and optimal learning strategies. This novel approach represents the first documented method that combines modality analysis, bag-specific learner selection, optimization mode integration, and complete configuration storage, creating an intelligent system that adapts its learning strategy to the specific characteristics of each bag's data composition, fundamentally advancing ensemble learning from static, one-size-fits-all approaches to dynamic, modality-aware, bag-specific learner selection that maximizes the potential of each individual bag within the ensemble.

//hyperparameters
🎯 CORE LEARNER SELECTION HYPERPARAMETERS
1. task_type
Description: Task type for learner selection - determines whether to use classification or regression learners
Range: ['classification', 'regression']
Default: 'classification'
Testing: FIXED - Determined by data analysis
2. optimization_mode
Description: Optimization strategy for learner configuration - controls complexity and performance trade-offs
Range: ['accuracy', 'performance', 'efficiency']
Default: 'accuracy'
Testing: VARY - Key parameter for learner selection strategy
3. n_classes
Description: Number of classes for classification tasks - affects learner architecture selection
Range: [2, 3, 4, 5, 10, 20, 50, 100]
Default: 2
Testing: FIXED - Determined by data analysis
🎯 ABLATION STUDY HYPERPARAMETERS
4. modality_aware
Description: Enable modality-aware learner selection - your novel feature vs fixed selection
Range: [True, False]
Default: True
Testing: VARY - Key parameter for ablation studies
5. bag_learner_pairing
Description: Enable complete bag-learner pairing storage vs separate storage
Range: [True, False]
Default: True
Testing: VARY - Affects pairing quality and storage efficiency
6. metadata_level
Description: Level of metadata storage for bag-learner configurations
Range: ['minimal', 'complete', 'enhanced']
Default: 'complete'
Testing: VARY - Affects metadata completeness
7. pairing_focus
Description: Focus for pairing optimization strategy
Range: ['performance', 'diversity', 'efficiency']
Default: 'performance'
Testing: VARY - Affects pairing optimization
�� MODALITY WEIGHTAGE ANALYSIS HYPERPARAMETERS
8. feature_ratio_weight
Description: Weight for feature selection ratio in modality importance calculation
Range: [0.2, 0.3, 0.4, 0.5]
Default: 0.4
Testing: VARY - Affects modality weightage calculation
9. variance_weight
Description: Weight for data variance in modality importance calculation
Range: [0.2, 0.3, 0.4, 0.5]
Default: 0.3
Testing: VARY - Affects modality weightage calculation
10. dimensionality_weight
Description: Weight for dimensionality in modality importance calculation
Range: [0.2, 0.3, 0.4, 0.5]
Default: 0.3
Testing: VARY - Affects modality weightage calculation
🎯 PERFORMANCE PREDICTION HYPERPARAMETERS
11. base_performance
Description: Base performance score for learner performance prediction
Range: [0.5, 0.6, 0.7, 0.8]
Default: 0.6
Testing: FIXED - Keep 0.6 for standard baseline
12. diversity_bonus
Description: Bonus multiplier for modality diversity in performance prediction
Range: [0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: VARY - Affects performance prediction accuracy
13. weightage_bonus
Description: Bonus multiplier for modality weightage in performance prediction
Range: [0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: VARY - Affects performance prediction accuracy
14. dropout_penalty
Description: Penalty multiplier for dropout rate in performance prediction
Range: [0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: VARY - Affects performance prediction accuracy
🎯 LEARNER TYPE SELECTION HYPERPARAMETERS
15. single_modality_threshold
Description: Threshold for single modality learner selection
Range: [1, 2, 3]
Default: 1
Testing: FIXED - Keep 1 for single modality detection
16. fusion_threshold
Description: Threshold for fusion learner selection
Range: [2, 3, 4, 5]
Default: 2
Testing: VARY - Affects learner type selection
17. advanced_fusion_threshold
Description: Threshold for advanced fusion learner selection
Range: [3, 4, 5, 6]
Default: 3
Testing: VARY - Affects learner type selection
🎯 VALIDATION AND CONTROL HYPERPARAMETERS
18. random_state
Description: Random seed for reproducible learner selection and configuration
Range: [42, 123, 456, 789, 1000, ...] (any integer)
Default: 42
Testing: EXPERIMENT-DEPENDENT - Set based on test run number, not varied per test
19. verbose
Description: Enables detailed learner selection progress logging
Range: [True, False]
Default: True
Testing: FIXED - Keep True for monitoring

//abalation studies
1. MODALITY-AWARE LEARNER SELECTION ABLATION STUDY
Study Name: Modality_Aware_Learner_Selection_Ablation
Feature Being Tested:
Modality-Aware Bag-Specific Learner Selection - Your novel adaptive learner selection system that analyzes modality patterns in each bag and selects optimal base learners based on the specific modality composition, rather than using a one-size-fits-all approach
Control (Baseline):
Fixed Learner Selection: optimization_mode='fixed' with modality_aware=False - Traditional ensemble methods that use the same learner type for all bags regardless of modality patterns
Variables for the Study:
ablation_variables = [
    # Control group (baselines)
    {'name': 'Fixed_Neural_Network', 'optimization_mode': 'fixed', 'modality_aware': False, 'default_learner_type': 'neural_network'},
    {'name': 'Fixed_Tree_Based', 'optimization_mode': 'fixed', 'modality_aware': False, 'default_learner_type': 'tree_based'},
    {'name': 'Fixed_Linear', 'optimization_mode': 'fixed', 'modality_aware': False, 'default_learner_type': 'linear'},
    {'name': 'Random_Selection', 'optimization_mode': 'random', 'modality_aware': False},
    
    # Treatment group (your novel feature)
    {'name': 'Modality_Aware_Adaptive', 'optimization_mode': 'adaptive', 'modality_aware': True},
    
    # Additional modality-aware variants
    {'name': 'Modality_Aware_Low_Diversity', 'optimization_mode': 'adaptive', 'modality_aware': True, 'learner_diversity_weight': 0.1},
    {'name': 'Modality_Aware_High_Diversity', 'optimization_mode': 'adaptive', 'modality_aware': True, 'learner_diversity_weight': 0.5},
]
Metrics to Compare:
Ensemble performance (accuracy, F1-score, AUC-ROC)
Learner diversity across bags
Modality utilization efficiency
Training time and computational cost
Bag-specific performance variance
Expected Outcome:
Modality-aware adaptive selection should outperform fixed selection by 5-15% in performance while maintaining or improving ensemble diversity

2. OPTIMIZATION MODE IMPACT ABLATION STUDY
Study Name: Optimization_Mode_Impact_Ablation
Feature Being Tested:
Optimization Mode Selection - Your novel system that adapts learner configuration based on optimization goals (accuracy, performance, efficiency) rather than using uniform configuration
Control (Baseline):
Uniform Configuration: optimization_mode='uniform' - Traditional ensemble methods that use the same configuration for all learners regardless of optimization goals
Variables for the Study:
ablation_variables = [
    # Control group (baselines)
    {'name': 'Uniform_Configuration', 'optimization_mode': 'uniform', 'adaptive_config': False},
    {'name': 'Fixed_Accuracy', 'optimization_mode': 'accuracy', 'adaptive_config': False},
    {'name': 'Fixed_Performance', 'optimization_mode': 'performance', 'adaptive_config': False},
    {'name': 'Fixed_Efficiency', 'optimization_mode': 'efficiency', 'adaptive_config': False},
    
    # Treatment group (your novel feature)
    {'name': 'Adaptive_Accuracy', 'optimization_mode': 'accuracy', 'adaptive_config': True},
    {'name': 'Adaptive_Performance', 'optimization_mode': 'performance', 'adaptive_config': True},
    {'name': 'Adaptive_Efficiency', 'optimization_mode': 'efficiency', 'adaptive_config': True},
    
    # Mixed optimization modes
    {'name': 'Mixed_Accuracy_Performance', 'optimization_mode': 'mixed', 'adaptive_config': True, 'mode_weights': {'accuracy': 0.6, 'performance': 0.4}},
    {'name': 'Mixed_Performance_Efficiency', 'optimization_mode': 'mixed', 'adaptive_config': True, 'mode_weights': {'performance': 0.6, 'efficiency': 0.4}},
]
Metrics to Compare:
Task-specific performance (accuracy for classification, MSE for regression)
Computational efficiency (training time, memory usage)
Model complexity (number of parameters, inference time)
Configuration diversity across bags
Optimization goal achievement
Expected Outcome:
Adaptive configuration should achieve 10-20% better performance in the target optimization mode while maintaining reasonable efficiency

3. MODALITY WEIGHTAGE ANALYSIS ABLATION STUDY
Study Name: Modality_Weightage_Analysis_Ablation
Feature Being Tested:
Modality Weightage Analysis - Your novel system that calculates importance scores for each modality in each bag and uses them for learner selection, rather than treating all modalities equally
Control (Baseline):
Equal Modality Treatment: modality_weightage='equal' - Traditional ensemble methods that treat all modalities with equal importance regardless of their actual contribution
Variables for the Study:
ablation_variables = [
    # Control group (baselines)
    {'name': 'Equal_Modality_Weightage', 'modality_weightage': 'equal', 'weightage_analysis': False},
    {'name': 'Random_Modality_Weightage', 'modality_weightage': 'random', 'weightage_analysis': False},
    {'name': 'Fixed_Modality_Weightage', 'modality_weightage': 'fixed', 'weightage_analysis': False, 'fixed_weights': {'text': 0.4, 'image': 0.3, 'tabular': 0.3}},
    
    # Treatment group (your novel feature)
    {'name': 'Variance_Based_Weightage', 'modality_weightage': 'variance', 'weightage_analysis': True, 'weightage_method': 'variance'},
    {'name': 'Feature_Based_Weightage', 'modality_weightage': 'feature', 'weightage_analysis': True, 'weightage_method': 'feature'},
    {'name': 'Combined_Weightage', 'modality_weightage': 'combined', 'weightage_analysis': True, 'weightage_method': 'combined'},
    
    # Weightage analysis variants
    {'name': 'High_Variance_Weight', 'modality_weightage': 'variance', 'weightage_analysis': True, 'variance_weight': 0.6},
    {'name': 'High_Feature_Weight', 'modality_weightage': 'feature', 'weightage_analysis': True, 'feature_weight': 0.6},
    {'name': 'High_Dimensionality_Weight', 'modality_weightage': 'dimensionality', 'weightage_analysis': True, 'dimensionality_weight': 0.6},
]
Metrics to Compare:
Modality importance accuracy (correlation with actual performance contribution)
Learner selection quality (how well selected learners match modality patterns)
Ensemble diversity (variety of learner types across bags)
Performance improvement (accuracy, F1-score, AUC-ROC)
Weightage stability across different datasets
Expected Outcome:
Modality weightage analysis should improve learner selection quality by 15-25% and lead to 5-10% better ensemble performance

4. BAG-LEARNER PAIRING ABLATION STUDY
Study Name: Bag_Learner_Pairing_Ablation
Feature Being Tested:
Bag-Learner Pairing Storage - Your novel system that stores complete bag-learner configurations with metadata, rather than just storing learners separately from bags
Control (Baseline):
Separate Storage: bag_learner_pairing=False - Traditional ensemble methods that store bags and learners separately without maintaining their relationships
Variables for the Study:
ablation_variables = [
    # Control group (baselines)
    {'name': 'Separate_Storage', 'bag_learner_pairing': False, 'metadata_storage': False},
    {'name': 'Basic_Pairing', 'bag_learner_pairing': True, 'metadata_storage': False},
    {'name': 'Minimal_Metadata', 'bag_learner_pairing': True, 'metadata_storage': True, 'metadata_level': 'minimal'},
    
    # Treatment group (your novel feature)
    {'name': 'Complete_Pairing', 'bag_learner_pairing': True, 'metadata_storage': True, 'metadata_level': 'complete'},
    {'name': 'Enhanced_Pairing', 'bag_learner_pairing': True, 'metadata_storage': True, 'metadata_level': 'enhanced'},
    
    # Pairing variants
    {'name': 'Performance_Focused_Pairing', 'bag_learner_pairing': True, 'metadata_storage': True, 'pairing_focus': 'performance'},
    {'name': 'Diversity_Focused_Pairing', 'bag_learner_pairing': True, 'metadata_storage': True, 'pairing_focus': 'diversity'},
    {'name': 'Efficiency_Focused_Pairing', 'bag_learner_pairing': True, 'metadata_storage': True, 'pairing_focus': 'efficiency'},
]
Metrics to Compare:
Bag-learner relationship accuracy (how well stored relationships match actual performance)
Metadata completeness (coverage of important bag characteristics)
Storage efficiency (memory usage, access time)
Ensemble coherence (consistency of bag-learner assignments)
Performance prediction accuracy (how well stored metadata predicts performance)
Expected Outcome:
Complete bag-learner pairing should improve ensemble coherence by 20-30% and lead to 3-8% better performance prediction accuracy

//intepretablity tests
1. Modality Importance Analysis
Description: Analyze how different modalities contribute to learner selection decisions
Implementation:
def test_modality_importance_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Test interpretability of modality importance in learner selection."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            # Create test model with specific modality configurations
            test_model = MultiModalEnsembleModel(
                n_bags=self.n_bags,
                dropout_strategy=self.dropout_strategy,
                optimization_mode=self.optimization_mode,
                modality_aware=self.modality_aware,
                feature_ratio_weight=scenario_config.get('feature_ratio_weight', 0.4),
                variance_weight=scenario_config.get('variance_weight', 0.3),
                dimensionality_weight=scenario_config.get('dimensionality_weight', 0.3),
                random_state=self.random_state,
                verbose=False
            )
            
            test_model.fit(self.train_data, self.train_labels)
            
            # Analyze modality importance
            bag_learner_summary = test_model.get_bag_learner_summary()
            stage3_interpretability = test_model.get_stage3_interpretability_data()
            
            # Calculate modality importance metrics
            modality_importance = self._calculate_modality_importance_interpretability(
                bag_learner_summary, stage3_interpretability
            )
            
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'modality_importance': modality_importance,
                'interpretability_score': self._calculate_modality_importance_interpretability_score(modality_importance),
                'success': True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'error': str(e),
                'success': False
            }
    
    return results
Test Scenarios:
Single Modality Dominance: Test with one modality having much higher importance
Balanced Modalities: Test with equal importance across all modalities
Modality Interaction: Test how modality combinations affect importance
Interpretability Questions:
How does modality importance correlate with learner selection?
Are high-importance modalities consistently assigned appropriate learners?
How does modality interaction affect selection decisions?

2. Learner Selection Decision Analysis
Description: Understand the decision-making process for learner selection
Implementation:
def test_learner_selection_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Test interpretability of learner selection decisions."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            # Create test model with specific selection parameters
            test_model = MultiModalEnsembleModel(
                n_bags=self.n_bags,
                dropout_strategy=self.dropout_strategy,
                optimization_mode=scenario_config.get('optimization_mode', 'accuracy'),
                modality_aware=scenario_config.get('modality_aware', True),
                bag_learner_pairing=scenario_config.get('bag_learner_pairing', True),
                random_state=self.random_state,
                verbose=False
            )
            
            test_model.fit(self.train_data, self.train_labels)
            
            # Analyze learner selection decisions
            bag_learner_summary = test_model.get_bag_learner_summary()
            pairing_stats = test_model.get_pairing_statistics()
            ensemble_coherence = test_model.get_ensemble_coherence()
            
            # Calculate selection decision metrics
            selection_decisions = self._calculate_learner_selection_interpretability(
                bag_learner_summary, pairing_stats, ensemble_coherence
            )
            
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'selection_decisions': selection_decisions,
                'interpretability_score': self._calculate_learner_selection_interpretability_score(selection_decisions),
                'success': True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'error': str(e),
                'success': False
            }
    
    return results
Test Scenarios:
Modality-Learner Mapping: Test how different modality combinations map to learner types
Optimization Mode Impact: Test how optimization mode affects selection decisions
Bag Characteristics: Test how bag properties influence learner selection
Interpretability Questions:
What factors drive learner selection decisions?
How consistent are selection decisions across similar bags?
Can we predict learner selection based on bag characteristics?

3. Performance Prediction Interpretability
Description: Understand how performance predictions are made and their accuracy
Implementation:
def test_performance_prediction_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Test interpretability of performance prediction system."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            # Create test model with specific prediction parameters
            test_model = MultiModalEnsembleModel(
                n_bags=self.n_bags,
                dropout_strategy=self.dropout_strategy,
                optimization_mode=self.optimization_mode,
                modality_aware=self.modality_aware,
                base_performance=scenario_config.get('base_performance', 0.6),
                diversity_bonus=scenario_config.get('diversity_bonus', 0.1),
                weightage_bonus=scenario_config.get('weightage_bonus', 0.1),
                dropout_penalty=scenario_config.get('dropout_penalty', 0.1),
                random_state=self.random_state,
                verbose=False
            )
            
            test_model.fit(self.train_data, self.train_labels)
            
            # Analyze performance prediction system
            bag_learner_summary = test_model.get_bag_learner_summary()
            stage3_interpretability = test_model.get_stage3_interpretability_data()
            
            # Calculate performance prediction interpretability
            prediction_interpretability = self._calculate_performance_prediction_interpretability(
                bag_learner_summary, stage3_interpretability
            )
            
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'prediction_interpretability': prediction_interpretability,
                'interpretability_score': self._calculate_performance_prediction_interpretability_score(prediction_interpretability),
                'success': True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'error': str(e),
                'success': False
            }
    
    return results
Test Scenarios:
Prediction Accuracy: Test how well predictions match actual performance
Prediction Consistency: Test consistency of predictions across similar bags
Prediction Calibration: Test if predictions are well-calibrated
Interpretability Questions:
How accurate are performance predictions?
What factors contribute most to performance predictions?
Are predictions well-calibrated across different scenarios?

4. Bag-Learner Pairing Interpretability
Description: Understand the quality and consistency of bag-learner pairings
Implementation:
def test_bag_learner_pairing_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Test interpretability of bag-learner pairing system."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            # Create test model with specific pairing configuration
            test_model = MultiModalEnsembleModel(
                n_bags=self.n_bags,
                dropout_strategy=self.dropout_strategy,
                optimization_mode=self.optimization_mode,
                modality_aware=self.modality_aware,
                bag_learner_pairing=scenario_config.get('bag_learner_pairing', True),
                metadata_level=scenario_config.get('metadata_level', 'complete'),
                pairing_focus=scenario_config.get('pairing_focus', 'performance'),
                random_state=self.random_state,
                verbose=False
            )
            
            test_model.fit(self.train_data, self.train_labels)
            
            # Analyze bag-learner pairing system
            pairing_stats = test_model.get_pairing_statistics()
            metadata_completeness = test_model.get_metadata_completeness()
            ensemble_coherence = test_model.get_ensemble_coherence()
            
            # Calculate pairing interpretability
            pairing_interpretability = self._calculate_bag_learner_pairing_interpretability(
                pairing_stats, metadata_completeness, ensemble_coherence
            )
            
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'pairing_interpretability': pairing_interpretability,
                'interpretability_score': self._calculate_bag_learner_pairing_interpretability_score(pairing_interpretability),
                'success': True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'error': str(e),
                'success': False
            }
    
    return results
Test Scenarios:
Pairing Quality: Test quality of bag-learner pairings
Pairing Consistency: Test consistency across different pairing configurations
Metadata Impact: Test how metadata completeness affects pairing quality
Interpretability Questions:
How well do bag-learner pairings match expected relationships?
What factors contribute to high-quality pairings?
How does metadata completeness affect pairing decisions?

5. Ensemble Coherence Interpretability
Description: Understand how coherent and consistent the ensemble is
Implementation:
def test_ensemble_coherence_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Test interpretability of ensemble coherence and consistency."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            # Create test model with specific ensemble configuration
            test_model = MultiModalEnsembleModel(
                n_bags=scenario_config.get('n_bags', 10),
                dropout_strategy=self.dropout_strategy,
                optimization_mode=self.optimization_mode,
                modality_aware=self.modality_aware,
                bag_learner_pairing=self.bag_learner_pairing,
                random_state=self.random_state,
                verbose=False
            )
            
            test_model.fit(self.train_data, self.train_labels)
            
            # Analyze ensemble coherence
            bag_learner_summary = test_model.get_bag_learner_summary()
            pairing_stats = test_model.get_pairing_statistics()
            ensemble_coherence = test_model.get_ensemble_coherence()
            
            # Calculate ensemble coherence interpretability
            coherence_interpretability = self._calculate_ensemble_coherence_interpretability(
                bag_learner_summary, pairing_stats, ensemble_coherence
            )
            
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'coherence_interpretability': coherence_interpretability,
                'interpretability_score': self._calculate_ensemble_coherence_interpretability_score(coherence_interpretability),
                'success': True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'error': str(e),
                'success': False
            }
    
    return results
Test Scenarios:
Ensemble Size Impact: Test how ensemble size affects coherence
Learner Diversity: Test how learner diversity affects coherence
Modality Distribution: Test how modality distribution affects coherence
Interpretability Questions:
How coherent is the ensemble across different configurations?
What factors contribute to ensemble coherence?
How does ensemble size affect coherence and interpretability?

6. Optimization Mode Interpretability
Description: Understand how different optimization modes affect the system
Implementation:
def test_optimization_mode_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Test interpretability of optimization mode selection."""
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            # Create test model with specific optimization mode
            test_model = MultiModalEnsembleModel(
                n_bags=self.n_bags,
                dropout_strategy=self.dropout_strategy,
                optimization_mode=scenario_config.get('optimization_mode', 'accuracy'),
                modality_aware=self.modality_aware,
                random_state=self.random_state,
                verbose=False
            )
            
            test_model.fit(self.train_data, self.train_labels)
            
            # Analyze optimization mode impact
            bag_learner_summary = test_model.get_bag_learner_summary()
            ensemble_coherence = test_model.get_ensemble_coherence()
            stage3_interpretability = test_model.get_stage3_interpretability_data()
            
            # Calculate optimization mode interpretability
            optimization_interpretability = self._calculate_optimization_mode_interpretability(
                bag_learner_summary, ensemble_coherence, stage3_interpretability
            )
            
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'optimization_interpretability': optimization_interpretability,
                'interpretability_score': self._calculate_optimization_mode_interpretability_score(optimization_interpretability),
                'success': True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'scenario_config': scenario_config,
                'error': str(e),
                'success': False
            }
    
    return results
Test Scenarios:
Accuracy Mode: Test interpretability under accuracy optimization
Performance Mode: Test interpretability under performance optimization
Efficiency Mode: Test interpretability under efficiency optimization
Interpretability Questions:
How do different optimization modes affect interpretability?
What trade-offs exist between optimization and interpretability?
Which optimization mode provides the most interpretable results?


//robustness tests
1. Modality-Aware Selection Robustness
Test Name: test_modality_aware_robustness
Description: Test robustness of the novel modality-aware learner selection under different modality patterns and edge cases
Implementation:
def test_modality_aware_robustness(model, modality_scenarios):
    """Test robustness of the modality-aware learner selection."""
    results = {}
    
    for scenario_name, scenario_config in modality_scenarios.items():
        # Create model with modality-aware selection and specific parameters
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            optimization_mode=scenario_config.get('optimization_mode', 'accuracy'),
            modality_aware=scenario_config.get('modality_aware', True),
            bag_learner_pairing=scenario_config.get('bag_learner_pairing', True),
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze modality-aware selection performance
        bag_learner_summary = test_model.get_bag_learner_summary()
        pairing_stats = test_model.get_pairing_statistics()
        ensemble_coherence = test_model.get_ensemble_coherence()
        
        # Calculate modality-aware robustness metrics
        modality_metrics = calculate_modality_aware_robustness(
            bag_learner_summary, pairing_stats, ensemble_coherence
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'bag_learner_summary': bag_learner_summary,
            'pairing_statistics': pairing_stats,
            'ensemble_coherence': ensemble_coherence,
            'modality_metrics': modality_metrics,
            'robustness_score': calculate_modality_aware_robustness_score(modality_metrics)
        }
    
    return results
Test Scenarios:
Modality Pattern Variation: Test with different combinations of modalities (single, dual, multi-modal)
Modality Imbalance: Test with highly imbalanced modality distributions
Missing Modality Handling: Test robustness when certain modalities are completely absent
Modality Quality Variation: Test with different quality levels of modality data
Robustness Questions:
How robust is the modality-aware selection to different modality patterns?
Does the modality-aware selection maintain performance across different data conditions?
How consistent are modality-learner assignments across different scenarios?
What's the impact of modality imbalance on selection quality?

2. Optimization Mode Robustness
Test Name: test_optimization_mode_robustness
Description: Test robustness of the optimization mode selection under different optimization strategies and their consistency
Implementation:
def test_optimization_mode_robustness(model, optimization_scenarios):
    """Test robustness of the optimization mode selection."""
    results = {}
    
    for scenario_name, scenario_config in optimization_scenarios.items():
        # Create model with specific optimization mode
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            optimization_mode=scenario_config.get('optimization_mode', 'accuracy'),
            modality_aware=scenario_config.get('modality_aware', True),
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze optimization mode performance
        bag_learner_summary = test_model.get_bag_learner_summary()
        ensemble_coherence = test_model.get_ensemble_coherence()
        stage3_interpretability = test_model.get_stage3_interpretability_data()
        
        # Calculate optimization mode robustness metrics
        optimization_metrics = calculate_optimization_mode_robustness(
            bag_learner_summary, ensemble_coherence, stage3_interpretability
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'bag_learner_summary': bag_learner_summary,
            'ensemble_coherence': ensemble_coherence,
            'stage3_interpretability': stage3_interpretability,
            'optimization_metrics': optimization_metrics,
            'robustness_score': calculate_optimization_mode_robustness_score(optimization_metrics)
        }
    
    return results
Test Scenarios:
Strategy Comparison: Test with accuracy, performance, efficiency optimization modes
Strategy Performance: Compare performance across different optimization strategies
Strategy Consistency: Verify consistent learner selection within same strategy
Strategy Adaptation: Test if strategy changes affect learner selection patterns
Robustness Questions:
How robust is the optimization mode selection to different strategies?
Does the optimization mode maintain performance across different data conditions?
How consistent are learner configurations within the same optimization mode?
What's the impact of optimization mode changes on learner selection patterns?

3. Modality Weightage Analysis Robustness
Test Name: test_modality_weightage_robustness
Description: Test robustness of the modality weightage calculation under different data characteristics and weight configurations
Implementation:
def test_modality_weightage_robustness(model, weightage_scenarios):
    """Test robustness of the modality weightage analysis."""
    results = {}
    
    for scenario_name, scenario_config in weightage_scenarios.items():
        # Create model with specific weightage parameters
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            optimization_mode=model.optimization_mode,
            modality_aware=model.modality_aware,
            feature_ratio_weight=scenario_config.get('feature_ratio_weight', 0.4),
            variance_weight=scenario_config.get('variance_weight', 0.3),
            dimensionality_weight=scenario_config.get('dimensionality_weight', 0.3),
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze weightage analysis performance
        bag_learner_summary = test_model.get_bag_learner_summary()
        stage3_interpretability = test_model.get_stage3_interpretability_data()
        
        # Calculate weightage analysis robustness metrics
        weightage_metrics = calculate_weightage_analysis_robustness(
            bag_learner_summary, stage3_interpretability
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'bag_learner_summary': bag_learner_summary,
            'stage3_interpretability': stage3_interpretability,
            'weightage_metrics': weightage_metrics,
            'robustness_score': calculate_weightage_analysis_robustness_score(weightage_metrics)
        }
    
    return results
Test Scenarios:
Weight Distribution Variation: Test with different weight combinations for feature_ratio, variance, dimensionality
Data Variance Robustness: Test with high/low variance data across modalities
Dimensionality Robustness: Test with different feature dimensions per modality
Weight Sensitivity: Test sensitivity to weight parameter changes
Robustness Questions:
How robust is the weightage analysis to different weight configurations?
Does the weightage analysis maintain accuracy across different data characteristics?
How sensitive are modality importance calculations to weight parameter changes?
What's the impact of data variance and dimensionality on weightage analysis?

4. Bag-Learner Pairing Robustness
Test Name: test_bag_learner_pairing_robustness
Description: Test robustness of the bag-learner pairing system under different storage and metadata scenarios
Implementation:
def test_bag_learner_pairing_robustness(model, pairing_scenarios):
    """Test robustness of the bag-learner pairing system."""
    results = {}
    
    for scenario_name, scenario_config in pairing_scenarios.items():
        # Create model with specific pairing configuration
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            optimization_mode=model.optimization_mode,
            modality_aware=model.modality_aware,
            bag_learner_pairing=scenario_config.get('bag_learner_pairing', True),
            metadata_level=scenario_config.get('metadata_level', 'complete'),
            pairing_focus=scenario_config.get('pairing_focus', 'performance'),
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze pairing system performance
        pairing_stats = test_model.get_pairing_statistics()
        metadata_completeness = test_model.get_metadata_completeness()
        ensemble_coherence = test_model.get_ensemble_coherence()
        
        # Calculate pairing system robustness metrics
        pairing_metrics = calculate_pairing_system_robustness(
            pairing_stats, metadata_completeness, ensemble_coherence
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'pairing_statistics': pairing_stats,
            'metadata_completeness': metadata_completeness,
            'ensemble_coherence': ensemble_coherence,
            'pairing_metrics': pairing_metrics,
            'robustness_score': calculate_pairing_system_robustness_score(pairing_metrics)
        }
    
    return results
Test Scenarios:
Pairing Storage Robustness: Test with different pairing storage configurations
Metadata Level Robustness: Test with minimal, complete, enhanced metadata levels
Pairing Focus Robustness: Test with performance, diversity, efficiency pairing focuses
Pairing Consistency: Verify consistent pairing quality across different scenarios
Robustness Questions:
How robust is the pairing system to different storage configurations?
Does the pairing system maintain quality across different metadata levels?
How consistent are bag-learner relationships across different pairing focuses?
What's the impact of metadata completeness on pairing quality?

5. Performance Prediction Robustness
Test Name: test_performance_prediction_robustness
Description: Test robustness of the performance prediction system under different prediction scenarios and parameter configurations
Implementation:
def test_performance_prediction_robustness(model, prediction_scenarios):
    """Test robustness of the performance prediction system."""
    results = {}
    
    for scenario_name, scenario_config in prediction_scenarios.items():
        # Create model with specific prediction parameters
        test_model = MultiModalEnsembleModel(
            n_bags=model.n_bags,
            dropout_strategy=model.dropout_strategy,
            optimization_mode=model.optimization_mode,
            modality_aware=model.modality_aware,
            base_performance=scenario_config.get('base_performance', 0.6),
            diversity_bonus=scenario_config.get('diversity_bonus', 0.1),
            weightage_bonus=scenario_config.get('weightage_bonus', 0.1),
            dropout_penalty=scenario_config.get('dropout_penalty', 0.1),
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze performance prediction system
        bag_learner_summary = test_model.get_bag_learner_summary()
        stage3_interpretability = test_model.get_stage3_interpretability_data()
        
        # Calculate performance prediction robustness metrics
        prediction_metrics = calculate_performance_prediction_robustness(
            bag_learner_summary, stage3_interpretability
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'bag_learner_summary': bag_learner_summary,
            'stage3_interpretability': stage3_interpretability,
            'prediction_metrics': prediction_metrics,
            'robustness_score': calculate_performance_prediction_robustness_score(prediction_metrics)
        }
    
    return results
Test Scenarios:
Prediction Parameter Variation: Test with different base_performance, diversity_bonus, weightage_bonus, dropout_penalty values
Prediction Accuracy: Test prediction accuracy across different bag configurations
Prediction Stability: Test prediction stability across multiple runs
Prediction Calibration: Test if predictions are well-calibrated with actual performance
Robustness Questions:
How robust is the performance prediction to parameter changes?
Does the performance prediction maintain accuracy across different bag configurations?
How stable are performance predictions across multiple runs?
What's the impact of prediction parameters on prediction quality?

6. Ensemble Size Robustness
Test Name: test_ensemble_size_robustness
Description: Test robustness of the learner selection system under different ensemble sizes and scaling scenarios
Implementation:
def test_ensemble_size_robustness(model, size_scenarios):
    """Test robustness of the learner selection system across different ensemble sizes."""
    results = {}
    
    for scenario_name, scenario_config in size_scenarios.items():
        # Create model with specific ensemble size
        test_model = MultiModalEnsembleModel(
            n_bags=scenario_config.get('n_bags', 10),
            dropout_strategy=model.dropout_strategy,
            optimization_mode=model.optimization_mode,
            modality_aware=model.modality_aware,
            bag_learner_pairing=model.bag_learner_pairing,
            random_state=model.random_state,
            verbose=False
        )
        
        test_model.fit(model.train_data, model.train_labels)
        
        # Analyze ensemble size impact
        bag_learner_summary = test_model.get_bag_learner_summary()
        pairing_stats = test_model.get_pairing_statistics()
        ensemble_coherence = test_model.get_ensemble_coherence()
        
        # Calculate ensemble size robustness metrics
        size_metrics = calculate_ensemble_size_robustness(
            bag_learner_summary, pairing_stats, ensemble_coherence
        )
        
        results[scenario_name] = {
            'scenario_config': scenario_config,
            'bag_learner_summary': bag_learner_summary,
            'pairing_statistics': pairing_stats,
            'ensemble_coherence': ensemble_coherence,
            'size_metrics': size_metrics,
            'robustness_score': calculate_ensemble_size_robustness_score(size_metrics)
        }
    
    return results
Test Scenarios:
Small Ensemble Robustness: Test with 3-5 bags
Medium Ensemble Robustness: Test with 10-20 bags
Large Ensemble Robustness: Test with 50-100 bags
Scaling Analysis: Analyze how learner selection quality scales with ensemble size
Robustness Questions:
How robust is the learner selection to different ensemble sizes?
Does the learner selection maintain quality across different scaling scenarios?
How does ensemble size affect learner selection consistency?
What's the impact of ensemble size on computational efficiency?

Stage 4 

//summary
Stage 4: Base Learner Training orchestrates the training of all weak learners assigned to ensemble bags through a sophisticated multi-step pipeline. The stage begins by creating an EnsembleTrainingPipeline with an AdvancedTrainingConfig that specifies training parameters including epochs, batch size, learning rate, weight decay, optimizer type (AdamW, SGD, Adam), and scheduler type (cosine annealing, one-cycle, reduce-on-plateau). The core train_ensemble() method iterates through each bag-learner configuration, creating specialized learner instances (TabularLearner, TextLearner, ImageLearner, FusionLearner) based on the assigned learner type from Stage 3. For each learner, the training process implements comprehensive overfitting prevention through early stopping with configurable patience, weight decay (L2 regularization), learning rate scheduling with warm restarts, gradient clipping to prevent exploding gradients, and validation-based training with configurable validation splits. The stage also includes advanced features like data augmentation (modality-specific text augmentation, image transformations, audio noise injection, and tabular feature perturbation), batch normalization for stable training, cross-validation with K-fold splits for robust evaluation, gradient accumulation for effective large batch training, progressive learning with curriculum stages, and comprehensive metrics tracking including training/validation loss, accuracy, and performance metrics. Each trained learner is wrapped in a TrainedLearnerInfo dataclass containing the trained model, modality weights, training metrics, and final performance score, with all results stored in memory for Stage 5 integration. The stage ensures ensemble diversity through the varied bag configurations from Stage 2 while maintaining training quality through multiple overfitting prevention mechanisms and comprehensive performance monitoring.

//novel features
1. Advanced Cross-Modal Denoising Loss
The AdvancedCrossModalDenoisingLoss class implements a novel multi-objective denoising framework that operates during ensemble training through a sophisticated forward pass mechanism. The implementation begins with the __init__ method that initializes reconstruction loss (MSE) and consistency loss (KL divergence) modules, followed by the core forward method that processes each modality's data through the learner and applies configurable denoising objectives including reconstruction (predicting modality data from itself), consistency (maintaining representation consistency across modalities), and alignment (ensuring cross-modal feature alignment). The loss calculation iterates through each modality, computes modality-specific reconstruction losses by comparing learner predictions to original data, applies consistency losses using KL divergence between predicted and original representations, and aggregates all losses with configurable weighting (denoising_weight). The function returns both the total denoising loss and a detailed breakdown of per-modality losses, enabling fine-grained monitoring of cross-modal denoising effectiveness during training.
2. Comprehensive Modal-Specific Training Metrics
The ComprehensiveTrainingMetrics dataclass implements novel granular performance tracking through a comprehensive metrics collection system that captures both traditional and modal-specific performance indicators. The implementation tracks standard metrics (epoch, train_loss, val_loss, accuracy, f1_score, mse) alongside novel modal-specific metrics including modal_reconstruction_loss (dictionary storing reconstruction loss per modality), modal_alignment_score (cross-modal alignment quality per modality), and modal_consistency_score (overall consistency across modalities). The metrics collection occurs during the training loop where each epoch's performance is captured, modal-specific losses are extracted from the denoising loss function, alignment scores are computed between modality representations, and consistency scores are calculated using cross-modal similarity measures. The system also tracks training efficiency metrics (training_time, memory_usage, learning_rate) and stores all metrics in a structured format that enables detailed analysis of how each modality contributes to overall ensemble performance.
3. Bag-Aware Training with Modality Characteristics Preservation
The TrainedLearnerInfo dataclass implements novel bag-model integration by preserving complete modality characteristics alongside trained models through a comprehensive information storage system. The implementation begins with the train_ensemble method that processes each bag-learner configuration, extracts modality masks and weights from the BagLearnerConfig objects created in Stage 3, and maintains these characteristics throughout the training process. During training, the system preserves the modality_mask (which modalities were active during bag creation), modality_weights (importance scores calculated in Stage 3), and bag_id (unique identifier linking back to Stage 2 bag generation). After training completion, the TrainedLearnerInfo object is created containing the trained model, preserved modality characteristics, training metrics history, and final performance score, creating a complete audit trail. This novel approach enables full traceability from bag generation through learner selection to final trained model, allowing analysis of how specific modality configurations and weightages influenced each model's training and performance, which is uncommon in ensemble literature where this level of bag-model integration is typically not maintained.

//hyperparameters
🎯 CORE TRAINING HYPERPARAMETERS
epochs
Description: Number of training epochs for base learners
Range: [25, 50, 100]
Default: 50
Testing: VARY - Key parameter for training duration and convergence
learning_rate
Description: Learning rate for optimizer
Range: [1e-4, 5e-4, 1e-3]
Default: 5e-4
Testing: VARY - Critical for convergence and performance
batch_size
Description: Batch size for training base learners
Range: [32, 64, 128]
Default: 32
Testing: VARY - Affects training stability and memory usage
🎯 OVERFITTING PREVENTION HYPERPARAMETERS
early_stopping_patience
Description: Patience for early stopping
Range: [5, 10, 15]
Default: 10
Testing: VARY - Affects training termination
dropout_rate
Description: Dropout rate for regularization
Range: [0.1, 0.2, 0.3]
Default: 0.2
Testing: VARY - Affects overfitting prevention
�� CROSS-MODAL DENOISING HYPERPARAMETERS (NOVEL FEATURE)
enable_denoising
Description: Enable cross-modal denoising system
Range: [True, False]
Default: True
Testing: VARY - Key parameter for ablation studies
denoising_weight
Description: Weight for denoising loss in total loss
Range: [0.05, 0.1, 0.2]
Default: 0.1
Testing: VARY - Affects denoising contribution
denoising_strategy
Description: Strategy for adaptive denoising
Range: ['adaptive', 'fixed']
Default: 'adaptive'
Testing: VARY - Novel feature testing
�� MODAL-SPECIFIC METRICS TRACKING HYPERPARAMETERS (NOVEL FEATURE)
modal_specific_tracking
Description: Enable modal-specific metrics tracking
Range: [True, False]
Default: True
Testing: VARY - Key parameter for ablation studies
track_modal_reconstruction
Description: Track reconstruction loss per modality
Range: [True, False]
Default: True
Testing: VARY - Affects tracking granularity
🎯 BAG CHARACTERISTICS PRESERVATION HYPERPARAMETERS (NOVEL FEATURE)
preserve_bag_characteristics
Description: Enable bag characteristics preservation
Range: [True, False]
Default: True
Testing: VARY - Key parameter for ablation studies
�� FIXED PARAMETERS (DON'T VARY)
weight_decay
Description: L2 regularization weight decay
Range: 1e-3
Default: 1e-3
Testing: FIXED - Standard value
optimizer_type
Description: Type of optimizer for training
Range: 'adamw'
Default: 'adamw'
Testing: FIXED - Best performer
scheduler_type
Description: Learning rate scheduler type
Range: 'cosine_restarts'
Default: 'cosine_restarts'
Testing: FIXED - Best performer
gradient_accumulation_steps
Description: Number of steps for gradient accumulation
Range: 1
Default: 1
Testing: FIXED - Standard value
gradient_clipping
Description: Gradient clipping threshold
Range: 1.0
Default: 1.0
Testing: FIXED - Standard value
label_smoothing
Description: Label smoothing factor
Range: 0.1
Default: 0.1
Testing: FIXED - Standard value
denoising_objectives
Description: Denoising objectives to use
Range: ['reconstruction', 'alignment']
Default: ['reconstruction', 'alignment']
Testing: FIXED - Standard combination
denoising_modalities
Description: Specific modalities to apply denoising to
Range: []
Default: []
Testing: FIXED - All modalities
track_modal_alignment
Description: Track alignment score per modality
Range: True
Default: True
Testing: FIXED - Standard tracking
track_modal_consistency
Description: Track consistency score per modality
Range: True
Default: True
Testing: FIXED - Standard tracking
modal_tracking_frequency
Description: Frequency of modal-specific tracking
Range: 'every_epoch'
Default: 'every_epoch'
Testing: FIXED - Standard frequency
track_only_primary_modalities
Description: Track only primary modalities
Range: False
Default: False
Testing: FIXED - Track all modalities
save_modality_mask
Description: Save modality mask for each bag
Range: True
Default: True
Testing: FIXED - Standard preservation
save_modality_weights
Description: Save modality weights for each bag
Range: True
Default: True
Testing: FIXED - Standard preservation
save_bag_id
Description: Save bag ID for traceability
Range: True
Default: True
Testing: FIXED - Standard preservation
save_training_metrics
Description: Save training metrics history
Range: True
Default: True
Testing: FIXED - Standard preservation
save_learner_config
Description: Save learner configuration
Range: True
Default: True
Testing: FIXED - Standard preservation
preserve_only_primary_modalities
Description: Preserve only primary modalities
Range: False
Default: False
Testing: FIXED - Preserve all modalities
enable_data_augmentation
Description: Enable data augmentation
Range: False
Default: False
Testing: FIXED - Keep simple
augmentation_strength
Description: Strength of data augmentation
Range: 0.1
Default: 0.1
Testing: FIXED - Standard value
use_batch_norm
Description: Use batch normalization
Range: True
Default: True
Testing: FIXED - Standard value
enable_cross_validation
Description: Enable cross-validation
Range: False
Default: False
Testing: FIXED - Keep simple
cv_folds
Description: Number of cross-validation folds
Range: 5
Default: 5
Testing: FIXED - Standard value
random_state
Description: Random seed for reproducible training
Range: 42
Default: 42
Testing: EXPERIMENT-DEPENDENT - Set based on test run number
verbose
Description: Enables detailed training progress logging
Range: True
Default: True
Testing: FIXED - Keep True for monitoring

//abalation studies
1. CROSS-MODAL DENOISING SYSTEM ABLATION STUDY
Study Name: Cross_Modal_Denoising_System_Ablation
Feature Being Tested: Advanced Cross-Modal Denoising Loss - Your novel comprehensive denoising system that reduces noise and improves consistency across modalities using multiple objectives (reconstruction, alignment, consistency), rather than training without noise reduction
Control (Baseline): No Denoising: enable_denoising=False, denoising_weight=0.0 - Traditional training without any cross-modal denoising or noise reduction mechanisms
Variables for the Study:
ablation_variables = [
    # Control group (baseline)
    {'name': 'No_Denoising', 'enable_denoising': False, 'denoising_weight': 0.0},
    
    # Treatment group (your novel feature)
    {'name': 'Reconstruction_Denoising', 'enable_denoising': True, 'denoising_weight': 0.1, 'denoising_objectives': ['reconstruction']},
    {'name': 'Alignment_Denoising', 'enable_denoising': True, 'denoising_weight': 0.1, 'denoising_objectives': ['alignment']},
    {'name': 'Consistency_Denoising', 'enable_denoising': True, 'denoising_weight': 0.1, 'denoising_objectives': ['consistency']},
    {'name': 'All_Denoising_Objectives', 'enable_denoising': True, 'denoising_weight': 0.1, 'denoising_objectives': ['reconstruction', 'alignment', 'consistency']},
    
    # Additional denoising variants
    {'name': 'Adaptive_Denoising_Strategy', 'enable_denoising': True, 'denoising_weight': 0.1, 'denoising_strategy': 'adaptive', 'denoising_objectives': ['reconstruction', 'alignment', 'consistency']},
    {'name': 'Fixed_Denoising_Strategy', 'enable_denoising': True, 'denoising_weight': 0.1, 'denoising_strategy': 'fixed', 'denoising_objectives': ['reconstruction', 'alignment', 'consistency']},
    {'name': 'High_Denoising_Weight', 'enable_denoising': True, 'denoising_weight': 0.3, 'denoising_objectives': ['reconstruction', 'alignment', 'consistency']},
    {'name': 'Low_Denoising_Weight', 'enable_denoising': True, 'denoising_weight': 0.05, 'denoising_objectives': ['reconstruction', 'alignment', 'consistency']},
]

2. MODAL-SPECIFIC METRICS TRACKING ABLATION STUDY
Study Name: Modal_Specific_Metrics_Tracking_Ablation
Feature Being Tested: Comprehensive Modal-Specific Training Metrics - Your novel granular performance tracking system that captures per-modality reconstruction losses, alignment scores, and consistency scores during training, rather than only tracking overall performance metrics
Control (Baseline): Standard Metrics: modal_specific_tracking=False - Traditional training with only overall performance metrics (accuracy, loss, f1_score) without modal-specific granularity
Variables for the Study:
ablation_variables = [
    # Control group (baseline)
    {'name': 'Standard_Metrics_Only', 'modal_specific_tracking': False, 'track_modal_reconstruction': False, 'track_modal_alignment': False, 'track_modal_consistency': False},
    
    # Treatment group (your novel feature)
    {'name': 'Modal_Reconstruction_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': True, 'track_modal_alignment': False, 'track_modal_consistency': False},
    {'name': 'Modal_Alignment_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': False, 'track_modal_alignment': True, 'track_modal_consistency': False},
    {'name': 'Modal_Consistency_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': False, 'track_modal_alignment': False, 'track_modal_consistency': True},
    {'name': 'All_Modal_Specific_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': True, 'track_modal_alignment': True, 'track_modal_consistency': True},
    
    # Additional tracking variants
    {'name': 'High_Frequency_Modal_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': True, 'track_modal_alignment': True, 'track_modal_consistency': True, 'modal_tracking_frequency': 'every_epoch'},
    {'name': 'Low_Frequency_Modal_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': True, 'track_modal_alignment': True, 'track_modal_consistency': True, 'modal_tracking_frequency': 'every_5_epochs'},
    {'name': 'Selective_Modal_Tracking', 'modal_specific_tracking': True, 'track_modal_reconstruction': True, 'track_modal_alignment': True, 'track_modal_consistency': True, 'track_only_primary_modalities': True},
]

3. BAG-AWARE TRAINING WITH MODALITY CHARACTERISTICS PRESERVATION ABLATION STUDY
Study Name: Bag_Aware_Training_Modality_Preservation_Ablation
Feature Being Tested: Bag-Aware Training with Modality Characteristics Preservation - Your novel system that preserves complete bag modality characteristics (modality_mask, modality_weights, bag_id) alongside trained models, creating full traceability from bag generation through learner selection to final trained model
Control (Baseline): Standard Training: preserve_bag_characteristics=False - Traditional training that only saves the trained model without preserving bag-specific modality characteristics or traceability information
Variables for the Study:
ablation_variables = [
    # Control group (baseline)
    {'name': 'Standard_Training_Only', 'preserve_bag_characteristics': False, 'save_modality_mask': False, 'save_modality_weights': False, 'save_bag_id': False},
    
    # Treatment group (your novel feature)
    {'name': 'Modality_Mask_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': False, 'save_bag_id': False},
    {'name': 'Modality_Weights_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': False, 'save_modality_weights': True, 'save_bag_id': False},
    {'name': 'Bag_ID_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': False, 'save_modality_weights': False, 'save_bag_id': True},
    {'name': 'Full_Bag_Characteristics_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': True, 'save_bag_id': True},
    
    # Additional preservation variants
    {'name': 'Extended_Bag_Metadata_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': True, 'save_bag_id': True, 'save_training_metrics': True, 'save_learner_config': True},
    {'name': 'Minimal_Bag_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': True, 'save_bag_id': True, 'save_training_metrics': False, 'save_learner_config': False},
    {'name': 'Selective_Modality_Preservation', 'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': True, 'save_bag_id': True, 'preserve_only_primary_modalities': True},
]

//inteperetablity test
1. CROSS-MODAL DENOISING SYSTEM INTERPRETABILITY TEST
Test Name: Cross_Modal_Denoising_Effectiveness_Analysis
Feature Being Tested: Advanced Cross-Modal Denoising Loss - Your novel comprehensive denoising system that reduces noise and improves consistency across modalities using multiple objectives (reconstruction, alignment, consistency)
Interpretability Questions:
How effective is the cross-modal denoising system?
Which denoising objectives contribute most to performance?
How does denoising loss change during training?
What's the trade-off between denoising and main task performance?
Which modalities benefit most from denoising?
How does adaptive denoising strategy compare to fixed strategies?
Analysis Functions:
def analyze_cross_modal_denoising_effectiveness(model):
    """Analyze effectiveness of cross-modal denoising system"""
    trained_learners = model.get_trained_learners()
    
    denoising_analysis = {
        'denoising_loss_progression': {},
        'reconstruction_accuracy': {},
        'alignment_consistency': {},
        'consistency_scores': {},
        'denoising_effectiveness_score': {},
        'objective_contribution': {},
        'modality_benefit_analysis': {}
    }
    
    for learner_info in trained_learners:
        bag_id = learner_info.bag_id
        metrics = learner_info.training_metrics
        
        # Extract denoising metrics from training history
        denoising_losses = [m.modal_reconstruction_loss for m in metrics]
        alignment_scores = [m.modal_alignment_score for m in metrics]
        consistency_scores = [m.modal_consistency_score for m in metrics]
        
        denoising_analysis['denoising_loss_progression'][bag_id] = denoising_losses
        denoising_analysis['reconstruction_accuracy'][bag_id] = alignment_scores
        denoising_analysis['alignment_consistency'][bag_id] = consistency_scores
        denoising_analysis['consistency_scores'][bag_id] = consistency_scores
        
        # Calculate effectiveness scores
        final_denoising_loss = denoising_losses[-1] if denoising_losses else {}
        final_alignment = alignment_scores[-1] if alignment_scores else {}
        final_consistency = consistency_scores[-1] if consistency_scores else 0.0
        
        denoising_analysis['denoising_effectiveness_score'][bag_id] = {
            'reconstruction_effectiveness': 1.0 - np.mean(list(final_denoising_loss.values())) if final_denoising_loss else 0.0,
            'alignment_effectiveness': np.mean(list(final_alignment.values())) if final_alignment else 0.0,
            'consistency_effectiveness': final_consistency,
            'overall_effectiveness': 0.0  # Will be calculated
        }
    
    return denoising_analysis

2. MODAL-SPECIFIC METRICS TRACKING INTERPRETABILITY TEST
Test Name: Modal_Specific_Metrics_Granularity_Analysis
Feature Being Tested: Comprehensive Modal-Specific Training Metrics - Your novel granular performance tracking system that captures per-modality reconstruction losses, alignment scores, and consistency scores during training
Interpretability Questions:
How do different modalities perform during training?
Which modalities show the most improvement over time?
What's the correlation between modal-specific metrics and overall performance?
How does modal tracking frequency affect interpretability?
Which modalities are most critical for ensemble performance?
How do modal-specific metrics vary across different bag configurations?
Analysis Functions:
def analyze_modal_specific_metrics_granularity(model):
    """Analyze granularity and insights from modal-specific metrics tracking"""
    trained_learners = model.get_trained_learners()
    
    modal_analysis = {
        'modal_performance_progression': {},
        'modal_improvement_rates': {},
        'modal_correlation_analysis': {},
        'critical_modality_identification': {},
        'tracking_frequency_impact': {},
        'bag_configuration_modal_variation': {}
    }
    
    # Collect modal-specific metrics across all learners
    all_modal_metrics = {}
    for learner_info in trained_learners:
        bag_id = learner_info.bag_id
        modality_mask = learner_info.modality_mask
        metrics = learner_info.training_metrics
        
        # Extract modal-specific data
        for epoch_metrics in metrics:
            epoch = epoch_metrics.epoch
            if epoch not in all_modal_metrics:
                all_modal_metrics[epoch] = {}
            
            # Process reconstruction losses per modality
            for modality, loss in epoch_metrics.modal_reconstruction_loss.items():
                if modality not in all_modal_metrics[epoch]:
                    all_modal_metrics[epoch][modality] = {'reconstruction': [], 'alignment': [], 'consistency': []}
                all_modal_metrics[epoch][modality]['reconstruction'].append(loss)
            
            # Process alignment scores per modality
            for modality, score in epoch_metrics.modal_alignment_score.items():
                if modality not in all_modal_metrics[epoch]:
                    all_modal_metrics[epoch][modality] = {'reconstruction': [], 'alignment': [], 'consistency': []}
                all_modal_metrics[epoch][modality]['alignment'].append(score)
    
    # Analyze modal performance progression
    for modality in all_modal_metrics[0].keys():
        reconstruction_progression = []
        alignment_progression = []
        
        for epoch in sorted(all_modal_metrics.keys()):
            if modality in all_modal_metrics[epoch]:
                reconstruction_progression.append(np.mean(all_modal_metrics[epoch][modality]['reconstruction']))
                alignment_progression.append(np.mean(all_modal_metrics[epoch][modality]['alignment']))
        
        modal_analysis['modal_performance_progression'][modality] = {
            'reconstruction_trend': reconstruction_progression,
            'alignment_trend': alignment_progression,
            'improvement_rate': (reconstruction_progression[0] - reconstruction_progression[-1]) / len(reconstruction_progression) if reconstruction_progression else 0.0
        }
    
    return modal_analysis

3. BAG-AWARE TRAINING WITH MODALITY CHARACTERISTICS PRESERVATION INTERPRETABILITY TEST
Test Name: Bag_Characteristics_Preservation_Traceability_Analysis
Feature Being Tested: Bag-Aware Training with Modality Characteristics Preservation - Your novel system that preserves complete bag modality characteristics (modality_mask, modality_weights, bag_id) alongside trained models, creating full traceability from bag generation through learner selection to final trained model
Interpretability Questions:
How do preserved bag characteristics correlate with model performance?
What's the relationship between modality weights and training effectiveness?
How does bag configuration influence final model quality?
Which bag characteristics are most predictive of performance?
How does the complete audit trail help understand ensemble behavior?
What insights can be gained from the bag-to-model traceability?
Analysis Functions:
def analyze_bag_characteristics_preservation_traceability(model):
    """Analyze traceability and insights from bag characteristics preservation"""
    trained_learners = model.get_trained_learners()
    
    traceability_analysis = {
        'bag_characteristics_performance_correlation': {},
        'modality_weight_effectiveness': {},
        'bag_configuration_impact': {},
        'audit_trail_insights': {},
        'performance_prediction_accuracy': {},
        'ensemble_behavior_analysis': {}
    }
    
    # Analyze bag characteristics vs performance
    bag_performance_data = []
    for learner_info in trained_learners:
        bag_data = {
            'bag_id': learner_info.bag_id,
            'modality_mask': learner_info.modality_mask,
            'modality_weights': learner_info.modality_weights,
            'final_performance': learner_info.final_performance,
            'learner_type': learner_info.learner_type
        }
        bag_performance_data.append(bag_data)
    
    # Calculate correlations
    modality_count_correlation = np.corrcoef(
        [len([m for m, active in bag['modality_mask'].items() if active]) for bag in bag_performance_data],
        [bag['final_performance'] for bag in bag_performance_data]
    )[0, 1]
    
    modality_weight_correlation = {}
    for modality in trained_learners[0].modality_weights.keys():
        weights = [bag['modality_weights'].get(modality, 0.0) for bag in bag_performance_data]
        performances = [bag['final_performance'] for bag in bag_performance_data]
        modality_weight_correlation[modality] = np.corrcoef(weights, performances)[0, 1]
    
    traceability_analysis['bag_characteristics_performance_correlation'] = {
        'modality_count_correlation': modality_count_correlation,
        'modality_weight_correlations': modality_weight_correlation,
        'learner_type_performance': {}
    }
    
    # Analyze audit trail completeness
    audit_trail_completeness = {
        'bag_id_preserved': all(hasattr(li, 'bag_id') and li.bag_id > 0 for li in trained_learners),
        'modality_mask_preserved': all(hasattr(li, 'modality_mask') and len(li.modality_mask) > 0 for li in trained_learners),
        'modality_weights_preserved': all(hasattr(li, 'modality_weights') and len(li.modality_weights) > 0 for li in trained_learners),
        'training_metrics_preserved': all(hasattr(li, 'training_metrics') and len(li.training_metrics) > 0 for li in trained_learners)
    }
    
    traceability_analysis['audit_trail_insights'] = {
        'completeness_score': sum(audit_trail_completeness.values()) / len(audit_trail_completeness),
        'preservation_details': audit_trail_completeness,
        'traceability_benefits': {
            'bag_to_performance_traceable': True,
            'modality_contribution_identifiable': True,
            'training_progression_visible': True,
            'ensemble_diversity_analyzable': True
        }
    }
    
    return traceability_analysis

//robustness test
1. CROSS-MODAL DENOISING SYSTEM ROBUSTNESS TEST
Test Name: Cross_Modal_Denoising_Robustness
Feature Being Tested: Advanced Cross-Modal Denoising Loss - Your novel comprehensive denoising system that reduces noise and improves consistency across modalities using multiple objectives (reconstruction, alignment, consistency)
Purpose: Test how well the cross-modal denoising system handles different configurations, noise levels, and training scenarios
Robustness Test Scenarios:
denoising_robustness_tests = {
    'denoising_strategies': ['adaptive', 'fixed', 'progressive'],
    'denoising_weights': [0.05, 0.1, 0.2, 0.3, 0.5],
    'denoising_objectives': [['reconstruction'], ['alignment'], ['consistency'], ['reconstruction', 'alignment'], ['reconstruction', 'alignment', 'consistency']],
    'noise_levels': [0.0, 0.05, 0.1, 0.2, 0.3],
    'modality_combinations': ['text_only', 'image_only', 'text+image', 'text+metadata', 'image+metadata', 'all_modalities'],
    'training_epochs': [10, 25, 50, 100],
    'batch_sizes': [16, 32, 64, 128]
}
Tests:
Strategy Robustness: Test with adaptive, fixed, progressive denoising strategies
Weight Sensitivity: Test with different denoising weight values (0.05 to 0.5)
Objective Robustness: Test with different denoising objective combinations
Noise Robustness: Test with different noise levels in training data
Modality Robustness: Test with different modality combinations
Training Robustness: Test with different training configurations (epochs, batch sizes)

2. MODAL-SPECIFIC METRICS TRACKING ROBUSTNESS TEST
Test Name: Modal_Specific_Metrics_Tracking_Robustness
Feature Being Tested: Comprehensive Modal-Specific Training Metrics - Your novel granular performance tracking system that captures per-modality reconstruction losses, alignment scores, and consistency scores during training
Purpose: Test how well the modal-specific metrics tracking system handles different tracking configurations, data variations, and computational constraints
Robustness Test Scenarios:
modal_tracking_robustness_tests = {
    'tracking_frequencies': ['every_epoch', 'every_2_epochs', 'every_5_epochs', 'every_10_epochs'],
    'tracking_combinations': [
        {'track_modal_reconstruction': True, 'track_modal_alignment': False, 'track_modal_consistency': False},
        {'track_modal_reconstruction': False, 'track_modal_alignment': True, 'track_modal_consistency': False},
        {'track_modal_reconstruction': False, 'track_modal_alignment': False, 'track_modal_consistency': True},
        {'track_modal_reconstruction': True, 'track_modal_alignment': True, 'track_modal_consistency': False},
        {'track_modal_reconstruction': True, 'track_modal_alignment': True, 'track_modal_consistency': True}
    ],
    'modality_selections': ['all_modalities', 'primary_modalities_only', 'secondary_modalities_only', 'mixed_selection'],
    'data_sizes': ['small', 'medium', 'large', 'very_large'],
    'computational_constraints': ['low_memory', 'high_memory', 'cpu_only', 'gpu_available'],
    'training_durations': ['short', 'medium', 'long', 'very_long']
}
Tests:
Frequency Robustness: Test with different tracking frequencies
Combination Robustness: Test with different tracking metric combinations
Modality Selection Robustness: Test with different modality selection strategies
Data Size Robustness: Test with different dataset sizes
Computational Robustness: Test under different computational constraints
Duration Robustness: Test with different training durations

3. BAG CHARACTERISTICS PRESERVATION ROBUSTNESS TEST
Test Name: Bag_Characteristics_Preservation_Robustness
Feature Being Tested: Bag-Aware Training with Modality Characteristics Preservation - Your novel system that preserves complete bag modality characteristics (modality_mask, modality_weights, bag_id) alongside trained models, creating full traceability from bag generation through learner selection to final trained model
Purpose: Test how well the bag characteristics preservation system handles different preservation configurations, memory constraints, and data variations
Robustness Test Scenarios:
bag_preservation_robustness_tests = {
    'preservation_combinations': [
        {'preserve_bag_characteristics': False, 'save_modality_mask': False, 'save_modality_weights': False, 'save_bag_id': False},
        {'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': False, 'save_bag_id': False},
        {'preserve_bag_characteristics': True, 'save_modality_mask': False, 'save_modality_weights': True, 'save_bag_id': False},
        {'preserve_bag_characteristics': True, 'save_modality_mask': False, 'save_modality_weights': False, 'save_bag_id': True},
        {'preserve_bag_characteristics': True, 'save_modality_mask': True, 'save_modality_weights': True, 'save_bag_id': True}
    ],
    'memory_constraints': ['low_memory', 'medium_memory', 'high_memory', 'unlimited_memory'],
    'bag_sizes': ['small_ensemble', 'medium_ensemble', 'large_ensemble', 'very_large_ensemble'],
    'modality_complexities': ['simple', 'moderate', 'complex', 'very_complex'],
    'training_metrics_levels': ['minimal', 'standard', 'detailed', 'comprehensive'],
    'learner_config_preservation': ['none', 'basic', 'detailed', 'complete']
}
Tests:
Preservation Combination Robustness: Test with different preservation configuration combinations
Memory Constraint Robustness: Test under different memory constraints
Bag Size Robustness: Test with different ensemble sizes
Modality Complexity Robustness: Test with different modality complexity levels
Metrics Level Robustness: Test with different training metrics preservation levels
Config Preservation Robustness: Test with different learner configuration preservation levels

4. INTEGRATED STAGE 4 ROBUSTNESS TEST
Test Name: Integrated_Stage4_Novel_Features_Robustness
Feature Being Tested: Combined Stage 4 Novel Features - Integration of cross-modal denoising, modal-specific metrics tracking, and bag characteristics preservation working together
Purpose: Test how well all novel Stage 4 features work together under various combined stress conditions
Robustness Test Scenarios:
integrated_robustness_tests = {
    'feature_combinations': [
        {'enable_denoising': False, 'modal_specific_tracking': False, 'preserve_bag_characteristics': False},
        {'enable_denoising': True, 'modal_specific_tracking': False, 'preserve_bag_characteristics': False},
        {'enable_denoising': False, 'modal_specific_tracking': True, 'preserve_bag_characteristics': False},
        {'enable_denoising': False, 'modal_specific_tracking': False, 'preserve_bag_characteristics': True},
        {'enable_denoising': True, 'modal_specific_tracking': True, 'preserve_bag_characteristics': False},
        {'enable_denoising': True, 'modal_specific_tracking': False, 'preserve_bag_characteristics': True},
        {'enable_denoising': False, 'modal_specific_tracking': True, 'preserve_bag_characteristics': True},
        {'enable_denoising': True, 'modal_specific_tracking': True, 'preserve_bag_characteristics': True}
    ],
    'stress_conditions': ['normal', 'high_noise', 'low_memory', 'fast_training', 'slow_training', 'mixed_stress'],
    'dataset_variations': ['synthetic', 'real_world', 'imbalanced', 'high_dimensional', 'sparse', 'dense'],
    'hardware_configurations': ['cpu_only', 'gpu_available', 'multi_gpu', 'distributed', 'edge_device']
}
Tests:
Feature Combination Robustness: Test with different combinations of novel features enabled/disabled
Stress Condition Robustness: Test under various stress conditions
Dataset Variation Robustness: Test with different dataset characteristics
Hardware Robustness: Test under different hardware configurations
Integration Robustness: Test how well features integrate with each other
Performance Robustness: Test performance under combined feature usage

stage 5:

//summary
Stage 5: Simplified Multimodal Ensemble Prediction System orchestrates the final prediction phase through a streamlined pipeline that transforms trained weak learners into intelligent ensemble predictions. Step 1 (Individual Predictions) employs _get_individual_predictions() to obtain predictions from each trained learner, handling both PyTorch neural networks and sklearn models through forward passes and probability calculations with robust fallback mechanisms. Step 2 (Confidence & Uncertainty Computation) applies _compute_confidence() and _compute_uncertainty() using entropy, variance, or confidence-based methods to estimate prediction reliability and confidence scores. Step 3 (Modality-Aware Transformer Aggregation) - the novel core feature - utilizes _transformer_fusion() to implement adaptive ensemble aggregation through a modality-aware transformer meta-learner that dynamically estimates relative importance of each learner's predictions, evaluates accuracy AND generalization ability across diverse subsets, and applies context-dependent attention weights that mirror adaptive dropout philosophy. The transformer meta-learner combines standard transformer attention with a novel signal integration scheme: generalization scores, confidence bonuses, uncertainty penalties, and modality importance weights to produce intelligent ensemble predictions specifically designed for multimodal learning scenarios. Step 4 (Result Assembly) orchestrates all previous steps through the main predict() function, combining predictions, confidence, uncertainty, modality importance, and metadata into a final PredictionResult object. The flow progresses linearly: test data → individual predictions → confidence/uncertainty computation → simplified transformer aggregation → final result assembly, delivering actual predictions with uncertainty quantification, confidence scores, and modality importance analysis through the novel simplified transformer meta-learner for adaptive ensemble aggregation.

//novel features
NOVEL FEATURE 1: MODALITY-AWARE TRANSFORMER ENSEMBLE AGGREGATION
The novel modality-aware transformer ensemble aggregation implements adaptive ensemble weighting specifically designed for multimodal learning scenarios through a streamlined but effective approach. The TransformerMetaLearner class operates through five key steps: (1) Input Projection where learner outputs are mapped to a hidden dimension through a linear layer, (2) Generalization Scoring where a neural network computes generalization scores for each learner based on their ability to perform across diverse subsets, (3) Transformer Encoding where a standard transformer encoder processes the projected outputs to create contextualized learner representations that capture relationships between learners, (4) Attention Computation where an attention head computes base attention weights from the contextualized embeddings, and (5) Adaptive Signal Integration where the final mixture weights combine base transformer attention with a novel weighting scheme: generalization scores (weighted by 2.0), confidence bonuses (weighted by 1.5), uncertainty penalties (weighted by 1.0), and modality importance weights (weighted by 0.5) to produce intelligent ensemble predictions. The novelty lies in the specific application of transformer attention to multimodal ensemble aggregation with modality-aware signal integration, ensuring that learners with higher generalization ability, confidence, and modality importance receive greater weight in the final ensemble decision, while overfit learners and those with high uncertainty are appropriately down-weighted.

NOVEL FEATURE 2: ROBUST PREDICTION HANDLING WITH FALLBACK MECHANISMS
The novel robust prediction handling system ensures reliable ensemble predictions even when individual learners fail through comprehensive fallback mechanisms in the _get_individual_predictions() function. The system attempts multiple prediction methods for each learner: first trying predict_proba() for classification tasks, then falling back to predict() for general predictions, then attempting forward() for PyTorch models, and finally creating dummy predictions as a last resort. This robust approach handles diverse learner types (sklearn models, PyTorch neural networks, custom learners) and gracefully manages prediction failures by providing sensible defaults, ensuring that the ensemble can continue operating even when some learners encounter issues. The system also includes comprehensive error handling with warnings for failed predictions while maintaining ensemble functionality, making the overall system more reliable and production-ready.

//hyperparameters
1. aggregation_strategy
Description: Ensemble aggregation strategy for combining predictions from multiple learners - includes modality-aware transformer-based fusion
Range: ['simple_average', 'weighted_average', 'transformer_fusion']
Default: 'transformer_fusion' (moderately novel feature)
Testing: VARY - Test all strategies, focus on transformer_fusion
2. uncertainty_method
Description: Method for quantifying prediction uncertainty - includes entropy, variance, and confidence-based estimation
Range: ['entropy', 'variance', 'confidence']
Default: 'entropy'
Testing: VARY - Key parameter for uncertainty quantification
🧠 SIMPLIFIED TRANSFORMER META-LEARNER HYPERPARAMETERS
3. transformer_num_heads
Description: Number of attention heads in the simplified transformer meta-learner - affects the model's ability to attend to different aspects of predictions
Range: [2, 4, 6, 8]
Default: 4
Testing: FIXED - Keep 4 for optimal performance
4. transformer_num_layers
Description: Number of transformer encoder layers in the meta-learner - affects model complexity and learning capacity
Range: [1, 2, 3, 4]
Default: 2
Testing: FIXED - Keep 2 for efficiency
5. transformer_hidden_dim
Description: Hidden dimension size in the transformer meta-learner - affects model capacity and computational cost
Range: [32, 64, 128, 256]
Default: 64
Testing: FIXED - Keep 64 for balanced performance
🔧 PREDICTION CONFIGURATION HYPERPARAMETERS
6. task_type
Description: Type of prediction task - affects aggregation strategy and output format
Range: ['classification', 'regression']
Default: 'classification'
Testing: FIXED - Keep 'classification' for standard tasks

//ablation studies
1. MODALITY-AWARE TRANSFORMER AGGREGATION ABLATION STUDY
Study Name: Aggregation_Strategy_Ablation
Feature Being Tested: Modality-Aware Transformer Ensemble Aggregation - The moderately novel transformer meta-learner that dynamically estimates relative importance of each learner's predictions through modality-aware adaptive weighting
Control (Baseline): Simple Averaging - Basic ensemble prediction using simple averaging of all learner predictions without any adaptive weighting
Variables for the Study:
ablation_variables = [
    # Control group (baseline)
    {'name': 'Simple_Average', 'aggregation_strategy': 'simple_average'},
    {'name': 'Weighted_Average', 'aggregation_strategy': 'weighted_average'},
    
    # Treatment group (novel feature)
    {'name': 'Transformer_Fusion', 'aggregation_strategy': 'transformer_fusion'},
]

//interpretability tests
1. MODALITY-AWARE ENSEMBLE INTERPRETABILITY TEST
Test Name: Modality_Aware_Ensemble_Interpretability
Feature Being Tested: Modality-Aware Transformer Ensemble Aggregation - The moderately novel transformer meta-learner that provides interpretable insights into ensemble decision-making through attention weights, modality importance, and uncertainty quantification
Interpretability Questions:
How does the modality-aware transformer meta-learner weight different learners based on their performance characteristics?
Which modalities contribute most to ensemble decisions and how is this reflected in attention weights?
How does uncertainty quantification correlate with prediction accuracy and confidence?
What is the relationship between generalization scores and final ensemble predictions?
How effective is the novel signal integration scheme (generalization + confidence + uncertainty + modality importance)?
Analysis Functions:
def analyze_modality_aware_ensemble_interpretability(model):
    """Analyze modality-aware ensemble interpretability"""
    # Test different aggregation strategies
    simple_result = model.predict(test_data, aggregation_strategy='simple_average')
    weighted_result = model.predict(test_data, aggregation_strategy='weighted_average')
    transformer_result = model.predict(test_data, aggregation_strategy='transformer_fusion')
    
    return {
        'aggregation_strategy_comparison': compare_aggregation_strategies(simple_result, weighted_result, transformer_result),
        'modality_importance_analysis': analyze_modality_importance(transformer_result),
        'attention_weight_analysis': analyze_attention_weights(transformer_result),
        'uncertainty_correlation_analysis': analyze_uncertainty_correlation(transformer_result),
        'generalization_score_analysis': analyze_generalization_scores(transformer_result)
    }

//robustness tests
1. MODALITY-AWARE ENSEMBLE ROBUSTNESS TEST
Test Name: Modality_Aware_Ensemble_Robustness
Feature Being Tested: Modality-Aware Transformer Ensemble Aggregation - The moderately novel transformer meta-learner's robustness to noise, perturbations, and varying input conditions
Robustness Questions:
How does the modality-aware transformer meta-learner perform under different noise levels?
How robust is the ensemble to missing or corrupted learner predictions?
How does the system handle varying numbers of learners and modalities?
What is the performance degradation under different perturbation scenarios?
How robust is the novel signal integration scheme to signal corruption?
Analysis Functions:
def test_modality_aware_ensemble_robustness(model):
    """Test modality-aware ensemble robustness"""
    # Test with noise perturbation
    noise_levels = [0.01, 0.05, 0.1]
    robustness_scores = []
    
    for noise_level in noise_levels:
        # Add noise to test data
        noisy_data = add_noise_to_data(test_data, noise_level)
        
        # Make predictions on noisy data
        result = model.predict(noisy_data)
        
        # Compute robustness score
        robustness_score = calculate_robustness_score(result, test_labels)
        robustness_scores.append(robustness_score)
    
    return {
        'noise_robustness_scores': robustness_scores,
        'overall_robustness_score': np.mean(robustness_scores),
        'robustness_degradation': analyze_robustness_degradation(robustness_scores),
        'learner_failure_robustness': test_learner_failure_robustness(model, test_data),
        'modality_perturbation_robustness': test_modality_perturbation_robustness(model, test_data)
    }
