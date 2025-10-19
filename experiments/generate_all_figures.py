#!/usr/bin/env python3
"""
Generate all publication-quality figures for dissertation experiments 6.1-6.6
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'neutral': '#6C757D'
}

def load_json(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_exp_6_1_figure(results_dir):
    """
    Experiment 6.1: Falsification Rate Comparison Bar Chart
    """
    print("Generating Figure 6.1: Falsification Rate Comparison")

    # Load latest results
    results_file = results_dir / "exp_6_1_results_20251018_180300.json"
    data = load_json(results_file)

    # Extract data
    methods = ['Grad-CAM', 'SHAP', 'LIME']
    frs = [data['results'][m]['falsification_rate'] for m in methods]
    ci_lower = [data['results'][m]['confidence_interval']['lower'] for m in methods]
    ci_upper = [data['results'][m]['confidence_interval']['upper'] for m in methods]

    # Calculate error bars
    errors_lower = [frs[i] - ci_lower[i] for i in range(len(methods))]
    errors_upper = [ci_upper[i] - frs[i] for i in range(len(methods))]
    errors = [errors_lower, errors_upper]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    x_pos = np.arange(len(methods))
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    bars = ax.bar(x_pos, frs, yerr=errors,
                  color=colors, alpha=0.8, capsize=5,
                  edgecolor='black', linewidth=1.2, ecolor='black')

    # Customize
    ax.set_xlabel('Attribution Method', fontweight='bold')
    ax.set_ylabel('Falsification Rate (%)', fontweight='bold')
    ax.set_title('Falsification Rates Across Attribution Methods (n=200)',
                 fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylim([0, 70])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, (bar, fr) in enumerate(zip(bars, frs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{fr:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()

    # Save
    output_file = results_dir / "figure_6_1_falsification_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return output_file

def generate_exp_6_2_figure(results_dir):
    """
    Experiment 6.2: Separation Margin vs Falsification Rate Scatter Plot
    """
    print("Generating Figure 6.2: Separation Margin Analysis")

    # Load latest results
    results_file = results_dir / "exp_6_2_results_20251018_183607.json"
    data = load_json(results_file)

    # Extract data
    strata = data['strata_results']

    # Get midpoints and FRs
    margin_midpoints = []
    frs = []
    ci_lower = []
    ci_upper = []
    n_samples = []

    for stratum_name, stratum_data in strata.items():
        margin_range = stratum_data['margin_range']
        midpoint = (margin_range[0] + margin_range[1]) / 2
        margin_midpoints.append(midpoint)
        frs.append(stratum_data['falsification_rate'])
        ci_lower.append(stratum_data['confidence_interval']['lower'])
        ci_upper.append(stratum_data['confidence_interval']['upper'])
        n_samples.append(stratum_data['n_pairs'])

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter plot with size proportional to sample size
    sizes = [n * 3 for n in n_samples]  # Scale for visibility
    scatter = ax.scatter(margin_midpoints, frs, s=sizes, c=COLORS['primary'],
                        alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add error bars
    ax.errorbar(margin_midpoints, frs,
                yerr=[np.array(frs) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(frs)],
                fmt='none', ecolor='black', capsize=5, alpha=0.7)

    # Add regression line
    reg_data = data['statistical_tests']['linear_regression']
    slope = reg_data['slope']
    intercept = reg_data['intercept']
    r_squared = reg_data['r_squared']

    x_line = np.linspace(0, max(margin_midpoints), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color=COLORS['quaternary'],
            linewidth=2, label=f'Linear fit: $R^2={r_squared:.3f}$')

    # Customize
    ax.set_xlabel('Separation Margin (δ)', fontweight='bold')
    ax.set_ylabel('Falsification Rate (%)', fontweight='bold')
    ax.set_title('Impact of Separation Margin on Falsification Rate',
                 fontweight='bold', pad=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left')

    # Add annotation for sample sizes
    ax.text(0.98, 0.02, 'Bubble size ∝ sample size',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, style='italic', color='gray')

    plt.tight_layout()

    # Save
    output_file = results_dir / "figure_6_2_margin_analysis.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return output_file

def generate_exp_6_3_figure(results_dir):
    """
    Experiment 6.3: Attribute Falsifiability Heatmap
    """
    print("Generating Figure 6.3: Attribute Falsifiability Heatmap")

    # Load latest results
    results_file = results_dir / "exp_6_3_results_20251018_180752.json"
    data = load_json(results_file)

    # Extract top 10 attributes
    top_attrs = data['top_10_attributes']

    # Organize by category
    categories = ['Expression', 'Demographic', 'Occlusion', 'Geometric']
    category_data = {cat: [] for cat in categories}

    for attr_name, attr_data in top_attrs.items():
        cat = attr_data['category']
        fr = attr_data['falsification_rate']
        category_data[cat].append((attr_name, fr))

    # Create matrix for heatmap
    max_attrs_per_cat = max(len(attrs) for attrs in category_data.values())

    # Prepare data
    heatmap_data = []
    row_labels = []

    for cat in categories:
        attrs = category_data[cat]
        if attrs:
            for attr_name, fr in attrs:
                heatmap_data.append([fr if c == cat else np.nan for c in categories])
                row_labels.append(attr_name)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    heatmap_matrix = []
    y_labels = []

    for cat in categories:
        attrs = sorted(category_data[cat], key=lambda x: x[1], reverse=True)
        for attr_name, fr in attrs:
            row = [np.nan] * len(categories)
            row[categories.index(cat)] = fr
            heatmap_matrix.append(row)
            y_labels.append(attr_name)

    # Plot heatmap
    im = ax.imshow(heatmap_matrix, cmap='YlOrRd', aspect='auto', vmin=40, vmax=70)

    # Set ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(categories, rotation=0, ha='center')
    ax.set_yticklabels(y_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Falsification Rate (%)', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(categories)):
            value = heatmap_matrix[i][j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.1f}',
                             ha='center', va='center', color='black', fontweight='bold',
                             fontsize=8)

    # Customize
    ax.set_xlabel('Attribute Category', fontweight='bold')
    ax.set_ylabel('Attribute Name', fontweight='bold')
    ax.set_title('Falsifiability of Top 10 Facial Attributes by Category',
                 fontweight='bold', pad=10)

    # Add grid
    ax.set_xticks(np.arange(len(categories))+0.5, minor=True)
    ax.set_yticks(np.arange(len(y_labels))+0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()

    # Save
    output_file = results_dir / "figure_6_3_attribute_heatmap.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return output_file

def generate_exp_6_4_figure(results_dir):
    """
    Experiment 6.4: Model-Agnostic Comparison Across Architectures
    """
    print("Generating Figure 6.4: Model-Agnostic Comparison")

    # Load latest results
    results_file = results_dir / "exp_6_4_results_20251018_180635.json"
    data = load_json(results_file)

    # Extract data
    models = data['models_tested']
    methods = data['attribution_methods']

    results = data['results_by_method']

    # Prepare data for grouped bar chart
    x = np.arange(len(models))
    width = 0.35

    method1_frs = [results[methods[0]][model]['falsification_rate'] for model in models]
    method2_frs = [results[methods[1]][model]['falsification_rate'] for model in models]

    method1_ci_lower = [results[methods[0]][model]['confidence_interval']['lower'] for model in models]
    method1_ci_upper = [results[methods[0]][model]['confidence_interval']['upper'] for model in models]
    method2_ci_lower = [results[methods[1]][model]['confidence_interval']['lower'] for model in models]
    method2_ci_upper = [results[methods[1]][model]['confidence_interval']['upper'] for model in models]

    method1_errors = [[method1_frs[i] - method1_ci_lower[i] for i in range(len(models))],
                      [method1_ci_upper[i] - method1_frs[i] for i in range(len(models))]]
    method2_errors = [[method2_frs[i] - method2_ci_lower[i] for i in range(len(models))],
                      [method2_ci_upper[i] - method2_frs[i] for i in range(len(models))]]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, method1_frs, width, yerr=method1_errors,
                   label=methods[0], color=COLORS['primary'], alpha=0.8,
                   capsize=5, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, method2_frs, width, yerr=method2_errors,
                   label=methods[1], color=COLORS['secondary'], alpha=0.8,
                   capsize=5, edgecolor='black', linewidth=1.2)

    # Customize
    ax.set_xlabel('Face Recognition Model', fontweight='bold')
    ax.set_ylabel('Falsification Rate (%)', fontweight='bold')
    ax.set_title('Model-Agnostic Testing: Attribution Methods Across FR Architectures (n=500)',
                 fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(title='Attribution Method', title_fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim([0, 80])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save
    output_file = results_dir / "figure_6_4_model_agnostic.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")
    return output_file

def main():
    """Generate all missing figures"""

    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60)
    print()

    base_dir = Path("/home/aaron/projects/xai/experiments/results_real")

    generated_files = []

    # Experiment 6.1
    exp_6_1_dir = base_dir / "exp_6_1"
    if exp_6_1_dir.exists():
        fig_file = generate_exp_6_1_figure(exp_6_1_dir)
        generated_files.append(fig_file)

    # Experiment 6.2
    exp_6_2_dir = base_dir / "exp_6_2"
    if exp_6_2_dir.exists():
        fig_file = generate_exp_6_2_figure(exp_6_2_dir)
        generated_files.append(fig_file)

    # Experiment 6.3
    exp_6_3_dir = base_dir / "exp_6_3"
    if exp_6_3_dir.exists():
        fig_file = generate_exp_6_3_figure(exp_6_3_dir)
        generated_files.append(fig_file)

    # Experiment 6.4
    exp_6_4_dir = base_dir / "exp_6_4"
    if exp_6_4_dir.exists():
        fig_file = generate_exp_6_4_figure(exp_6_4_dir)
        generated_files.append(fig_file)

    print()
    print("=" * 60)
    print(f"COMPLETE: Generated {len(generated_files)} figures")
    print("=" * 60)
    print()
    print("Generated files:")
    for f in generated_files:
        print(f"  - {f}")

    return generated_files

if __name__ == "__main__":
    main()
