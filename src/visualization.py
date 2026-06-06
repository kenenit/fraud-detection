import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_class_distribution(df, target_col, title, save_path=None):
    """Bar chart of class distribution."""
    counts = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    print(pd.DataFrame({'Count': counts, 'Percentage': pct.round(2)}))

    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df,
                  palette=['steelblue', 'crimson'])
    plt.title(title)
    plt.xticks([0, 1], ['Legitimate', 'Fraud'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_fraud_by_country(df, min_transactions=100, top_n=15, save_path=None):
    """Bar chart of fraud rate by country."""
    country_fraud = df.groupby('country').agg(
        total=('class', 'count'),
        fraud=('class', 'sum')
    ).reset_index()
    country_fraud['fraud_rate'] = (
        country_fraud['fraud'] / country_fraud['total']
    )
    country_fraud = country_fraud[
        country_fraud['total'] >= min_transactions
    ]
    top = country_fraud.sort_values(
        'fraud_rate', ascending=False
    ).head(top_n)

    plt.figure(figsize=(12, 5))
    sns.barplot(x='country', y='fraud_rate',
                data=top, palette='Reds_r')
    plt.title(f'Top {top_n} Countries by Fraud Rate')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Fraud Rate')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_vs_fraud(df, feature, target_col='class', save_path=None):
    """Boxplot of a feature split by fraud class."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target_col, y=feature, data=df,
                palette=['steelblue', 'crimson'])
    plt.title(f'{feature} by Class')
    plt.xticks([0, 1], ['Legitimate', 'Fraud'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(results_list, title, save_path=None):
    """Bar chart comparing model AUC-PR and AUC-ROC."""
    results_df = pd.DataFrame(results_list).set_index('model')
    results_df = results_df.sort_values('AUC-PR', ascending=False)
    results_df.plot(kind='bar', figsize=(8, 4),
                    colormap='Set2', ylim=(0, 1))
    plt.title(title)
    plt.ylabel('Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return results_df