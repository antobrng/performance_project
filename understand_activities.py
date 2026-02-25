import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def understand_activities():
    path = "/Users/antoninberanger/Documents/performance_project/data/activities_clean.csv"
    df = pd.read_csv(path)

    # 1. Conversion Helpers
    def duration_to_minutes(time_str):
        try:
            parts = str(time_str).split(':')
            if len(parts) == 3: return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
            if len(parts) == 2: return int(parts[0]) + float(parts[1]) / 60
            return 0
        except: return 0

    def pace_to_speed(pace_str):
        try:
            parts = str(pace_str).split(':')
            minutes = int(parts[0]) + int(parts[1])/60
            return 60 / minutes
        except: return 0

    # 2. Data Preparation
    df['Durée_min'] = df['Durée'].apply(duration_to_minutes)
    df['Vitesse_kmh'] = df['Allure moyenne'].apply(pace_to_speed)
    df['FC_moyenne'] = pd.to_numeric(df['Fréquence cardiaque moyenne'], errors='coerce')
    df['TE_aerobie'] = pd.to_numeric(df['TE aérobie'], errors='coerce')
    
    # Ensure Distance is numeric (handling the comma/dot issue if needed)
    df['Distance'] = pd.to_numeric(df['Distance'].astype(str).str.replace(',', '.'), errors='coerce')

    # Filter
    reg_cols = ['TE_aerobie', 'Durée_min', 'Vitesse_kmh', 'FC_moyenne', 'Distance']
    reg_df = df.dropna(subset=reg_cols)
    reg_df = reg_df[(reg_df[reg_cols] > 0).all(axis=1)].copy()

    if not reg_df.empty:
        # 3. LINEAR REGRESSION (Adding Distance)
        X = reg_df[['Durée_min', 'Vitesse_kmh', 'FC_moyenne', 'Distance']]
        y = reg_df['TE_aerobie']
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        print("--- Linear Regression: Drivers of Aerobic TE ---")
        print(f"R-squared: {model.rsquared:.4f}")
        print(model.summary())
        
        return reg_df, model
    else:
        print("Error: No valid data points found.")
        return None, None

def visualize_performance_metrics(data, model):
    # Prepare X with the same columns used in training
    X = sm.add_constant(data[['Durée_min', 'Vitesse_kmh', 'FC_moyenne', 'Distance']])
    
    # Predict and calculate residuals
    data['pred'] = model.predict(X)
    data['residuals'] = data['TE_aerobie'] - data['pred']

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Actual vs Predicted
    sns.scatterplot(ax=axes[0], x=data['pred'], y=data['TE_aerobie'], hue=data['FC_moyenne'], palette='flare')
    axes[0].plot([1, 5], [1, 5], 'r--')
    axes[0].set_title(f'Model Accuracy (R2={model.rsquared:.3f})')
    axes[0].set_ylabel('Actual TE')
    axes[0].set_xlabel('Predicted TE')

    # Plot 2: Distance Impact
    sns.regplot(ax=axes[1], x=data['Distance'], y=data['TE_aerobie'], scatter_kws={'alpha':0.5}, line_kws={'color':'green'})
    axes[1].set_title('Distance vs TE')
    axes[1].set_xlabel('Distance (km)')

    # Plot 3: Coefficients (The Drivers)
    coefs = model.params.drop('const')
    coefs.plot(kind='barh', ax=axes[2], color='skyblue')
    axes[2].set_title('Impact of Each Variable')
    axes[2].set_xlabel('Coefficient Value')
    axes[2].axvline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    save_path = "/Users/antoninberanger/Documents/performance_project/te_regression_plots.png"
    plt.savefig(save_path)
    print(f"Regression plots saved at: {save_path}")
    plt.show()

def solve_for_duration(model, target_te=4.0):
    b0 = model.params['const']
    b_dur = model.params['Durée_min']
    b_vit = model.params['Vitesse_kmh']
    b_fc = model.params['FC_moyenne']
    b_dist = model.params['Distance']

    print(f"\nThe best way to achieve a TE of {target_te} is to adjust duration/distance, speed, and heart rate.\nHere's how long I would need to run at different zones:")

    print(f"\n{'ZONE':<15} | {'HR':<5} | {'SPEED':<5} | {'OPTIM. DURATION (min)'}")
    print(f"{'-'*55}")

    zones = [("Base", 145, 10.5), ("Tempo", 158, 11.6), ("Threshold", 182, 15.5)]

    for name, hr, speed in zones:
        # Since Distance = Speed * (Duration/60), we substitute it:
        # TE = b0 + b_dur*T + b_vit*S + b_fc*HR + b_dist*(S * T/60)
        # TE - b0 - b_vit*S - b_fc*HR = T * (b_dur + b_dist*S/60)
        
        numerator = target_te - b0 - (b_vit * speed) - (b_fc * hr)
        denominator = b_dur + (b_dist * speed / 60)
        
        if denominator != 0:
            dur = numerator / denominator
            if dur > 0:
                print(f"{name:<15} | {hr:<5} | {speed:<5} | {int(dur)}m {int((dur%1)*60)}s")
            else:
                print(f"{name:<15} | {hr:<5} | {speed:<5} | Target already met")

def solve_for_distance(model, target_te=4.0):
    b0 = model.params['const']
    b_dur = model.params['Durée_min']
    b_vit = model.params['Vitesse_kmh']
    b_fc = model.params['FC_moyenne']
    b_dist = model.params['Distance']

    print(f"\n{'ZONE':<15} | {'HR':<5} | {'SPEED':<5} | {'OPTIM. DISTANCE (km)'}")
    print(f"{'-'*55}")

    zones = [("Base", 145, 10.5), ("Tempo", 158, 11.6), ("Threshold", 182, 15.5)]

    for name, hr, speed in zones:
        # Since Duration = (Distance / Speed) * 60
        # We substitute T with (D/S)*60 in the regression equation
        # TE = b0 + b_dur*((D/S)*60) + b_vit*S + b_fc*HR + b_dist*D
        # TE - b0 - b_vit*S - b_fc*HR = D * (b_dist + b_dur*60/S)
        
        numerator = target_te - b0 - (b_vit * speed) - (b_fc * hr)
        denominator = b_dist + (b_dur * 60 / speed)
        
        if denominator != 0:
            dist = numerator / denominator
            if dist > 0:
                print(f"{name:<15} | {hr:<5} | {speed:<5} | {dist:.2f} km")
            else:
                print(f"{name:<15} | {hr:<5} | {speed:<5} | Target already met")

if __name__ == "__main__":
    data, trained_model = understand_activities()
    if data is not None:
        visualize_performance_metrics(data, trained_model)
        solve_for_duration(trained_model, target_te=3.5)
        solve_for_distance(trained_model, target_te=3.5)