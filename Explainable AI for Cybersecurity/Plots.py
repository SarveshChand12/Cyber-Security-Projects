import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

# Sample data generation (replace with actual model outputs)
def generate_sample_data():
    """Generate sample SHAP and Grad-CAM outputs for visualization"""
    # SHAP values (feature importance)
    shap_values = np.random.rand(224, 224)  # Replace with actual SHAP output
    
    # Grad-CAM activation map
    grad_cam = np.random.rand(224, 224)  # Replace with actual Grad-CAM output
    
    # Model prediction probabilities
    prediction_probs = [0.85, 0.15]  # Example: [Benign, Malicious]
    
    return shap_values, grad_cam, prediction_probs

# Plot SHAP Feature Importance
def plot_shap(shap_values):
    """Plot SHAP feature importance using Plotly"""
    fig = px.imshow(
        shap_values,
        color_continuous_scale='viridis',
        title="SHAP Feature Importance",
        labels=dict(x="Feature Index", y="Feature Index", color="Importance")
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Importance"))
    return fig

# Plot Grad-CAM Activation Map
def plot_grad_cam(grad_cam):
    """Plot Grad-CAM activation map using Plotly"""
    fig = px.imshow(
        grad_cam,
        color_continuous_scale='viridis',
        title="Grad-CAM Activation Map",
        labels=dict(x="Width", y="Height", color="Activation")
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Activation"))
    return fig

# Plot Prediction Probabilities
def plot_prediction_probs(prediction_probs):
    """Plot prediction probabilities using Matplotlib"""
    labels = ['Benign', 'Malicious']
    plt.figure(figsize=(6, 4))
    plt.bar(labels, prediction_probs, color=['green', 'red'])
    plt.title("Prediction Probabilities")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.savefig('prediction_probs.png')  # Save as image
    plt.close()

# Interactive Dashboard
def create_dashboard():
    """Create an interactive dashboard using Dash"""
    app = dash.Dash(__name__)
    
    # Generate sample data
    shap_values, grad_cam, prediction_probs = generate_sample_data()
    
    # Dashboard layout
    app.layout = html.Div([
        html.H1("Explainable AI (XAI) for Malware Detection", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                dcc.Graph(id='shap-plot', figure=plot_shap(shap_values))
            ], className='six columns'),
            html.Div([
                dcc.Graph(id='grad-cam-plot', figure=plot_grad_cam(grad_cam))
            ], className='six columns')
        ], className='row'),
        html.Div([
            html.Img(src='./prediction_probs.png', style={'width': '50%', 'margin': 'auto', 'display': 'block'})
        ])
    ])
    
    # Run the app
    app.run_server(debug=False)

# Main function
if __name__ == "__main__":
    # Generate and save prediction probabilities plot
    _, _, prediction_probs = generate_sample_data()
    plot_prediction_probs(prediction_probs)
    
    # Launch the dashboard
    create_dashboard()