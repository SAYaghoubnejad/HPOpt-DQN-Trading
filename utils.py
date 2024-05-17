import pickle
import random
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import torch
import os
import time


def save_pkl(path, obj):
    with open(path, 'wb') as writer:
        pickle.dump(obj, writer, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as reader:
        obj = pickle.load(reader)
    return obj


def set_random_seed(seed=None):
    if seed is None:
        # Generate a new random seed, e.g., based on the current time
        seed = int(time.time()) + os.getpid()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_hyperparams(params, optimizer, init_set):
    for index, param in enumerate(params):
        # Your existing data
        x = np.arange(len(optimizer.Y.flatten()))
        y = optimizer.X[:, index]
        returns = optimizer.Y.flatten()
        
        # Map labels to symbols
        symbols = np.array(['Random Initiation' if i < init_set else 'Bayesian Optimization' for i in range(len(x))])
        
        # Create the scatter plot with returns as color and symbols for 'Train' and 'Test'
        fig = px.scatter(
            x=x,
            y=y,
            color=returns,
            symbol=symbols,  # Use symbols here
            # labels={"x": "Iteration", "y": "Log2 of Batch Size"},
            # title="Return Sensitivity on Batch Size"
        )
        
        # Update plot layout
        fig.update_layout(title=f'Return per {param} Using {optimizer.optimizer_name} Optimizer',
            xaxis_title='Iteration',
            yaxis_title=f'{param}',
            legend_title="Phase of Optimization",
            font=dict(size=10))
        
        # Update trace properties
        fig.update_traces(marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                color='black',
                width=1
            )
        ))
        
        # Update color axis
        fig.update_layout(coloraxis_colorbar=dict(
            title="Return (%)",
            x=1.01,  # Adjust the position
            y=0.35,
            lenmode='fraction',
            len=0.75
        ))
        
        # Add annotations for 'Train' and 'Test'
        # fig.add_annotation(
        #     x=init_set - 0.15 * max(x) , y=max(y)*1.1, text="Random Initiation", showarrow=False, yshift=10
        # )
        # fig.add_annotation(
        #     x=init_set + 0.1 * max(x), y=max(y)*1.1, text="Bayesian Optimization", showarrow=False, yshift=10
        # )
        
        # Add a vertical line to separate 'Train' and 'Test'
        fig.add_vline(x=init_set - 0.5, line_width=2, line_dash="dash", line_color="black", opacity=0.3)
        
        # Show the plot
        fig.show()

def plot_optimization_progress(optimizer, init_set):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, init_set+1), y=optimizer.history['Y'][:init_set+1], mode='lines', name='Random Initiation'))
    fig.add_trace(go.Scatter(x=np.arange(init_set, len(optimizer.history['Y'])), y=optimizer.history['Y'][init_set:], mode='lines', name=optimizer.optimizer_name))

    fig.update_layout(
        xaxis_title="Iteration No.",
        yaxis_title="Best Loss",
        title=f"Best Loss Found Till Each Iteration Using {optimizer.optimizer_name} Optimizer",
        font=dict(size=10)
    )
    fig.add_vline(x=init_set, line_width=2, line_dash="dash", line_color="black", opacity=0.3)
    
    # Show the plot    
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, init_set), y=optimizer.Y.flatten()[:init_set], mode='lines', name='Random Initiation'))
    fig.add_trace(go.Scatter(x=np.arange(init_set-1, optimizer.Y.flatten().shape[-1]), y=optimizer.Y.flatten()[init_set-1:], mode='lines', name=optimizer.optimizer_name))

    fig.update_layout(
        xaxis_title="Iteration No.",
        yaxis_title="Return (%)",
        title=f"Progress of Bayesian Optimization Using {optimizer.optimizer_name} Optimizer",
        font=dict(size=10),
    )

    fig.add_vline(x=init_set - 1, line_width=2, line_dash="dash", line_color="black", opacity=0.3)

    # Show the plot    
    fig.show()

def plot_prob_accusition(optimizer, plot_path=None):
    PI_values = [item[0]['PI'] for item in optimizer.acq_history]
    EI_values = [item[0]['EI'] for item in optimizer.acq_history]
    UCB_values = [item[0]['UCB'] for item in optimizer.acq_history]

    chosen = [item[1] for item in optimizer.acq_history]
    chosen_y = [item[0][item[1]] for item in optimizer.acq_history]
    colors = {}
    color = 0
    for i in chosen:
        if i in colors.keys():
            continue
        colors[i] = px.colors.qualitative.Plotly[color]
        color += 1

    for i in optimizer.acq_history[0][0].keys():
        if i in colors.keys():
            continue
        colors[i] = px.colors.qualitative.Plotly[color]
        color += 1

    fig = px.scatter(
        x=np.arange(0, len(chosen_y)),
        y=chosen_y,
        color=chosen
    )

    fig.update_traces(marker=dict(
            size=8,
        ))

    fig.add_trace(go.Scatter(x=np.arange(0, len(PI_values)), y=PI_values, mode='lines', name='PI', line=dict(color=colors['PI'])))
    fig.add_trace(go.Scatter(x=np.arange(0, len(EI_values)), y=EI_values, mode='lines', name='EI', line=dict(color=colors['EI'])))
    fig.add_trace(go.Scatter(x=np.arange(0, len(UCB_values)), y=UCB_values, mode='lines', name='UCB', line=dict(color=colors['UCB'])))

    fig.update_layout(
        xaxis_title="Iteration No.",
        yaxis_title="Probabilities",
        title=f"Probabilities of each Acquisition Function to Choose the Next Candidate Using {optimizer.optimizer_name} Optimizer",
        font=dict(size=10),
    )

    # Display the figure
    fig.show()
    if plot_path:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        # Calculate the width and height in pixels (300 DPI for high-quality print)
        dpi = 300
        width_inches = 3.5  # Single column width in inches
        height_inches = 2.625  # Adjusted height in inches for a 4:3 aspect ratio

        # Convert inches to pixels
        width_pixels = int(width_inches * dpi)
        height_pixels = int(height_inches * dpi)

        # Save the plot to files in different formats with dimensions suitable for single column figures
        fig.write_html(f'{plot_path}/{optimizer.optimizer_name}_accusition_prob.html')
        save_format = ['pdf', 'svg', 'jpg']
        for f in save_format:
            fig.write_image(f'{plot_path}/{optimizer.optimizer_name}_accusition_prob.{f}', format=f, width=width_pixels, height=height_pixels)