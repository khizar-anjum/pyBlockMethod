import plotly.graph_objects as go
import numpy as np

def create_triangular_ramp():
    # Define ramp dimensions
    ramp_length = 16
    ramp_width = 16
    ramp_height = 3

    # Create larger square floor
    floor_size = 60
    floor_center = floor_size / 2

    # Center ramp on floor
    ramp_x_offset = floor_center - ramp_length / 2
    ramp_y_offset = floor_center - ramp_width / 2

    # Create triangular ramp surface
    x_ramp = np.linspace(ramp_x_offset, ramp_x_offset + ramp_length, 50)
    y_ramp = np.linspace(ramp_y_offset, ramp_y_offset + ramp_width, 40)
    X_ramp, Y_ramp = np.meshgrid(x_ramp, y_ramp)

    # Create triangular profile: height decreases linearly from back to front
    Z_ramp = ramp_height * (1 - (X_ramp - ramp_x_offset) / ramp_length)

    # Create floor surface
    x_floor = np.linspace(0, floor_size, 50)
    y_floor = np.linspace(0, floor_size, 50)
    X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
    Z_floor = np.zeros_like(X_floor)

    # Create the 3D plot
    fig = go.Figure()

    # Add floor
    fig.add_trace(go.Surface(
        x=X_floor,
        y=Y_floor,
        z=Z_floor,
        colorscale='Greys',
        opacity=0.8,
        showscale=False,
        name='Floor'
    ))

    # Add triangular ramp with smooth surface (DARK BLUE)
    fig.add_trace(go.Surface(
        x=X_ramp,
        y=Y_ramp,
        z=Z_ramp,
        surfacecolor=np.ones_like(Z_ramp),
        colorscale=[[0, 'darkblue'], [1, 'darkblue']],
        opacity=1.0,
        showscale=False,
        name='Ramp Top'
    ))

    # Left triangular side cover (LIGHT BLUE)
    fig.add_trace(go.Mesh3d(
        x=[ramp_x_offset, ramp_x_offset + ramp_length, ramp_x_offset],
        y=[ramp_y_offset, ramp_y_offset, ramp_y_offset],
        z=[ramp_height, 0, 0],
        i=[0],
        j=[1],
        k=[2],
        color='lightblue',
        opacity=0.8,
        showscale=False
    ))

    # Right triangular side cover (ROYAL BLUE)
    fig.add_trace(go.Mesh3d(
        x=[ramp_x_offset, ramp_x_offset + ramp_length, ramp_x_offset],
        y=[ramp_y_offset + ramp_width, ramp_y_offset + ramp_width, ramp_y_offset + ramp_width],
        z=[ramp_height, 0, 0],
        i=[0],
        j=[1],
        k=[2],
        color='royalblue',
        opacity=0.8,
        showscale=False
    ))

    # Back rectangular face (NAVY BLUE)
    fig.add_trace(go.Mesh3d(
        x=[ramp_x_offset, ramp_x_offset, ramp_x_offset, ramp_x_offset],
        y=[ramp_y_offset, ramp_y_offset + ramp_width, ramp_y_offset + ramp_width, ramp_y_offset],
        z=[0, 0, ramp_height, ramp_height],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='navy',
        opacity=0.8,
        showscale=False
    ))

    # Update layout
    fig.update_layout(
        # title='Triangular Ramp on Floor',
        scene=dict(
            xaxis_title='Length',
            yaxis_title='Width',
            zaxis_title='Height',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.2)
        ),
        width=800,
        height=600
    )

    return fig

if __name__ == "__main__":
    fig = create_triangular_ramp()
    fig.show()
