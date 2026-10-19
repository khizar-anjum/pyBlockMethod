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

    # Create bottom floor surface
    x_floor = np.linspace(0, floor_size, 50)
    y_floor = np.linspace(0, floor_size, 50)
    X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
    Z_floor = np.zeros_like(X_floor)

    # Create top floor in 4 sections around the ramp cutout
    # Section 1: Left of ramp (x < ramp_x_offset)
    x_top_left = np.linspace(0, ramp_x_offset, 50)
    y_top_left = np.linspace(0, floor_size, 50)
    X_top_left, Y_top_left = np.meshgrid(x_top_left, y_top_left)
    Z_top_left = np.ones_like(X_top_left) * ramp_height

    # Section 2: Right of ramp (x > ramp_x_offset + ramp_length)
    x_top_right = np.linspace(ramp_x_offset + ramp_length, floor_size, 50)
    y_top_right = np.linspace(0, floor_size, 50)
    X_top_right, Y_top_right = np.meshgrid(x_top_right, y_top_right)
    Z_top_right = np.ones_like(X_top_right) * ramp_height

    # Section 3: Front of ramp (y < ramp_y_offset, in ramp x range)
    x_top_front = np.linspace(ramp_x_offset, ramp_x_offset + ramp_length, 50)
    y_top_front = np.linspace(0, ramp_y_offset, 50)
    X_top_front, Y_top_front = np.meshgrid(x_top_front, y_top_front)
    Z_top_front = np.ones_like(X_top_front) * ramp_height

    # Section 4: Back of ramp (y > ramp_y_offset + ramp_width, in ramp x range)
    x_top_back = np.linspace(ramp_x_offset, ramp_x_offset + ramp_length, 50)
    y_top_back = np.linspace(ramp_y_offset + ramp_width, floor_size, 50)
    X_top_back, Y_top_back = np.meshgrid(x_top_back, y_top_back)
    Z_top_back = np.ones_like(X_top_back) * ramp_height

    # Create the 3D plot
    fig = go.Figure()

    # Add bottom floor (light gray)
    fig.add_trace(
        go.Surface(
            x=X_floor,
            y=Y_floor,
            z=Z_floor,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            opacity=0.9,
            showscale=False,
            name="Bottom Floor",
        )
    )

    # Add top floor sections (darker gray)
    fig.add_trace(
        go.Surface(
            x=X_top_left,
            y=Y_top_left,
            z=Z_top_left,
            colorscale=[[0, "darkgray"], [1, "darkgray"]],
            opacity=0.9,
            showscale=False,
            name="Top Floor Left",
        )
    )

    fig.add_trace(
        go.Surface(
            x=X_top_right,
            y=Y_top_right,
            z=Z_top_right,
            colorscale=[[0, "darkgray"], [1, "darkgray"]],
            opacity=0.9,
            showscale=False,
            name="Top Floor Right",
        )
    )

    fig.add_trace(
        go.Surface(
            x=X_top_front,
            y=Y_top_front,
            z=Z_top_front,
            colorscale=[[0, "darkgray"], [1, "darkgray"]],
            opacity=0.9,
            showscale=False,
            name="Top Floor Front",
        )
    )

    fig.add_trace(
        go.Surface(
            x=X_top_back,
            y=Y_top_back,
            z=Z_top_back,
            colorscale=[[0, "darkgray"], [1, "darkgray"]],
            opacity=0.9,
            showscale=False,
            name="Top Floor Back",
        )
    )

    # Add triangular ramp with smooth surface (DARK BLUE)
    fig.add_trace(
        go.Surface(
            x=X_ramp,
            y=Y_ramp,
            z=Z_ramp,
            surfacecolor=np.ones_like(Z_ramp),
            colorscale=[[0, "darkblue"], [1, "darkblue"]],
            opacity=1.0,
            showscale=False,
            name="Ramp Top",
        )
    )

    # Left triangular side cover (LIGHT BLUE)
    fig.add_trace(
        go.Mesh3d(
            x=[ramp_x_offset, ramp_x_offset + ramp_length, ramp_x_offset],
            y=[ramp_y_offset, ramp_y_offset, ramp_y_offset],
            z=[ramp_height, 0, 0],
            i=[0],
            j=[1],
            k=[2],
            color="lightblue",
            opacity=0.8,
            showscale=False,
        )
    )

    # Right triangular side cover (ROYAL BLUE)
    fig.add_trace(
        go.Mesh3d(
            x=[ramp_x_offset, ramp_x_offset + ramp_length, ramp_x_offset],
            y=[
                ramp_y_offset + ramp_width,
                ramp_y_offset + ramp_width,
                ramp_y_offset + ramp_width,
            ],
            z=[ramp_height, 0, 0],
            i=[0],
            j=[1],
            k=[2],
            color="royalblue",
            opacity=0.8,
            showscale=False,
        )
    )

    # Back rectangular face (NAVY BLUE)
    fig.add_trace(
        go.Mesh3d(
            x=[ramp_x_offset, ramp_x_offset, ramp_x_offset, ramp_x_offset],
            y=[
                ramp_y_offset,
                ramp_y_offset + ramp_width,
                ramp_y_offset + ramp_width,
                ramp_y_offset,
            ],
            z=[0, 0, ramp_height, ramp_height],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="navy",
            opacity=0.8,
            showscale=False,
        )
    )

    # Add door on bottom floor (right side of stairs, along y-axis edge)
    door_width = 6
    door_height = 2.5
    door_x = floor_center  # Center of the edge
    door_y = 0  # At the opposite edge along y-axis

    # Door frame (brown) - oriented along y-axis
    fig.add_trace(
        go.Mesh3d(
            x=[
                door_x - door_width / 2,
                door_x + door_width / 2,
                door_x + door_width / 2,
                door_x - door_width / 2,
            ],
            y=[door_y, door_y, door_y, door_y],
            z=[0, 0, door_height, door_height],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="saddlebrown",
            opacity=1.0,
            showscale=False,
            name="Door",
        )
    )

    # Simple trajectory: From upper left corner to door
    traj_x = [
        5,  # Start at upper left corner
        ramp_x_offset,  # Walk toward ramp on top floor
        ramp_x_offset + ramp_length / 2,  # Middle of ramp
        ramp_x_offset + ramp_length + 2,  # Bottom of ramp
        floor_center + 10,  # Walk toward door
        floor_center,  # At door
    ]
    traj_y = [
        floor_size - 5,  # Upper corner
        floor_center + 3,  # Approaching ramp
        floor_center,  # On ramp
        floor_center - 2,  # After ramp
        15,  # Near door
        0,  # At door
    ]
    traj_z = [
        ramp_height + 0.2,  # On top floor
        ramp_height + 0.2,  # Still on top floor
        ramp_height / 2,  # Middle of ramp descent
        0.2,  # On bottom floor
        0.2,  # Walking to door
        0.2,  # At door
    ]

    fig.add_trace(
        go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode="lines+markers",
            line=dict(color="red", width=6),
            marker=dict(size=2, color="red"),
            name="Exit Path",
        )
    )

    # Update layout
    fig.update_layout(
        # title='Triangular Ramp on Floor',
        scene=dict(
            xaxis_title="Length",
            yaxis_title="Width",
            zaxis_title="Height",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.2),
        ),
        width=800,
        height=600,
    )

    return fig


if __name__ == "__main__":
    fig = create_triangular_ramp()
    fig.show()
