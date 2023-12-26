import matplotlib.pyplot as plt
import numpy as np
import torch

# Constants
material_y_width = 100
material_x_z_width = 10
distance_between_electrons = 5
# The number_of_electrons_wide should be an odd number so that we get an electron at the middle of each layer
number_of_electrons_wide = material_x_z_width // distance_between_electrons * 2 + 1
# It is easier to compute the electric field at the electron positions if the electron layers have integer value y-coordinate
distance_between_layers = 10
# Make sure that material_y_width is divisible by distance_between_layers
material_y_width = material_y_width + material_y_width % distance_between_layers
number_of_layers = material_y_width // distance_between_layers + 1
charge_electron = 1 / number_of_electrons_wide ** 2  # this is to ensure the amount of charge per layer is constant
TIME_END = 100
TIME_STEPS = 100
DT = TIME_END / TIME_STEPS
start_y = 25 # this is where we will start plotting y from
# It is easier to compute the electric field at the electron positions if the evaluation points match the integer values of the y-coordinate
number_of_evaluation_points = start_y + material_y_width + 1
c = 5  # speed of light
hookes_constant = 0.1
mass_electron = 1
damping_constant = 0
sigma = 5
# Allways use the GPU for tensor operations
torch.set_default_device('cuda') 

# Initialise the electric field
def original_electric_field(t, y):
    """This function calculates the original electric field of a light pulse along the y-axis.

    Args:
        t (float): The t-coordinate of the current time step
        y (torch.tensor): The y-coordinates of the evaluation points. The tensor has size [number_of_evaluation_points]

    Returns:
        torch.tensor: The electric field of the light pulse at time t. The tensor has size [number_of_evaluation_points]
    """
    # The pulse is a gaussian, cf. https://wikimedia.org/api/rest_v1/media/math/render/svg/c35de83757f2e4b6bf6cc570ada72ada21705b23 
    coefficient = 50 / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))
    exponential_term = torch.exp(-0.5 * ((y + c * t - start_y) / sigma) ** 2)
    return coefficient * exponential_term


def set_electron_y_positions():
    """This function sets the y coordinates of all electrons for all time steps

    Returns:
        torch.tensor: Initial y coordinates of all electrons at all times
                        with shape [t_steps, n_el_wide, n_layers, n_el_wide]
    """
    # Create a tensor of layer positions, with shape [n_layers]
    y_positions = torch.linspace(0, -material_y_width, number_of_layers)

    # We need to reshape and repeat this tensor to match the desired dimensions
    # Firstly, add x- and z-dimensions to y_positions for broadcasting, making it [1, n_layers, 1]
    y_e = y_positions.view(1, -1, 1)

    # Repeat the y-coordinates to match the dimensions of x and z electrons
    # The final shape will be [e_el_wide, n_layers, n_el_wide]
    y_e = y_e.repeat(number_of_electrons_wide, 1, number_of_electrons_wide)

    # Now add the time step dimension and repeat across it
    # The final shape will be [TIME_STEPS, n_el_wide, n_layers, n_el_wide]
    return y_e.unsqueeze(0).repeat(TIME_STEPS, 1, 1, 1)


def set_electron_xz_positions():
    """This function sets the initial x and z coordinates of all electrons for all time steps

    Returns:
        torch.tensor: Initial x coordinates of all electrons at all times
                        with shape [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide]
        torch.tensor: Initial z coordinates of all electrons at all times
                        with shape [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide]
    """
    if number_of_electrons_wide == 1:
        # Single electron at the center for each layer and each time step
        x_e = torch.zeros((TIME_STEPS, 1, number_of_layers, 1))
        z_e = torch.zeros((TIME_STEPS, 1, number_of_layers, 1))
    else:
        # Create a grid of electron positions
        x_positions = torch.linspace(-material_x_z_width, material_x_z_width, number_of_electrons_wide)
        z_positions = torch.linspace(-material_x_z_width, material_x_z_width, number_of_electrons_wide)
        x_e, z_e = torch.meshgrid(x_positions, z_positions, indexing='ij')

        # Expand dimensions to match the final shape and repeat for layers
        x_e = x_e.unsqueeze(1).repeat(1, number_of_layers, 1)
        z_e = z_e.unsqueeze(1).repeat(1, number_of_layers, 1)

        # Now add the time step dimension and repeat across it
        # The final shape will be [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide]
        x_e = x_e.unsqueeze(0).repeat(TIME_STEPS, 1, 1, 1)
        z_e = z_e.unsqueeze(0).repeat(TIME_STEPS, 1, 1, 1)

    return x_e, z_e


def electrons_electric_field(electron_positions, electron_accelerations, t_step):
    """This function computes the electric field of the moving electrons

    Args:
        electron_positions (Tensor):    The positions of the electrons in 3D space for all time steps
                                        The size is [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3]
        electron_accelerations (Tensor): The accelerations of all electrons for all time steps
                                        The size is [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3]
        t_step (int): The current time step

    Returns:
        torch.Tensor: The electric field due to the electrons for all evaluation points. The size is [number_of_evaluation_points]
    """
    t_p = t_step * DT # The time of the current time step
    # Fist, we compute the distances between the evaluation points and the electron positions at the given time
    # To compute the distenaces, we must flatten the electron positions
    electron_positions_flattend = electron_positions.flatten(start_dim=1, end_dim=3)
    
    # Compute the Euclidean spatial distances
    distances = torch.norm(electron_positions_flattend.unsqueeze(1) - eval_points.unsqueeze(1), dim=-1)
    distance_mask = distances > 0 # This is to prevent a point from experiences a field due to an electron located there.
    number_of_all_electrons = distances.size()[2]

    # Compute the past time steps
    t_e = torch.arange(TIME_STEPS) * DT
    past_time_mask = t_e <= t_p 
    # Compute the time differences between the current time and all the past time steps
    delta_t = t_p - t_e
    # Adjust the shape to match the distances
    delta_t = delta_t.unsqueeze(1).unsqueeze(1).repeat(1, number_of_evaluation_points, number_of_all_electrons)
    past_time_mask = past_time_mask.unsqueeze(1).unsqueeze(1).repeat(1, number_of_evaluation_points, number_of_all_electrons)

    # Calculate the retardation time for each distance
    retardation_time = delta_t - distances / c
    
    # If delta_t is approximately r / c, this electron is currently contributing to the electric field of this point
    relevant_past_steps = (-DT < retardation_time) & (retardation_time < DT)

    # Adjust the shape of the accelerations, too
    electron_accel_flattend = electron_accelerations.flatten(start_dim=1)
    electron_accel_flattend = electron_accel_flattend.unsqueeze(1).repeat(1, number_of_evaluation_points, 1)
    relevant_accel = electron_accel_flattend * relevant_past_steps # Sets all irrelevant accelerations to 0

    # Calculate the contribution of the electric field
    # The size of the contribution is given in the Feynman lectures, vol 1, eq 29-1
    # https://www.feynmanlectures.caltech.edu/I_29.html
    field_contributions = - charge_electron * (xy_distances / distances) * (relevant_accel / distances)
    # This is to prevent a point from experiences a field due to an electron located there.
    field_contributions = field_contributions * distance_mask
    # This is to ensure that the point is at a later time than the electron.
    field_contributions = field_contributions * past_time_mask

    # Sum the contributions for this time step
    return field_contributions.sum(dim=(0, 2))


def force(total_electric_field, el_pos, velocity, electron_original_z_positions):
    """This function computes the forces all the electrons experience due to the electric field.
        It computes the force only for the electrons in the middle of each layer but applies it to all electrons in the layer.

    Args:
        total_electric_field (Tensor):  The total electric field at each evaliation point at the current time step 
                                        The size is [number_of_evaluation_points].
        el_pos (Tensor):    The position vectors of all electrons at a given time step
                            The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3].
        velocity (Tensor):  The velocities of all electrons at a given time step
                            The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide].
                            The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide]
        electron_original_z_positions (Tensor): The original z-coordinates of all electrons
                                                The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide].

    Returns:
        torch.Tensor: The forces all the electrons experience. The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide].
    """
    # The damping, only relevant if damping_constant > 0.
    damping = - damping_constant * velocity

    # For Hooke's law, we need the amounts by which the electrons are displaced from their original positions.
    # el_pos has the size [number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3],
    # but we need only the z-coordinate
    hookes_law = - hookes_constant * (el_pos[..., 2] - electron_original_z_positions)

    # The total_electric_field is given on the evaluation points as a tensor of size [number_of_evaluation_points].
    # To use it, we must broadcast it into the right shape.
    # We need the field at the electrons y-coordinates. We convert the coordinates into indices
    index = el_pos[..., 1] - start_y
    field = total_electric_field[index.long()]

    return field + hookes_law + damping


def new_positions(el_pos, velocity, accel):
    """This function updates the z-coordinates of all electrons at the current time step.

    Args:
        el_pos (Tensor):    The position vectors of all electrons at the current time step
                            The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3].
        velocity (Tensor):  The velocoties of all electrons at a given time step
                            The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide].
        accel (Tensor): The accelerations of all electrons at a given time step
                        The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide].

    Returns:
        torch.Tensor:   The updated position vectors of all electrons at the current time step
                        The size is [number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3]
    """
    # Calculates the force the electron in the middle of the layer experiences
    # z is the previous z positions of the electrons in the middle of each layer
    # The output shall have the size [number_of_electrons_wide, number_of_layers, number_of_electrons_wide].
    # Constant term
    z_const = el_pos[..., 2]
    # Linear term
    z_linear = velocity * DT
    # Acceleration term
    z_accelerated = 0.5 * accel * DT ** 2
    # We change only the z-coordinates
    el_pos[...,  2] = z_const + z_linear + z_accelerated
    
    return el_pos


if __name__ == '__main__':
    # The electron y positions are a tensor of size [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide]
    electron_y_positions = set_electron_y_positions()
    # The electron x and z positions are tensors of size [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide]
    electron_x_positions, electron_z_positions = set_electron_xz_positions()
    electron_original_z_positions = electron_z_positions[0] # We need those later for Hooke's law.
    # The electron positions in a single tensor of shape [TIME_STEPS, number_of_electrons_wide, number_of_layers, number_of_electrons_wide, 3]
    electron_positions = torch.stack((electron_x_positions, electron_y_positions, electron_z_positions), dim=-1)
    # We plot only the electrons in the yz-plane, i.e. electrons with x = 0. They have the middle index in the x-coordinate
    middle_index = number_of_electrons_wide // 2

    # We are only going to evaluate the field along the line x = 0, z = 0 (the y-axis).
    # These are the y values for those points as a tensor of size [number_of_evaluation_points]
    y_axis = torch.linspace(-material_y_width, start_y, number_of_evaluation_points)
    y_for_plot = y_axis.cpu().numpy()
    # Reshape evaluation points to make them 3d-vectors with x- and z-coordinate 0 and size [number_of_evaluation_points, 3] 
    eval_points = y_axis.unsqueeze(-1)
    eval_points = torch.cat([torch.zeros_like(eval_points), eval_points, torch.zeros_like(eval_points)], dim=-1)

    # According to the formular given in the Feynman lectures, vol 1, eq 29-1
    # https://www.feynmanlectures.caltech.edu/I_29.html,
    # we need the sin of the angle θ from the axis of the electrons motion (the z-direction) and the vector from the evaluation point to the electron.
    # The sin is the distance in the xy-plane divided by the total distance.
    # Since the electrons move only in z-direction, the distances in the xy-plane between the evaluation points and the electrons do not change.
    # Therefore, we can compute them here.
    electron_xy_positions = torch.stack((electron_x_positions, electron_y_positions), dim=-1)
    electron_xy_positions_flattend = electron_positions.flatten(start_dim=1, end_dim=3)
    xy_distances = torch.norm(electron_xy_positions_flattend.unsqueeze(1) - eval_points.unsqueeze(1), dim=-1)
    
    # These keep track of the z-coordinate of the velocity and acceleration of the electrons for all previous time steps.
    # Initialised to contain 0s in the same shape as the single coordinate position tensors
    # We assume all electrons in a layer have the same values for these for simplicity
    z_velocity = torch.zeros_like(electron_y_positions)
    z_accel = torch.zeros_like(electron_y_positions)
    
    # We plot only the yz-plane. Therefore, we do not need a 3D plot.
    fig = plt.figure(figsize=plt.figaspect(0.5)*2) # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    ax = fig.add_subplot(111)

    plt.ion()

    for step in range(TIME_STEPS):
        # Clear the screen
        ax.cla()
        # The t-coordinate of the current time step
        t = step * DT

        # We are going to calculate the field along the line x = 0, z = 0
        # This is the original electric field of the light pulse at time t as a tensor of size [number_of_evaluation_points]
        ef_original = original_electric_field(t, y_axis) 
        # This is the electric field created by the electrons at time t as a tensor of size [number_of_evaluation_points]
        ef_due_to_electrons = electrons_electric_field(electron_positions, z_accel, step)
        ef_combined = ef_original + ef_due_to_electrons
        
        # Now we can plot the electric fields
        ax.plot(y_for_plot, ef_combined.cpu().numpy(), '.m')
        ax.plot(y_for_plot, ef_original.cpu().numpy(), '.r')
        ax.plot(y_for_plot, ef_due_to_electrons.cpu().numpy(), '.b')

        # Now, we can update the accelerations, z-positions and velocities of all electrons for the current time step.
        # They depent on the previous time step.
        if step > 0:
            z_accel[step] = force(ef_combined, electron_positions[step - 1], z_velocity[step - 1], electron_original_z_positions) / mass_electron
            electron_positions[step] = new_positions(electron_positions[step - 1], z_velocity[step - 1], z_accel[step])
            z_velocity[step] = z_velocity[step - 1] + z_accel[step] * DT
        else: # At the first time step there is no previous one.
            z_accel[step] = force(ef_combined, electron_positions[step], z_velocity[step], electron_original_z_positions) / mass_electron
            electron_positions[step] = new_positions(electron_positions[step], z_velocity[step], z_accel[step])
            z_velocity[step] = z_accel[step] * DT

        # For the plotting, we must update the electron_z_positions, too.
        electron_z_positions = electron_positions[..., 2]
        # Extract y and z positions for x = 0 and the given time step
        yz_plane_y_positions = electron_y_positions[step, middle_index, :, :].cpu().numpy()
        yz_plane_z_positions = electron_z_positions[step, middle_index, :, :].cpu().numpy()

        # Flatten the tensors to 1D arrays for plotting
        yz_plane_y_positions_flat = yz_plane_y_positions.flatten()
        yz_plane_z_positions_flat = yz_plane_z_positions.flatten()

        # Now you can plot the electrons
        plt.scatter(yz_plane_y_positions_flat, yz_plane_z_positions_flat, c='b', alpha=0.5)

        # Adjust axis limits
        ax.set_ylim([-material_x_z_width, material_x_z_width])
        ax.set_xlim([-material_y_width, start_y])
        ax.set_aspect('auto', adjustable='box')
            
        ax.set_ylabel('Z')
        ax.set_xlabel('Y')

        plt.draw()
        plt.pause(0.1)

        if not plt.get_fignums():
            break
