import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Close all existing plots to avoid clutter
plt.close('all')


def clean_data_v1(file_path, expected_number_of_fields):
    cleaned_out_lines = []
    Invalidlines = []
    Index =[]
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Identify lines to clean out and split the data, skipping the first 11 lines
    for i, line in enumerate(lines[11:], start=12):  #The start=12 parameter specifies that the counter should start at 12 instead of the default 0.
        if len(line.split(',')) == expected_number_of_fields:
            pass
        else:
            cleaned_out_lines.append((i, line))
            print(f"Line {i} has {len(line.split(','))} fields instead of {expected_number_of_fields}")
            
            print(f"Line {i}: {line}") 
    # Assume we are working with the first and the second invalid line (indices 0 and 13149)
    for i in range (len(cleaned_out_lines)):
        Index.append(cleaned_out_lines[i][0])
    # print(f'The Index is {Index}')

    # Create new variables to store the cleaned data
    StraightLines =  lines[Index[0]:Index[1]-1] + lines[Index[1]:Index[2]-1] + lines[Index[2]:Index[3]-1] + lines[Index[3]:Index[4]-1] + lines[Index[4]:Index[5]-1]
    Perimeters = lines[Index[5]:Index[6]-1]
    Surface= StraightLines + Perimeters

    if len(Index) > 7:
        cuboid = lines[Index[12]:Index[13]-1]
        SurfacenCuboid = Surface + cuboid
    else:
        SurfacenCuboid = Surface
    
    return Surface, SurfacenCuboid


def process_data(data_lines):
    processed_data = []
    for line in data_lines:
        # Remove newline character and split by commas
        substrings = line.strip().split(',')
        # Convert substrings to floats and store in a list of lists
        float_values = [float(value) for value in substrings]
        processed_data.append(float_values)
    return processed_data



# Set font to bold Calibri
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
# Function to plot data
def plot_3d_data(datasets, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for data, color, marker, label, size, alpha in datasets:
        x_data = [point[0] for point in data]
        y_data = [point[1] for point in data]
        z_data = [point[2] for point in data]
        ax.scatter(x_data, y_data, z_data, c=color, marker=marker, label=label, s=size, alpha=alpha)  # s is the size of the scatter points, alpha is the transparency
    # Set axis labels
    ax.set_xlabel('X-Axis (mm)', fontweight='bold')
    ax.set_ylabel('Y-Axis (mm)', fontweight='bold')
    ax.set_zlabel('Z-Axis (mm)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    # Set axis limits to approximate Y/X = 250/150 ratio
    ax.set_xlim([0, 150])  # Set the X-axis range
    ax.set_ylim([0, 250])  # Set the Y-axis range
    # Add legend
    ax.legend()

    plt.show()


# Call the clean_data_v1 function
file_path_BeforeDED6screws = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_BeforeDED_12.09.2024_6screws.csv'
file_path_BeforeDED2screws = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_BeforeDED_12.09.2024_2screws.csv'

file_path_AfterDED6screws = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_AfterDED_12.13.2024_6screws.csv'
file_path_AfterDED2screws = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_AfterDED_12.13.2024_2screws.csv'
file_path_AfterMachining6screws = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_AfterMachining_12.13.2024_6screws.csv'
file_path_AfterMachining2screws = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_AfterMachining_12.13.2024_2screws.csv'


expected_number_of_fields = 3


# Clean and process data for 6 screws data measured on 12/09/2024
Surface_6ScrewsClamped, Surface_6ScrewsClamped_cuboid = clean_data_v1(file_path_BeforeDED6screws, expected_number_of_fields)
Surface_6ScrewsClamped_processed = process_data(Surface_6ScrewsClamped)
# Clean and process data for 2 screws data measured on 12/09/2024
Surface_2ScrewsClamped, Surface_2ScrewsClamped_cuboid= clean_data_v1(file_path_BeforeDED2screws, expected_number_of_fields)
Surface_2ScrewsClamped_processed = process_data(Surface_2ScrewsClamped)



# Clean and process data for 6 screws data -AfterDED - measured on 12/13/2024
Surface_6ScrewsClamped_AfterDED, Surface_6ScrewsClamped_AfterDED_cuboid = clean_data_v1(file_path_AfterDED6screws, expected_number_of_fields)
Surface_6ScrewsClamped_AfterDED_processed = process_data(Surface_6ScrewsClamped_AfterDED_cuboid)
# print(Surface_6ScrewsClamped_AfterDED_processed)
# print(f'The length of the Surface_6ScrewsClamped_AfterDED_processed data is {len(Surface_6ScrewsClamped_AfterDED_processed)}')
# # Clean and process data for 2 screws data -AfterDED - measured on 11/25/2024
Surface_2ScrewsClamped_AfterDED, Surface_2ScrewsClamped_AfterDED_cuboid = clean_data_v1(file_path_AfterDED2screws, expected_number_of_fields)
Surface_2ScrewsClamped_AfterDED_processed = process_data(Surface_2ScrewsClamped_AfterDED_cuboid)
# print(Surface_2ScrewsClamped_AfterDED_processed)
# print(f'The length of the Surface_2ScrewsClamped_AfterDED_processed data is {len(Surface_2ScrewsClamped_AfterDED_processed)}')


# # Clean and process data for 2 screws data -AfterDED - measured on 11/25/2024
Surface_6ScrewsClamped_AfterMachining, Surface_6ScrewsClamped_AfterMachining_cuboid = clean_data_v1(file_path_AfterMachining6screws, expected_number_of_fields)
Surface_6ScrewsClamped_AfterMachining_processed = process_data(Surface_6ScrewsClamped_AfterMachining_cuboid)
# Clean and process data for 2 screws data -AfterDED - measured on 11/25/2024
Surface_2ScrewsClamped_AfterMachining, Surface_2ScrewsClamped_AfterMachining_cuboid = clean_data_v1(file_path_AfterMachining2screws, expected_number_of_fields)
Surface_2ScrewsClamped_AfterMachining_processed = process_data(Surface_2ScrewsClamped_AfterMachining_cuboid)




# Plot data for 6 screws and 2 screws in the same plot
datasets = [
    (Surface_6ScrewsClamped_processed, 'darkorange', 'o', 'Before DED - 6 Screws Clamped', 25, 0.5),
    (Surface_2ScrewsClamped_processed, 'darkorange', 's', 'Before DED - 2 Screws Clamped', 25, 1.0),

    
    (Surface_6ScrewsClamped_AfterDED_processed, 'Blue', 'o', 'After DED - 6 Screws Clamped', 20, 0.5),
    (Surface_2ScrewsClamped_AfterDED_processed, 'blue', 's', 'After DED - 2 Screws Clamped', 20, 1.0),
    (Surface_6ScrewsClamped_AfterMachining_processed, 'green', 'o', 'After Machining - 6 Screws Clamped', 15, 0.5),
    (Surface_2ScrewsClamped_AfterMachining_processed, 'green', 's', 'After Machining - 2 Screws Clamped', 15, 1.0)
]
plot_3d_data(datasets, 'Surface - 6 Screws and 2 Screws Clamped')



