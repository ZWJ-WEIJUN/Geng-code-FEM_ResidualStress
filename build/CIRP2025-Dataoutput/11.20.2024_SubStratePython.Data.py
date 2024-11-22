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
    
    # Create new variables to store the cleaned data
    StraightLines =  lines[Index[0]:Index[1]-1] + lines[Index[1]:Index[2]-1] + lines[Index[2]:Index[3]-1] + lines[Index[3]:Index[4]-1] + lines[Index[4]:Index[5]-1]
    Perimeters = lines[Index[5]:Index[6]-1]
    
    return StraightLines, Perimeters


def process_data(data_lines):
    processed_data = []
    for line in data_lines:
        # Remove newline character and split by commas
        substrings = line.strip().split(',')
        # Convert substrings to floats and store in a list of lists
        float_values = [float(value) for value in substrings]
        processed_data.append(float_values)
    return processed_data


# Call the clean_data_v1 function
file_path = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/CIRP2025_SubstrateMeas_BeforeDED_11.19.2024_6screws.csv'
expected_number_of_fields = 3
StraightLines_6ScrewsClamped, Perimeters_6ScrewsClamped = clean_data_v1(file_path, expected_number_of_fields)

# Process the cleaned data as floats instead of strings
StraightLines_processed_6ScrewsClamped = process_data(StraightLines_6ScrewsClamped)
Perimeters_processed_6ScrewsClamped = process_data(Perimeters_6ScrewsClamped )
# Combine the processed data into a single list
Surface_6ScrewsClamped= StraightLines_processed_6ScrewsClamped + Perimeters_processed_6ScrewsClamped

# Convert processed data to DataFrame for plotting
df_straight_lines_6ScrewsClamped = pd.DataFrame(Surface_6ScrewsClamped, columns=['x', 'y', 'z'])

# Extract coordinates for the straight lines
x_surface_6ScrewsClamped = df_straight_lines_6ScrewsClamped['x']
y_surface_6ScrewsClamped = df_straight_lines_6ScrewsClamped['y']
z_surface_6ScrewsClamped = df_straight_lines_6ScrewsClamped['z']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the straight lines
ax.scatter(x_surface_6ScrewsClamped, y_surface_6ScrewsClamped, z_surface_6ScrewsClamped, color='red', marker='^', label='Before DED - 6 Screws Clamped')

# Set axis labels
ax.set_xlabel('X-Axis (mm)')
ax.set_ylabel('Y-Axis (mm)')
ax.set_zlabel('Z-Axis (mm)')





# Load the first CSV file (adjust the path as needed)
file_path_1 = r'/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/SubstrateMeasurements_10.17.2024_Converted_Cleaned.csv'
data1 = pd.read_csv(file_path_1)

# Load the second CSV file (adjust the path as needed)
file_path_2 = r'/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/build/CIRP2025-Dataoutput/SubstrateMeasurements_11.7.2024_.csv'
data2 = pd.read_csv(file_path_2)

# Assuming the columns in both CSV files are 'x', 'y', 'z'. Adjust if necessary.
# Extract coordinates for the first dataset
x1 = data1['x']
y1 = data1['y']
z1 = data1['z']

# Extract coordinates for the second dataset
x2 = data2['x']
y2 = data2['y']
z2 = data2['z']


# Plot the first set of points and connect them with a line (solid line, blue)
ax.scatter(x1, y1, z1, color='blue', marker='^', label='TU Wien - Grinding Surface')

# Plot the second set of points and connect them with a line (dashed line, red)
ax.scatter(x2, y2, z2, color='darkorange', marker='s', label='UC Davis - Milling Surface')
        
# Set axis labels
ax.set_xlabel('X-Axis (mm)')
ax.set_ylabel('Y-Axis (mm)')
ax.set_zlabel('Z-Axis (mm)')

# Add a legend to distinguish the two sets of points
ax.legend()

# Show the plot
plt.show()