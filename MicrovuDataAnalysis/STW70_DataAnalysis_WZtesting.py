import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def clean_data(file_path, expected_number_of_fields):
    # Variables definition and initialization
    cleaned_out_lines = []
    Invalidlines = []

    # Open and read the CSV file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Identify lines to clean out and split the data
    for i, line in enumerate(lines):
        if len(line.split(',')) == expected_number_of_fields:
            pass
        else:
            cleaned_out_lines.append((i+1, line))
            print(f"Line {i+1} has {len(line.split(','))} fields instead of {expected_number_of_fields}")
            print(f"Line {i+1}: {line}")

    # Assume we are working with the first and the second invalid line (indices 0 and 13149)
    first_invalid_index = cleaned_out_lines[0][0] - 1  # line 1 (index 0)
    second_invalid_index = cleaned_out_lines[1][0] - 1  # line 13150 (index 13149)
    print(f"First invalid index: {first_invalid_index}")
    print(f"Second invalid index: {second_invalid_index}")

    # Split the data into two sections based on invalid lines
    section1_lines = lines[first_invalid_index + 1:second_invalid_index]
    section2_lines = lines[second_invalid_index + 1:]
    print(f"Section 1 has {len(section1_lines)} lines")
    print(f"Section 2 has {len(section2_lines)} lines")

    # Create new variables to store the cleaned data
    STW70_LP = section1_lines
    STW70_FR = section2_lines

    return STW70_LP, STW70_FR


if __name__ == '__main__':
    # Call the clean_data function
    file_path = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/MicrovuDataAnalysis/STW70_LP_and_FR_Ctrl_Spline.csv'
    expected_number_of_fields = 3
    STW70_LP, STW70_FR = clean_data(file_path, expected_number_of_fields)

    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
    df = pd.DataFrame([x.split(',') for x in STW70_LP]) # Convert the data into a DataFrame
    df = df.applymap(str.strip)  # Remove leading and trailing whitespace characters
    

    df = df.astype(float)  # Convert the data into float
    print(df)

    # Create a 2D plot
    plt.scatter(df[0], df[1])

    plt.xlabel('X')
    plt.ylabel('Y')

    # plt.show()

    # Seperate the data in to three parts according to the value of the second column
    # First part is the data with the value of the second column larger than 3 and smaller than 18
    # Second part is the data with the value of the second column larger than 24 and smaller than 39
    # Third part is the data with the value of the second column larger than 45 and smaller than 60
    df_low = df[(df[1] > 3) & (df[1] < 19+1)]
    df_middle = df[(df[1] > 24) & (df[1] < 39+1)]
    df_high = df[(df[1] > 45) & (df[1] < 60+1)]

    print(df_low)
    # Create a 2D plot for df_low
    plt.scatter(df_low[0], df_low[1], color='red')
   

    # Calculate the mean value of the first column for df_low everch ONE unit of the second column
    mean_values_low = []
    standard_deviation_low = []
    for i in range(3, 18+1):  
        df_low_1mmseg = df_low[(df_low[1] > i) & (df_low[1] < i+1)][0]
        mean_value_low_y= df_low[(df_low[1] > i) & (df_low[1] < i+1)][1].mean()
        print(f'The value of mean_value_low_y is {mean_value_low_y}')
        
        mean_value = df_low_1mmseg.mean()
        std_dev = df_low_1mmseg.std()
        mean_values_low.append(mean_value)
        standard_deviation_low.append(std_dev)
    print(mean_values_low) 
    print(standard_deviation_low)

    # Plot the mean value and standard deviation for df_low
    plt.errorbar(mean_values_low, range(3, 18+1), xerr=standard_deviation_low, fmt='o', color='red')

    # Calculate the mean value of the first column for df_middle everch ONE unit of the second column
    mean_values_middle = []
    standard_deviation_middle = []
    for i in range(24, 39+1):  
        df_middle_1mmseg = df_middle[(df_middle[1] > i) & (df_middle[1] < i+1)][0]
        mean_value_middle_y= df_middle[(df_middle[1] > i) & (df_middle[1] < i+1)][1].mean()
        print(f'The value of mean_value_middle_y is {mean_value_middle_y}')
        
        mean_value = df_middle_1mmseg.mean()
        std_dev = df_middle_1mmseg.std()
        mean_values_middle.append(mean_value)
        standard_deviation_middle.append(std_dev)
    print(mean_values_middle) 
    print(standard_deviation_middle)

    # Plot the mean value and standard deviation for df_middle
    plt.errorbar(mean_values_middle, range(24, 39+1), xerr=standard_deviation_middle, fmt='o', color='blue')

    # Calculate the mean value of the first column for df_high everch ONE unit of the second column
    mean_values_high = []
    standard_deviation_high = []
    for i in range(45, 60+1):  
        df_high_1mmseg = df_high[(df_high[1] > i) & (df_high[1] < i+1)][0]
        mean_value_high_y= df_high[(df_high[1] > i) & (df_high[1] < i+1)][1].mean()
        print(f'The value of mean_value_high_y is {mean_value_high_y}')
        
        mean_value = df_high_1mmseg.mean()
        std_dev = df_high_1mmseg.std()
        mean_values_high.append(mean_value)
        standard_deviation_high.append(std_dev)
    print(mean_values_high) 
    print(standard_deviation_high)

    # Plot the mean value and standard deviation for df_high
    plt.errorbar(mean_values_high, range(45, 60+1), xerr=standard_deviation_high, fmt='o', color='green')
    plt.xlabel('Mean Value')
    plt.ylabel('Location')
    plt.title('Mean Value and Standard Deviation')
    plt.show()