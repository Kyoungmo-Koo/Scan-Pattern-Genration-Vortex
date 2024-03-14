AMP = 0.5

def process_line(line):
    # Split the line into two hexadecimal numbers (8 digits each)
    num1 = int(line[:4], 16)
    num2 = int(line[4:], 16)

    # Perform the required operations
    num1 = hex(int((num1 - 32768) * AMP + 32768))[2:].upper()
    num2 = hex(int((num2 - 32768) * AMP + 32768))[2:].upper()

    # Concatenate the hexadecimal numbers and return
    return num1 + num2

# Read the input file and process each line
with open("raster_hexadecimal.txt", "r") as infile:
    lines = infile.readlines()

processed_lines = [process_line(line.strip()) for line in lines]

# Write the processed lines to a new file
with open("raster_hexadecimal_half.txt", "w") as outfile:
    outfile.write("\n".join(processed_lines))
