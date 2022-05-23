#!/usr/local/bin/python3
#
# Authors:
# - Harsh Srivastava <hsrivas>
# - Ritwik Budhiraja <rbudhira>
# - Yash Shah <yashah>
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
import numpy as np
from numpy.core.defchararray import count
from numpy.core.fromnumeric import product, size
from numpy.core.numeric import zeros_like
from numpy.lib.shape_base import column_stack
from scipy.ndimage import filters
import sys
import imageio
import math

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = np.array(input_image.convert('L'))
    filtered_y = np.zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return np.sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

def normalize_array(a, mi, ma):
    return (a - (mi * np.ones(a.shape))) / ma

def take_till_diff_abs(ls, condition_lambda, key=None):
    result = [ls[0]]
    for i in range(1, len(ls)):
        get = lambda i: ls[i] if key == None else key(ls[i])
        diff = abs(get(i - 1) - get(i))
        if condition_lambda(diff):
            result.append(ls[i])
        else:
            break
    return result

def apply_mask(mask, matrix, r, c):
    mask_center = (mask.shape[0] // 2, mask.shape[1] // 2)

    mask_value = 0.0

    for i_r in range(mask.shape[0]):
        row_diff = i_r - mask_center[0]
        matrix_r = r + row_diff

        if matrix_r not in range(matrix.shape[0]):
            continue

        for j_c in range(mask.shape[1]):
            col_diff = j_c - mask_center[1]
            matrix_c = c + col_diff

            if matrix_c not in range(matrix.shape[1]):
                continue

            mask_value += matrix[matrix_r][matrix_c] * mask[i_r][j_c]

    return mask_value / np.sum(np.abs(mask))

def apply_averaging_mask(matrix):
    averaging_mask = np.array([[ +1],
                               [ +1],
                               [ +8],
                               [+16],
                               [+50],
                               [100],
                               [+50],
                               [+16],
                               [ +8],
                               [ +1],
                               [ +1]]) / 100

    masked = np.zeros_like(edge_strength)

    for column in range(edge_strength.shape[1]):
        target_slice = np.array([[x] for x in edge_strength[:, column]])

        for row in range(edge_strength.shape[0]):
            masked[row][column] = apply_mask(averaging_mask, target_slice, row, 0)

    return masked

def find_line_simple(matrix, limiting_line=[]):
    # The most basic way to find the line is using the edge matrix
    #   - If a pixel is high, it is more likely to be on an edge boundary

    shape = matrix.shape
    global_min = np.min(matrix)
    global_max = np.max(matrix)

    suspicion_cache = []

    line_result = []

    matrix = apply_averaging_mask(matrix)

    for column in range(shape[1]):
        # Calculate limiting conditions, if any
        row_start, row_end = 0, shape[0]
        if column < len(limiting_line):
            row_start = limiting_line[column] + 10

        # Take the original target column slice of matrix and process it
        target_slice_original = matrix[row_start:row_end, column]

        # Normalize on global max/min (all values between 0.0 and 1.0)
        #   - Here since the data is normalized we can assume that in every slice
        #     the top brightest pixels have the highest chance to be the boundary
        #     in the edge strength matrix
        target_slice = normalize_array(target_slice_original, global_min, global_max)

        # Convert the slice into indexed pairs
        target_slice = [(i + row_start, target_slice[i]) for i in range(len(target_slice))]

        # Sort the pixels with indices, this takes us closer to what usually 'argmax' would do
        #   - Basically, if a pixel is high value, it has a high chance to be an edge
        #     on the Sobel filtered edge strength passed as the matrix
        target_slice = sorted(target_slice, key=lambda x: -x[1])

        # Filter list of pixels over 0.5
        #   - The reason to choose 0.5 is because most of the time
        target_slice = list(filter(lambda x: x[1] > 0.5, target_slice))

        # Just in case there are no points found, then we normalize locally and re-filter at 0.5
        if len(target_slice) == 0:
            # Normalize on local max/min (all values between 0.0 and 1.0)
            #   - Here the data's max will be mapped to 1.0 and min to 0.0
            target_slice = normalize_array(target_slice_original, np.min(target_slice_original), np.max(target_slice_original))

            # Convert the slice into indexed pairs
            target_slice = [(i + row_start, target_slice[i]) for i in range(len(target_slice))]

            # Sort the pixels with indices
            target_slice = sorted(target_slice, key=lambda x: -x[1])

            # Filter list of pixels over 0.5 (manually provided threshold)
            target_slice = list(filter(lambda x: x[1] > 0.5, target_slice))

        # To enhance the 'argmax' logic we simply take the values until the difference between
        # any consecutive values is higher than 0.2 (manually provided cutoff)
        suspected_pairs = take_till_diff_abs(target_slice, lambda x: x <= 0.2, key=lambda x: x[1])

        # We sort them again based on their index
        suspected_pairs = sorted(suspected_pairs, key=lambda x: x[0])

        # And simply take the first one as the required (index, value) pair
        suspicion_cache.append(suspected_pairs)

        # Just add 1 to the index since the calculations were done for the difference.
        # Hence, the actual result is one row below.
        line_result.append(suspected_pairs[0][0] + 1)

    # Return the result
    return line_result

# HMM
def calculate_edge_point_probability(target_slice):
    difference_mask = np.array([[-10],
                               [- 8],
                               [- 8],
                               [- 4],
                               [- 2],
                               [  0],
                               [+ 2],
                               [+ 4],
                               [+ 4],
                               [+ 8],
                               [+10]]) / 10
    products = np.array([x[0] * x[1] for x in zip(target_slice, difference_mask)])
    # products = products / np.sum(products)
    return np.sum(products)

def find_max_successor(matrix, v_table, r, c, span=10):
    if c < matrix.shape[1] - 1:
        from_r = 0 if (r - span) < 0 else (r - span)
        to_r = (matrix.shape[0] - 1) if (r + span) > (matrix.shape[0] - 1) else (r + span)
        target_slice = matrix[from_r : to_r + 1, c + 1]
        max_i = np.argmax(target_slice)
        return (from_r + max_i, v_table[r][c - 1] * math.sqrt(abs(max_i - span) / 10.0))
    return (r, matrix[r, c])

def find_line_hmm(image_matrix, edge_matrix, limiting_line=[], force=None):
    # This implementation is based on the material provided during the class activity for Viterbi algorithm
    # Ref: https://iu.instructure.com/courses/2027431/files/128919777?module_item_id=25300577

    v_table = np.zeros_like(edge_matrix)
    lookup = np.zeros_like(edge_matrix).astype('int')

    initial_probabilities = image_matrix[:, 0]
    initial_probabilities = initial_probabilities / np.max(initial_probabilities)

    image_matrix = image_matrix / np.max(image_matrix)

    if force != None:
        image_matrix[force[0], force[1]] = 1.0

    start_r = max(0, limiting_line[0] + 10 if len(limiting_line) > 0 else -math.inf)
    for r in range(start_r, edge_matrix.shape[0]):
        v_table[r][0] = initial_probabilities[r]

    c = 0
    for c in range(1, edge_matrix.shape[1]):
        if force != None and force[1] == c + 1:
            lookup[r][c], v_table[r][c] = force[0], 1.0
            continue

        start_r = max(0, limiting_line[c] + 10 if len(limiting_line) > 0 else -math.inf)
        for r in range(start_r, edge_matrix.shape[0]):
            lookup[r][c], v_table[r][c] = find_max_successor(edge_matrix, v_table, r, c)

    viterbi_sequence = [-1] * edge_matrix.shape[1]
    limit_last = 0 if len(limiting_line) == 0 else (limiting_line[edge_matrix.shape[1] - 1] + 10)
    viterbi_sequence[edge_matrix.shape[1] - 1] = limit_last + np.argmax([v_table[tag_i][c] for tag_i in range(limit_last, edge_matrix.shape[0])])

    for i in range(edge_matrix.shape[1] - 2, -1, -1):
        viterbi_sequence[i] = lookup[viterbi_sequence[i + 1]][i + 1]

    return viterbi_sequence

# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = np.array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', np.uint8(255 * edge_strength / (np.amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    airice_simple = find_line_simple(edge_strength)
    airice_hmm = find_line_hmm(image_array, edge_strength)
    airice_feedback = find_line_hmm(image_array, edge_strength, force=gt_airice)

    icerock_simple = find_line_simple(edge_strength, limiting_line=airice_simple)
    icerock_hmm = find_line_hmm(image_array, edge_strength, airice_hmm)
    icerock_feedback = find_line_hmm(image_array, edge_strength, airice_feedback, force=gt_icerock)

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
