# import statments
import math
import numpy as np
import matplotlib.pyplot as plt
import turtle
import math
import time
import matplotlib.pyplot as plt

# total Data. Column 1 is Heliocentric longitude, Column 2 is Elongation of Mars, Column 3 is East or West. 
# The rows are already paired. The first two rows are a pair of observations 687 days apart, then rows 3 and 4 are then 5 and 6 ect..
allData = [242.881,142.838,"east",
200.758,115.126,"west",
263.505,175.806,"west",
222.238,93.553,"west",
355.838,163.05,"east",
313.632,96.196,"west",
47.055,179.291,"west",
3.634,94.173,"west",
131.16,132.513,"east",
86.825,128.375,"west",
157.298,142.577,"east",
113.25,122.009,"west",
222.738,111.952,"east",
180,155.067,"west",
233.578,135.376,"east",
191.26,122.870,"west",
243.473,179.334,"east",
201.36,95.872,"west",
337.042,149.133,"east",
295.175,100.734,"west",
36.211,107.238,"east",
353.061,139.849,"west",
112.29,136.184,"east",
67.93,122.951,"west",
47.485,137.773,"east",
4.053,111.781,"west",
173.046,108.558,"east",
129.287,163.082,"west",
145.978,173.876,"east",
101.775,103.153,"west",
152.013,165.574,"east",
107.892,107.466,"west",
163.037,151.146,"east",
119.091,116.079,"west",
173,139.321,"east",
129.243,124.843,"west",
182.904,128.852,"east",
139.368,134.692,"west",
192.756,119.644,"east",
149.45,145.757,"west",
207.423,107.854,"east",
164.479,164.583,"west",
162.597,116.665,"east",
118.633,149.381,"west",
96.202,125.423,"east",
51.936,129.588,"west",
292.135,147.998,"east",
250.601,104.238,"west",
323.764,119.972,"east",
282.081,120.361,"west",
352.89,103.873,"east",
310.743,142.905,"west",
343.131,108.577,"east",
301.165,134.189,"west",
333.426,113.881,"east",
291.615,126.751,"west",
357.001,118.264,"east",
314.752,120.801,"west",
6.84,112.689,"east",
324.377,127.959,"west",
16.738,107.709,"east",
334.039,136.402,"west"]

# For calcuating the lengths we actully dont need the data for eastern or western direction.
# This functions drops the east and west data by dropping the string elements
def drop_strings_from_array(arr):
    # Initialize an empty list to store non-string items
    filtered_arr = []

    # Loop over each item in the array
    for item in arr:
        # Check if the item is not a string
        if not isinstance(item, str):
            # If not a string, append the item to the filtered list
            filtered_arr.append(item)

    return filtered_arr

# use above function, to get only numerical data to calcuate distances
dataForDistance = drop_strings_from_array(allData)

# Now we just have the helio and the elongation columns. We must split again to just have the helio Data and the elongation data

# This function work by only saving elements with odd indices effectively dropping left column
def deleteHelioData( lst):
    result = []
    for i in range(len(lst)):
        if i % 2 != 0:
            result.append(lst[i])

    return result

# use above function to only get elongation data
elongationData = deleteHelioData( dataForDistance)

# now we pair the elongation data. Note: This table already has them in row pairs

# this function pairs array neighbors.
def pair_neighbors(lst):
    # Initialize an empty list to store pairs
    pairs = []
    
    # Iterate through the list, skipping the last element
    for i in range(0, len(lst) - 1, 2):
        # Create a tuple of current and next elements and append to pairs list
        pairs.append((lst[i], lst[i + 1]))
    
    return pairs

# use above function to pair elongation dates, to be ready to use for calculation
pairedElongation = pair_neighbors(elongationData)

# Now we can implement the Math equation from presentation to calcute the radius from sun to mars for each pair of data
def MarsRadius(elongation1, elongation2):
    term1 = ((math.sin( (elongation1 - 68.526) * math.pi / 180) )/ ( (math.sin( ( (elongation1 + elongation2) - (2 * 68.526)) * math.pi / 180)) )) * 0.73216
    term1 = term1 ** 2 

    term2 = 2 * ((math.sin( (elongation1 - 68.526) * (math.pi / 180) ) )/ ( (math.sin( (( elongation1 + elongation2) - (2 * 68.526)) * (math.pi / 180) )) )) * 0.73216 * math.cos(elongation2 * math.pi / 180)

    final = math.sqrt( 1 + term1 - term2)
    return final


# Lets use the function to build an array to store all Mars distances
marsRadi = []
for elongations in pairedElongation:
    radi =  MarsRadius( elongations[0], elongations[1])
    marsRadi.append(radi)


# Now Lets find the max and min of distance in the array. Python makes this easy
def maxMinRadi(lst):
    rmin = min(lst)
    rmax = max(lst)
    return rmin, rmax

rmin, rmax = maxMinRadi(marsRadi)
#
# Lets print what we have so far
# We can print Mars radi, this will print all the distances we calcuted
print()
print("These are all the distances to Mars Calculated: ")
print(marsRadi)
print()
# We can print what the max and min where found to be
print("The Minimum and Maximum distance from Sun to Mars")
print("Minimum: " + str(rmin)+ " Maximum: " +  str(rmax))
print()

# Remember from our book we can find the semi major axis as the sum of Rmax and Rmin. Look at page 12 in Geiges if you dont remember 
a = (rmin + rmax) /2
# And then eccentricty can be calcuted too. 
eAttmpt1 = rmax/a - 1
eAttempt2 = 1 - rmin/a
print("a is", str(a))
print("Calculation eccentricty ")
print ( eAttmpt1, eAttempt2)
print()
# somehow eccentricy calcualted differnt ways yeilds still got same result of 0.0907067550866677

# now we need to do the opposite of what we did above and only save the helioData

# this time this funcitons works by only saving the data that are at an even distance, so basically drops second column
def HelioData( lst):
    result = []
    for i in range(len(lst)):
        if i % 2 == 0:
            result.append(lst[i])

    return result

# use above function to get helio data
Helio = HelioData(dataForDistance)

# unfortuantely we need to do some more data augmentaion. now do need the data for eastern or western elongation so we can draw the line
# lets first drop the heliocentric data from the original data, becasue we need to pair the elongation measurments with its east or west direction
def remove_modulo_zero_elements(arr):
    return [elem for i, elem in enumerate(arr) if i % 3 != 0]

# use above function
removedHelio = remove_modulo_zero_elements(allData)

# now Lets use our pair neighbors function again to pair the measurment with its respective direction
finalElongations =  pair_neighbors(removedHelio)


# functions for fitting Elipse, this section is not my code!------------------------------------------------------------------------------------------------
def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

# End of section that is not my code---------------------------------------------------------------------------------------------------------------------------------

# This is the function that challenged me the most even tho in hindsight it is so easy. 
# I wanted to plot the intersection of the paired lines, so this function take the starting and ending points of both lines
# and finds where the intersect. ( of course the staring point is earths position because thats where the observation begins, 
# the ending point are just far enough away so that I know lines will intersect by then)

def intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# this is a helper function to split x and y Points. This is needed because the function to approximate ellipse does not seperate x, y points
def split_coordinates(coordinates):
    # Convert the list of tuples to a NumPy array
    coordinates_array = np.array(coordinates)
    
    # Split the array into separate arrays for x and y coordinates
    x_points = coordinates_array[:, 0]
    y_points = coordinates_array[:, 1]
    
    return x_points, y_points

# initalize array to save mars location. Of course these locations will be in terms of pixel locations not any real distances
marsPoints = []

# Ok now comes the biggest part of the code. The visualization. Dont be afraid of how ugly it looks, comments will explain all.
def plot_lines(degrees, turn_angles):
    # Set up the turtle gird and screen
    turtle.setup(900, 900)
    window = turtle.Screen()
    window.bgcolor("white")
    # Title of window
    turtle.title("Lines Leaving Points on Unit Circle with Corresponding Turn Angles")
    t = turtle.Turtle()
    t.speed(1)  # Set the turtle speed to fastest
    
    # Draw the unit circle. This is Earths orbit we assume its sufficently circular
    t.penup()
    # we start drawing circle at the bottom 
    t.goto(0, -200)
    t.pendown()
    t.circle(200)
    
    # we need a counter to keep track of when a pair starts over again
    count  = 0 
    
    # Draw the lines leaving each point on the unit circle with corresponding turn angles.
    # degree is heliocentric degree, turn angles is how much to turn repsectively toward the dirction of mars
    for degree, turn_angle in zip(degrees, turn_angles):
        # we need to make sure the heading is always 0 initaily becuase we need a base to start turning the turtle the right amount 
        t.setheading(0)
        
        # this just sets the shape
        t.shape("classic")
        t.shapesize(1)

        # now go to the point on the circle, this is earths position
        x = math.cos(math.radians(degree)) * 200
        y = math.sin(math.radians(degree)) * 200
        
        # set color to blue cus its earth, and stamp this position
        t.color("blue")
        t.penup()
        t.goto(x, y)
        t.left(degree)
        t.dot(5)  # Dot at the point

        # now change color to red cus its mars, to draw arrows
        t.color("red")

        # I just set the speed to be faster that way, its not slow the entire time
        if (count == 8):
            t.speed(90)

        turned = degree
        # Turn turtle by the corresponding angle and plot line leaving the point
        if (turn_angle[1] == 'east'): # if its east we go right, and we need to subtract it from 180, because turtle works by adjusting direction not reseting direction
            t.right( 180 - turn_angle[0]) # we need to change the direction the differnece of the orginal heading and the elongation direction
            turned += (180 + turn_angle[0])
            
        # same idea but we turn left if west
        if (turn_angle[1] == 'west'):
            t.left( 180 - turn_angle[0])
            turned += (180 - turn_angle[0])
            

        # now go forward in the direction turtle is facing. By this point we are facing the correct direction of mars    
        t.pendown()
        t.forward(100)
        # I only go 100 and make sure im still facing the right direction. so by set heading to turned Im still going the same direction
        t.setheading(turned)
        t.forward(200)

        # now that turtle has gone 300 we get the postion. this is the ending position for the intersection function
        xnew,ynew = t.pos()
        
        # this line actually wont show the turtle moving we just make sure its at the right spot
        t.goto(xnew,ynew)
        t.stamp()
        t.penup()

        # this code handles if its the first or second pair in the observation
        if (count % 2 == 0):
            l1 = []
            # update the initial and final points for the first line in the observations
            initial1 = (x,y)
            final1 = (xnew, ynew)
            l1.append(initial1)
            l1.append(final1)

        if (count % 2 != 0 ):
            # update the inital and final points for the second line in the observations
            l2 = []
            initial2 = (x,y)
            final2 = (xnew,ynew)
            l2.append(initial2)
            l2.append(final2)

            # now get the points of the intersection from helper function
            points = intersection(l1,l2)

            # stamp the green dot
            t.color("green")
            t.shape("circle")
            t.shapesize(.3)
            t.goto(points)
            t.stamp()
            marsPoints.append(points)
        

        # Return turtle to center to begin loop again
        t.penup()
        t.goto(0, 0)
        count += 1
    
    # Hide the turtle and now graph the ellipse of best fit. This is found from functions 
    xpoints, ypoints = split_coordinates(marsPoints)

    # find the equation for the ellipse based on the given points
    coeffs = fit_ellipse(xpoints, ypoints)
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)

    # now get the x,y points of the predicted ellipse
    x, y = get_ellipse_pts((x0, y0, ap, bp, e, phi))


    # go to first points in the array containing the points of fitted ellipse 
    t.penup()
    t.goto(x[0], y[0])
    t.pendown()

    # i set color to orange, you could change it to whatever you want
    t.color("orange")  # Set color to orange
    for i in range(1, len(x)):
        t.goto(x[i], y[i])

    t.hideturtle()

    # finally we are done
    turtle.done()

# use above function to makethe visual
plot_lines(Helio, finalElongations)