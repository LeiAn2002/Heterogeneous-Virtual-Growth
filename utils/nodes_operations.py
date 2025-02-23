import numpy as np
from utils.coord_transformation import cartesian2polar, polar2cartesian


def to_acute(angle_rad):
    angle_deg = np.degrees(angle_rad)
    acute_angle = abs(angle_deg) % 180
    if acute_angle > 90:
        acute_angle = 180 - acute_angle
    angle = np.radians(acute_angle)
    return angle


def get_center_radius(point1, point2):
    mid_point = (point1 + point2) / 2
    lenth = np.linalg.norm(point1 - point2)
    angle_with_horizontal = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    angle_with_vertical = np.acrctan2(point2[0] - point1[0], point2[1] - point1[1])
    acute_horizontal = to_acute(angle_with_horizontal)
    acute_vertical = to_acute(angle_with_vertical)
    angle = acute_horizontal if acute_horizontal < acute_vertical else acute_vertical
    radius = lenth / (2 * np.sin(angle))
    distance = np.sqrt(radius ** 2 - (lenth / 2) ** 2)
    x1 = mid_point[0] + distance * (point1[1] - point2[1]) / lenth
    x2 = mid_point[0] - distance * (point1[1] - point2[1]) / lenth
    x_center = x1 if np.abs(x1) < np.abs(x2) else x2
    y1 = mid_point[1] + distance * (point2[0] - point1[0]) / lenth
    y2 = mid_point[1] - distance * (point2[0] - point1[0]) / lenth
    y_center = y1 if np.abs(y1) < np.abs(y2) else y2
    center = np.array([x_center, y_center])
    return center, radius


def arc_points(point1, point2, n):
    center, radius = get_center_radius(point1, point2)
    theta1 = np.arctan2(point1[1] - center[1], point1[0] - center[0])
    theta2 = np.arctan2(point2[1] - center[1], point2[0] - center[0])
    if theta2 < theta1:
        theta2 += 2 * np.pi
    angles = np.linspace(theta1, theta2, n)
    arc_list = [[center[0] + radius * np.cos(theta), center[1] + radius
                * np.sin(theta), 0] for theta in angles]
    return arc_list


def rotate_nodes(coords, rotation):
    """Rotate nodes in the x-y plane."""
    coord_x, coord_y, coord_z = coords.T
    rho, phi = cartesian2polar(coord_x, coord_y)
    phi += np.pi/2 * rotation
    coord_x, coord_y = polar2cartesian(rho, phi)
    return np.vstack((coord_x, coord_y, coord_z)).T
