import numpy as np

def get_points_on_triangle(v, n):
    """
    Give n random points uniformly on a triangle.

    The vertices of the triangle are given by the shape
    (3, 3) array *v*: one vertex per row
    """
    x = np.sort(np.random.rand(3, n), axis=0)
    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v

def generate_points_on_mesh(verts, faces, num_samples_triangle = 10):
    
    num_faces = faces.shape[0]
    num_points = num_faces * num_samples_triangle
    points_on_mesh = np.zeros((num_points, 3), dtype=np.float32)
    
    for idx_tri, cur_face in enumerate(faces):

        cur_verts = verts[cur_face]
        points_on_triangle = get_points_on_triangle(cur_verts, num_samples_triangle)
        
        idx_start = idx_tri * num_samples_triangle
        idx_end = (idx_tri + 1)* num_samples_triangle
        points_on_mesh[idx_start:idx_end] = points_on_triangle
        
    return points_on_mesh