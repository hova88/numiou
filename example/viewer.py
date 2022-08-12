import numpy as np 
import open3d as o3d 
from open3d import geometry

# utils function
def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def box2corners(box):
    """
    box: [x, y, z, dx, dy, dz, yaw]
    """

    # 8 corners: np.array = n*8*3(x, y, z)
    #         7 -------- 6
    #        /|         /|
    #       4 -------- 5 .
    #       | |        | |
    #       . 3 -------- 2
    #       |/         |/
    #       0 -------- 1

    #             ^ dx(l)
    #             |
    #             |
    #             |
    # dy(w)       |
    # <-----------O

    x,y,z,l,w,h,yaw = box[0],box[1],box[2],box[3],box[4],box[5],box[6]
    # 3d bounding box corners
    Box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    R = rotz(yaw)
    corners_3d = np.dot(R, Box)  # corners_3d: (3, 8)

    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    return np.transpose(corners_3d)

def create_box_from_corners(corners, color=None):
    """
	corners: 8 corners(x, y, z)
	corners: array = 8*3
	#         7 -------- 6
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       0 -------- 1
	"""
    # 12 lines in a box
    lines = [
        [0, 1],[1, 2],[2, 3],[3, 0], # bottom
        [4, 5],[5, 6],[6, 7],[7, 4], # upper
        [0, 4],[1, 5],[2, 6],[3, 7], # 4 sides
    ]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def get_pts_in_box3d(points , boxes):
    """
    :param 
        points: (#points,3/4)
        boxes: (N,7)
    :return:
        pt_in_box3d: (n, 3/4)
    """
    condition = (np.abs(points[:,0])  < 0.2) \
              * (np.abs(points[:,1])  < 0.2) \
              * (np.abs(points[:,2])  < 5) 

    for box in boxes:
        cx,cy,cz,dx,dy,dz,rz = box[:7]
        
        local_z = points[:,2] - cz
        cosa = np.cos(-rz) 
        sina = np.sin(-rz)
        local_x = (points[:,0] - cx) * cosa + (points[:,1] - cy) * (-sina)
        local_y = (points[:,0] - cx) * sina + (points[:,1] - cy) * cosa
        
        # Finding the intersection 
        condition += (np.abs(local_z)  < dz/2) \
                   * (np.abs(local_y ) < dy/2) \
                   * (np.abs(local_x ) < dx/2) 

    return points[condition] , points[~condition]

# create wrap
def create_box(box, color=None):
    """
    box: list(8) [ x, y, z, dx, dy, dz, yaw]
    """
    box_corners = box2corners(box)
    box = create_box_from_corners(box_corners, color)
    return box


def create_coordinate(size=2.0, origin=[0, 0, 0]):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0]
    )
    return mesh_frame

# o3d class 

class PointCloudVis(object):
    def __init__(self):
        self.colorbar = [[1,0,0],[0,1,0],[0,0,1]]

    @staticmethod
    def get_points(vis , pts , color= [0.5 , 0.5 , 0.5]):
        colors = [color] * pts.shape[0]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts[:,:3])
        pc.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pc)

    @staticmethod
    def get_boxes(vis , boxes , color= [0.5 , 0.5 , 0.5]):
        boxes_o3d = []
        for box in boxes:
            box_o3d = create_box(box , color)
            boxes_o3d.append(box_o3d)
        [vis.add_geometry(element) for element in boxes_o3d] 

    def DRAW_CLOUD(self,clouds_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for cloud in clouds_list:
            # add point cloud with color
            self.get_points(vis , cloud )
            # add coordinate frame
            coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
            # drop the window
            vis.get_render_option().point_size = 2
            vis.run()
            vis.destroy_window()

    def DRAW_BOXES(self , boxes_a , boxes_b):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        center_a = boxes_a[:,:3]
        center_b = boxes_b[:,:3]
        self.get_points(vis , boxes_a[:,:3], color=[1,0,0])
        self.get_boxes(vis , boxes_a, color=[1,0,0])


        self.get_points(vis , boxes_b[:,:3], color=[0,0,1])
        self.get_boxes(vis , boxes_b, color=[0,0,1])
        
        # coordinate frame
        coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)        
        
        # drop the window
        vis.get_render_option().point_size = 2
        vis.run()
        vis.destroy_window()
        


if __name__ == "__main__":
    import numiou as niou
    boxes_a = np.array([
        # [-4.56641, -27.0469, 0.1, 3.421875, 2.465332, 0.1,  1.45996],
        [ 1.0,   16.7188, 0.1, 3.392578, 2.406006, 0.1,  0.65380], 
        # [-4.57812,  27.0938, 0.1, 3.416748, 2.372070, 0.1,  0.77002],
        # [-0.43676,  18.8900, 0.1, 3.386475, 2.377930, 0.1, -1.43555],
        # [ 5.03125,  15.4141, -0.52832, 3.388184, 2.375488, 1.07160, -1.50293],
        # [-1.21582,  10.5156, -0.41040, 3.373291, 2.371338, 0.99881, -0.61425], 
        # [-0.14209,  24.3438, -0.14904, 3.388428, 2.367920, 0.77050, -1.02148],
        # [9.77344 , -9.17188, -1.64551, 3.363037, 2.379883, 0.75560, -0.20227], 
        # [11.1484 ,  3.42773, -1.23535, 3.396240, 2.401611, 0.76975, -0.27465],
        # [16.7781 , -9.61719, -1.65234, 3.369873, 2.383301, 0.74388, -0.27563],  
    ])

    boxes_b = np.array([
        [ 0.33056,  24.375 , 0.1, 3.410156, 2.371826, 0.1, -0.70996], 
        [ 1.0,   16.7188, 0.1, 3.392578, 2.406006, 0.1,  0.65380], 
        # [ 1.6875,   16.7188, 0.1, 3.392578, 2.406006, 0.1,  0.65380], 
        [-13.4141,  30.1875, 0.1, 3.367432, 2.368408, 0.1,  0.68798], 
        # [-1.38965,  24.0625, -0.18713, 3.356445, 2.387207, 0.84293,  0.80712], 
        # [-10.7188,  9.79688, -0.54248, 3.368652, 2.374268, 1.27221,  0.30835],
        # [-4.98828, -26.1875, -1.92871, 3.387695, 2.404541, 0.95586,  1.51855],
        # [11.3906 ,  3.65625, -1.17285, 3.404297, 2.402344, 0.81262, -0.19824], 
        # [9.28125 , -9.47656, -1.68848, 3.395020, 2.404297, 0.73809, -0.21447], 
        # [19.4688 , -9.82031, -1.67969, 3.400391, 2.418701, 0.76937, -0.11938], 
        # [18.7969 , -10.1328, -1.71191, 3.392090, 2.395264, 0.71504, -0.19311], 
        # [14.0156 , -10.6250, -1.72363, 3.375977, 2.394775, 0.75930, -0.15759], 
    ])
    print(boxes_a.shape)
    # boxes_a = boxes_a.astype(float)
    # boxes_b = boxes_b.astype(float)
    iou = niou.bev(boxes_a,boxes_b)#.astype(float)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print(iou)
    # print(iou.shape)

    # V = PointCloudVis()
    # V.DRAW_BOXES(boxes_a , boxes_b) 