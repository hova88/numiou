/*
 ref: OpenPCdet
 url:https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp
*/

#include <stdio.h>
#include <math.h>

#include <vector>

#include "iou.h"


inline double min(double a, double b) {
    return a > b ? b : a;
}

inline double max(double a, double b) {
    return a > b ? a : b;
}

const double EPS = 1e-8;

struct Point {
    double x, y;
    Point() {}
    Point(double _x, double _y) {
        x = _x, y = _y;
    }
    
    void set(double _x, double _y) {
        x = _x; y = _y;
    }

    Point operator +(const Point &b) const {
        return Point(x + b.x, y + b.y);
    }
    Point operator -(const Point &b) const {
        return Point(x - b.x, y - b.y);
    }

};

inline double cross(const Point &a, const Point &b) {
    return a.x * b.y - a.y * b.x;
}

inline double cross(const Point &p1, const Point &p2, const Point &p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2) {
    int ret = min(p1.x,p2.x) <= max(q1.x,q2.x) &&
              min(q1.x,q2.x) <= max(p1.x,p2.x) &&
              min(p1.y,p2.y) <= max(q1.y,q2.y) &&
              min(q1.y,q2.y) <= max(p1.y,p2.y);
    return ret;
}

inline int check_in_box2d(const double *box, const Point &p) {
    // params: [x,y,z,dx,dy,dz,yaw]
    const double MARGIN = 1e-2;

    double center_x = box[0], center_y = box[1];
    // rotate the point in the opposite direction of box
    double angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);
    double rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    double rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3]/2 + MARGIN && fabs(rot_y) < box[4]/2 + MARGIN);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans) {
    // fast exclusion
    if (check_rect_cross(p0,p1,q0,q1) == 0) return 0;

    // check cross standing
    double s1 = cross(q0, p1, p0);
    double s2 = cross(p1, q1, p0);
    double s3 = cross(p0, q1, q0);
    double s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    double s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    }
    else {
        double a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        double a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        double D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D; 
    }

    return 1;
}

inline void rotate_around_center(const Point &center, const double angle_cos, const double angle_sin, Point &p) {
    double new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    double new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

inline int point_cmp(const Point &a, const Point &b, const Point &center) {
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

inline double box_overlap(const double *box_a, const double *box_b) {
    // params: box_a,box_b (7) [x,y,z,dx,dy,dz,yaw]
    /*
          -----2                 3----2
         |  .  |  --corners-->  |  .  |
         1-----                 0----1 
    */

    double a_angle = box_a[6], b_angle = box_b[6];
    double a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    double a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    double a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    double b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    double b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    double a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    double b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon, antoclockwise
    Point temp;
    for (int j = 0; j < cnt - 1; j++){
        for (int i = 0; i < cnt - j - 1; i++){
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)){
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // get the overlap areas
    double area = 0;
    for (int k = 0; k < cnt - 1; k++){
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

inline double iou_bev_kernel(const double *box_a, const double *box_b) {
    // params: box_a,box_b (7) [x,y,z,dx,dy,dz,yaw]
    double sa = box_a[3] * box_a[4];
    double sb = box_b[3] * box_b[4];
    double s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

py::array_t<double> iou_bev(py::array_t<double> boxes_a_numpy, py::array_t<double> boxes_b_numpy) {
    // params boxes_a_numpy: (N,7) [x,y,z,dx,dy,dz,yaw]
    // params boxes_b_numpy: (M,7) [x,y,z,dx,dy,dz,yaw]
    // params ans_iou_numpy: (N,M) 

    py::buffer_info buf_a = boxes_a_numpy.request(), buf_b = boxes_b_numpy.request();

    if (buf_a.shape[1] != 7 || buf_b.shape[1] != 7)
        throw std::runtime_error("shape of dimensions must be (N,7)");

    ssize_t num_boxes_a = buf_a.shape[0], num_boxes_b = buf_b.shape[0];
    
    // No pointer is passed, so Numpy will allocate the buffer
    auto iou = py::array_t<double>(num_boxes_a * num_boxes_b);
    py::buffer_info buf_iou = iou.request();

    const double* boxes_a_ptr = static_cast<double*>(buf_a.ptr);
    const double* boxes_b_ptr = static_cast<double*>(buf_b.ptr);
    double* iou_ptr = static_cast<double*>(buf_iou.ptr);


    for (int i = 0; i < num_boxes_a; i++) {
        for (int j = 0; j < num_boxes_b; j++) {
            iou_ptr[i * num_boxes_b + j] = iou_bev_kernel(boxes_a_ptr + i * 7, boxes_b_ptr + j * 7);
        }
    }

    return iou.reshape({num_boxes_a,num_boxes_b});
}